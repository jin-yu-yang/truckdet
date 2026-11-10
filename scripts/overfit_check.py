"""Stage 4: the training loop on 16 images.

    python scripts/overfit_check.py --data-root ... --output-root /content/overfit_runs

- Arm F reaches AP50 above 0.9 on the 16 images that it trained on.
- The loss of arms H, D and B decreases.
- A stopped and resumed run continues the loss curve (arm F is stopped after a third of its epochs).

The runs use batch 4 with no accumulation, so 16 images are 4 optimizer steps per epoch, and rates from the
middle of the design's lists (heads 1e-3, the body of F 3e-5). Use a new output folder (not the folder of the
real runs).
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from truckdet.config import for_run, load_config
from truckdet.data.miotcd import load_annotations, split_image_ids, subset
from truckdet.evaluation import score_predictions
from truckdet.reporting import read_jsonl
from truckdet.runs import _predict_split, load_run_model
from truckdet.training import ANNOTATIONS, resolve_loaders, run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--epochs-f", type=int, default=200)
    parser.add_argument("--epochs-other", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3, help="heads of F, and all trainable parameters of H, D, B")
    parser.add_argument("--body-lr", type=float, default=3e-5, help="all other parameters of F")
    parser.add_argument("--arms", default="FHDB")
    args = parser.parse_args(argv)

    coco = load_annotations(args.data_root / ANNOTATIONS)
    ids = split_image_ids(coco, args.splits_dir / "train.txt")[: args.images]
    gt = subset(coco, ids)
    ok = True
    for arm in args.arms:
        cfg = load_config(args.configs / f"arm_{arm.lower()}.yaml")
        cfg = replace(cfg, train=replace(cfg.train, batch_size=4, accum_steps=1, flip=False, num_workers=0))
        cfg = for_run(cfg, args.lr, 0, args.body_lr if arm == "F" else None)
        epochs = args.epochs_f if arm == "F" else args.epochs_other
        common = dict(device=args.device, epochs=epochs, train_ids=ids, log_every=0)
        if arm == "F":  # a disconnect after a third of the run, then a resume
            run_training(cfg, args.data_root, args.splits_dir, args.output_root, stop_after=max(1, epochs // 3), **common)
        run_dir = run_training(cfg, args.data_root, args.splits_dir, args.output_root, **common)
        rows = read_jsonl(run_dir / "metrics.jsonl")
        losses = [r["loss"] for r in rows]
        print(f"arm {arm}: loss {losses[0]:.3f} -> {losses[-1]:.3f} over {len(rows)} epochs; "
              f"skipped batches {sum(r['skipped'] for r in rows)}; the fp16 scaler skipped "
              f"{sum(r['scaler_skips'] for r in rows)} of {sum(r['steps'] for r in rows)} optimizer steps")
        complete = [r["epoch"] for r in rows] == list(range(epochs))
        decreased = losses[-1] < 0.9 * losses[0]
        ok &= complete and decreased
        print(f"  [{'ok' if complete else 'FAILED'}] one row for each of the {epochs} epochs")
        print(f"  [{'ok' if decreased else 'FAILED'}] the loss decreases")
        if arm == "F":
            k = max(1, epochs // 3)
            # The first epoch after the resume is near the epoch before it, not near the start of the run, and
            # the learning rate goes on down its cosine. (The tests show the exact case: the same weights.)
            continues = (abs(losses[k] - losses[k - 1]) < abs(losses[k] - losses[0])
                         and rows[k]["lr"] < rows[k - 1]["lr"])
            print(f"  [{'ok' if continues else 'FAILED'}] the resumed run continues the loss curve: epoch {k - 1} "
                  f"{losses[k - 1]:.3f}, epoch {k} (first after the resume) {losses[k]:.3f}, epoch 0 {losses[0]:.3f}; "
                  f"learning rate {rows[k - 1]['lr']:.2e} -> {rows[k]['lr']:.2e}")
            model, meta = load_run_model(run_dir, args.device)
            predictions, _ = _predict_split(model, gt, args.data_root, resolve_loaders()[1](cfg.model_name),
                                            torch.device(args.device), 8, 0)
            ap50 = score_predictions(gt, predictions)["AP50"]
            reached = ap50 > 0.9
            print(f"  [{'ok' if reached else 'FAILED'}] AP50 on the {len(ids)} training images is above 0.9: {ap50:.3f}"
                  + ("" if reached else " (if the loss still decreases, try more epochs: --epochs-f 400)"))
            ok &= continues and reached
    print("stage 4: " + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
