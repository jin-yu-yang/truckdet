"""Score a finished run on a split from its final checkpoint, or the start model with no training.

    python scripts/evaluate.py --run runs/b_lr3e-4_s0 --split val --data-root ...
    python scripts/evaluate.py --run runs/b_lr3e-4_s0 --split test --data-root ...   # one time for each final checkpoint
    python scripts/evaluate.py --final-runs runs --split test --data-root ...        # every final checkpoint not yet scored
    python scripts/evaluate.py --start-model --out runs/start_model --data-root ...  # the check of stage 2

The predictions are saved next to the checkpoint (pred_<split>.npz); all analysis reads that file.
"""

import argparse
from pathlib import Path

import torch

from truckdet.runs import evaluate_run, evaluate_start_model, final_runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--run", type=Path, help="a run folder with final.pt")
    what.add_argument("--final-runs", type=Path, metavar="RUNS_ROOT",
                      help="every run of a chosen learning rate (selection.json) that has no eval_<split>.json yet")
    what.add_argument("--start-model", action="store_true", help="the COCO checkpoint with the 11-class heads, no training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="--final-runs: the seeds")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="predict again; for the test split only after a code error")
    parser.add_argument("--out", type=Path, help="--start-model: the output folder")
    parser.add_argument("--model-name", default="Roboflow/rf-detr-base")
    parser.add_argument("--max-images", type=int, help="--start-model: score only the first images of the split")
    parser.add_argument("--num-workers", type=int, help="data-loader workers (default: the run's config; 2 for --start-model)")
    args = parser.parse_args(argv)

    if args.start_model:
        if args.split == "test":
            raise SystemExit("the start model is scored on the val split: the test split is for the final checkpoints")
        if args.out is None:
            raise SystemExit("--start-model needs --out")
        result = evaluate_start_model(args.model_name, args.data_root, args.splits_dir, args.out, args.split,
                                      args.device, max_images=args.max_images,
                                      num_workers=2 if args.num_workers is None else args.num_workers)
        ap50 = result["AP50 per class"]
        print(f"start model, {result['images']} {args.split} images: AP {result['AP']:.4f}, AP50 {result['AP50']:.4f}")
        print("AP50 per class: " + ", ".join(f"{c} {v:.3f}" for c, v in ap50.items()))
        present = [c for c in ("car", "bus") if ap50[c] == ap50[c]]  # NaN: the images have no box of the class
        if len(present) < 2:
            print("check of stage 2: not possible, these images have no boxes of " +
                  " and ".join(c for c in ("car", "bus") if c not in present))
            raise SystemExit(1)
        ok = all(ap50[c] > 0.02 for c in present)
        print("check of stage 2 (AP50 of car and bus is above zero): " + ("passed" if ok else
              "FAILED - an AP near zero shows an error in the box format or the resize"))
        raise SystemExit(0 if ok else 1)
    if args.final_runs:
        todo = [d for dirs in final_runs(args.final_runs, tuple(args.seeds)).values() for d in dirs
                if args.force or not (d / f"eval_{args.split}.json").exists()]
        print(f"{len(todo)} final checkpoints to score on the {args.split} split")
    else:
        todo = [args.run]
    bad = []
    for run_dir in todo:
        result = evaluate_run(run_dir, args.data_root, args.splits_dir, args.split, args.device, force=args.force,
                              num_workers=args.num_workers)
        print(f"{run_dir.name}, {result['images']} {args.split} images: AP {result['AP']:.4f}, AP50 {result['AP50']:.4f}, "
              f"AP75 {result['AP75']:.4f}, truck family {result['AP truck family']:.4f}, box AP {result['box AP']:.4f}")
        print(f"  merge difference {result['merge_max_abs_diff']:.1e} (must be below 1e-4); "
              f"skipped batches {result['skipped_batches']} (must be 0); "
              f"steps the fp16 scaler skipped {result.get('scaler_skips', 0)}", flush=True)
        if result["merge_max_abs_diff"] >= 1e-4 or result["skipped_batches"]:
            bad.append(run_dir.name)
    if bad:
        raise SystemExit(f"the checks of stage 6 failed for: {', '.join(bad)}")


if __name__ == "__main__":
    main()
