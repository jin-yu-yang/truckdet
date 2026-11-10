"""Train one run of an arm, then score its final checkpoint on the val split.

    python scripts/train.py --config configs/arm_b.yaml --lr 3e-4 --seed 0 ...      # a learning-rate run
    python scripts/train.py --config configs/arm_f.yaml --lr 3e-5 --seed 0 ...      # arm F: --lr is the body rate
    python scripts/train.py --config configs/arm_b.yaml --selected --seed 1 ...     # the chosen rate, another seed

Re-running resumes from <output-root>/<run name>/last.pt; a finished run trains nothing.
"""

import argparse
from pathlib import Path

import torch

from truckdet.config import for_run, load_config
from truckdet.runs import evaluate_run, read_selection
from truckdet.training import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, type=Path, help="holds annotations.json and train/")
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument("--lr", type=float, help="the rate to try: all trainable parameters (H, D, B) or the body (F)")
    rate.add_argument("--selected", action="store_true", help="use the rate in <output-root>/selection.json")
    parser.add_argument("--head-lr", type=float, help="arm F: the rate of the heads (default: the rate that H chose)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="override the config (checks only)")
    parser.add_argument("--batch-size", type=int, help="override the config; batch-size x accum-steps stays 16")
    parser.add_argument("--accum-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-eval", action="store_true", help="do not score the val split after training")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    for key in ("batch_size", "accum_steps", "num_workers"):
        if getattr(args, key) is not None:
            setattr(cfg.train, key, getattr(args, key))
    selection = read_selection(args.output_root)
    if args.selected:
        if cfg.arm not in selection:
            raise SystemExit(f"no learning rate is chosen for arm {cfg.arm} yet; run scripts/select_lr.py")
        lr, body_lr = selection[cfg.arm]["lr"], selection[cfg.arm]["body_lr"]
    elif cfg.arm == "F":
        head_lr = args.head_lr if args.head_lr is not None else selection.get("H", {}).get("lr")
        if head_lr is None:
            raise SystemExit("arm F needs the rate that arm H chose: run scripts/select_lr.py for H, or pass --head-lr")
        lr, body_lr = head_lr, args.lr
    else:
        lr, body_lr = args.lr, None
    cfg = for_run(cfg, lr, args.seed, body_lr)
    run_dir = run_training(cfg, args.data_root, args.splits_dir, args.output_root, args.device, args.epochs)
    if not args.no_eval:
        result = evaluate_run(run_dir, args.data_root, args.splits_dir, "val", args.device)
        print(f"[eval] {run_dir.name}: val AP {result['AP']:.4f}, AP50 {result['AP50']:.4f}, "
              f"merge difference {result['merge_max_abs_diff']:.1e}, skipped batches {result['skipped_batches']}, "
              f"steps the fp16 scaler skipped {result.get('scaler_skips', 0)}")


if __name__ == "__main__":
    main()
