"""Choose the learning rate of an arm from its seed-0 runs on the val split (design §6).

Rule: the highest val AP of the final checkpoint; if two are equal to 3 decimal places, the lower rate. If the
best rate is at an end of its list, one more step in that direction is run, one time. The choice goes to
<runs-root>/selection.json. Choose H before F: the heads of F use the rate that H chose.
"""

import argparse
import json
from pathlib import Path

from truckdet.config import load_config
from truckdet.runs import select_lr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the arm's config, e.g. configs/arm_h.yaml")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="print the status as JSON (the notebook reads it)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    status = select_lr(args.runs_root, cfg)
    if args.json:
        print(json.dumps(status))
        return
    tried = ", ".join(f"{lr:g}: {ap:.4f}" for lr, ap in sorted(status.get("val_ap", {}).items()))
    if status["status"] == "chosen":
        tried = ", ".join(f"{lr}: {ap:.4f}" for lr, ap in status["val_ap"].items())
        rate = f"body {status['body_lr']:g}, heads {status['lr']:g}" if status["body_lr"] is not None else f"{status['lr']:g}"
        print(f"arm {cfg.arm}: chosen rate {rate} (val AP by rate: {tried}); the seed-0 run is {status['run']}")
    elif status["status"] == "run":
        reason = f" ({status['reason']})" if "reason" in status else ""
        print(f"arm {cfg.arm}: run these rates with seed 0 first: {', '.join(f'{lr:g}' for lr in status['lr'])}{reason}"
              + (f"; val AP so far: {tried}" if tried else ""))
    else:
        print(f"arm {cfg.arm}: {status['reason']}")


if __name__ == "__main__":
    main()
