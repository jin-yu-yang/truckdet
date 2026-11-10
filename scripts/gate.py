"""The gate of design §2: is full fine-tuning far enough above heads-only for the question to have an answer?

After the learning-rate runs of H and F: AP_F - AP_H on the val split with seed 0 must be at least 0.05 AP and
at least 5 times its bootstrap standard error. If not, stop, and report that the data cannot separate the arms.
"""

import argparse
from pathlib import Path

from truckdet.runs import CHECK_FAILED, gate_text, run_gate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args(argv)
    result = run_gate(args.runs_root, args.data_root, args.splits_dir, args.bootstrap)
    print(gate_text(result))
    raise SystemExit(0 if result["passed"] else CHECK_FAILED)


if __name__ == "__main__":
    main()
