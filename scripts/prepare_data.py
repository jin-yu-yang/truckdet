"""Stage 1, part 1: gt_train.csv -> one COCO-format JSON, with the checks that need no model.

    python scripts/prepare_data.py --data-root /content/data/MIO-TCD-Localization --sheets-dir /content/checks

Writes <data-root>/annotations.json and counts.json, compares the counts with the reference counts of the
design (§4.5), and draws 100 random images with their boxes for the label completeness check: count the
vehicles and pedestrians that have no box. If more than 5% have no box, the labels are not complete: record
the rate and stop.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from truckdet.data.miotcd import (
    IGNORE_UNDER, REFERENCE_COUNTS, check_ignore_classes, convert, count_differences, count_notes,
)
from truckdet.runs import CHECK_FAILED
from truckdet.sheets import label_sheets
from truckdet.training import ANNOTATIONS, IMAGE_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="the extracted archive: gt_train.csv and train/")
    parser.add_argument("--eval-code", type=Path,
                        help="localization_evaluation.py of the dataset's code archive: confirm the ignore classes")
    parser.add_argument("--sheets-dir", type=Path, help="where the label-check sheets go (not into the repository)")
    parser.add_argument("--label-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="convert again if annotations.json exists")
    args = parser.parse_args(argv)

    out = args.data_root / ANNOTATIONS
    counts_path = args.data_root / "counts.json"
    if out.exists() and counts_path.exists() and not args.force:
        coco, counts = json.loads(out.read_text()), json.loads(counts_path.read_text())
        print(f"[data] {out} exists; not converted again")
    else:
        coco, counts = convert(args.data_root / "gt_train.csv", args.data_root / IMAGE_DIR, args.workers)
        out.write_text(json.dumps(coco))
        counts_path.write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    differences = count_differences(counts)
    print("[check] counts: " + ("equal to the reference counts " + json.dumps(REFERENCE_COUNTS) if not differences
                                else "DIFFERENT from the reference counts; find the cause of each difference:"))
    for line in differences:
        print("  - " + line)
    for line in count_notes(counts):
        print("[note] " + line)

    if args.eval_code:
        check_ignore_classes(args.eval_code)
        print(f"[check] ignore classes: {args.eval_code.name} lists the same {len(IGNORE_UNDER)} classes: {', '.join(IGNORE_UNDER)}")
    else:
        print("[check] ignore classes: not confirmed (pass --eval-code localization_evaluation.py)")

    if args.sheets_dir:
        ids = [im["id"] for im in coco["images"]]
        picks = np.random.default_rng(args.seed).choice(ids, size=min(args.label_images, len(ids)), replace=False)
        sheets = label_sheets(coco, sorted(picks.tolist()), args.data_root / IMAGE_DIR, args.sheets_dir)
        print(f"[check] label completeness: look at {len(sheets)} sheets in {args.sheets_dir} ({len(picks)} images). "
              "Count the vehicles and pedestrians with no box. More than 5%: record the rate and stop.")
    raise SystemExit(CHECK_FAILED if differences else 0)


if __name__ == "__main__":
    main()
