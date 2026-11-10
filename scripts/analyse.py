"""Stages 7 and 8: the report, from the saved test predictions of the chosen runs (3 seeds for each arm).

    python scripts/analyse.py --runs-root .../runs --data-root ... --out results

Writes results/report.md, numbers.json and recovered_share.png. It needs no GPU, and a second run gives the
same text. With --samples N it also draws N test images with the ground truth and each arm's detections.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from truckdet.data.miotcd import TRUCK_FAMILY, category_id
from truckdet.evaluation import load_predictions
from truckdet.report import analyse
from truckdet.runs import final_runs, split_ground_truth
from truckdet.sheets import contact_sheet, detection_image
from truckdet.training import IMAGE_DIR


def draw_samples(runs_root, data_root, splits_dir, out_dir, n: int, split: str = "test", seed: int = 0) -> None:
    """For `n` random images of the split that have a truck-family box: one sheet with a tile for each arm
    (seed 0), the ground truth in white and the arm's detections in its colour."""
    gt = split_ground_truth(data_root, splits_dir, split)
    family = {category_id(c) for c in TRUCK_FAMILY}
    with_truck = sorted({a["image_id"] for a in gt["annotations"] if a["category_id"] in family})
    picks = sorted(np.random.default_rng(seed).choice(with_truck, size=min(n, len(with_truck)), replace=False).tolist())
    images = {im["id"]: im for im in gt["images"]}
    predictions = {}
    for arm, dirs in final_runs(runs_root, seeds=(0,)).items():
        pred, _ = load_predictions(dirs[0] / f"pred_{split}.npz")
        predictions[arm] = (pred, {int(i): k for k, i in enumerate(pred["image_ids"])})
    for image_id in picks:
        anns = [a for a in gt["annotations"] if a["image_id"] == image_id]
        with Image.open(Path(data_root) / IMAGE_DIR / images[image_id]["file_name"]) as im:
            image = im.convert("RGB")
        tiles = []
        for arm, (pred, row) in predictions.items():
            k = row[image_id]
            tiles.append(detection_image(image, anns, {arm: (pred["scores"][k], pred["boxes"][k])}))
        contact_sheet(tiles, [f"arm {arm}" for arm in predictions], columns=2, tile_size=(640, 480),
                      out_path=Path(out_dir) / "samples" / f"{images[image_id]['miotcd_id']}.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=0, help="draw this many sample images (they hold dataset images: "
                        "read the licence before you publish them)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.samples:
        draw_samples(args.runs_root, args.data_root, args.splits_dir, args.out, args.samples, args.split)
    print(analyse(args.runs_root, args.data_root, args.splits_dir, args.out, args.split, args.bootstrap,
                  tuple(args.seeds), cache=not args.no_cache))


if __name__ == "__main__":
    main()
