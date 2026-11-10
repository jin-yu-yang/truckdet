"""MIO-TCD localization: class order, and `gt_train.csv` -> one COCO-format JSON.

`gt_train.csv` has no header. Columns: image_id, class, xmin, ymin, xmax, ymax in pixels. The image
ID is a string with leading zeros ("00000012"), and `train/<image_id>.jpg` is its image.

The COCO JSON keeps the ID as `miotcd_id` (a string) and uses `id = int(miotcd_id)` for pycocotools.
Category ids are 1..11 in the order of `CLASSES`; the model's class index is the category id minus 1.
"""

import ast
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

CLASSES = (
    "articulated_truck", "bicycle", "bus", "car", "motorcycle", "motorized_vehicle",
    "non-motorized_vehicle", "pedestrian", "pickup_truck", "single_unit_truck", "work_van",
)
# Vehicles that are too small to identify. A class in training, an ignore region in evaluation.
MOTORIZED_VEHICLE = "motorized_vehicle"
SPECIFIC_CLASSES = tuple(c for c in CLASSES if c != MOTORIZED_VEHICLE)
TRUCK_FAMILY = ("articulated_truck", "pickup_truck", "single_unit_truck", "work_van")

# The classes that get an ignore region on each `motorized_vehicle` box (design §4.4). This is
# `motorized_vehicle_classes` of the dataset's `localization_evaluation.py`: six classes. The design
# named seven (these and `motorcycle`) and asked to confirm the list against that file; the file does
# not have `motorcycle`. `check_ignore_classes` repeats the comparison on the real file.
IGNORE_UNDER = ("articulated_truck", "bus", "car", "pickup_truck", "single_unit_truck", "work_van")

# Design §4.1. The first is from the dataset's authors, the other two from another user's notes.
REFERENCE_COUNTS = {"images": 110_000, "boxes": 351_549, "images_with_motorized_vehicle": 20_299}


def category_id(name: str) -> int:
    return CLASSES.index(name) + 1


def categories() -> list[dict]:
    return [{"id": i, "name": name} for i, name in enumerate(CLASSES, start=1)]


def read_rows(csv_path: str | Path) -> list[tuple[str, str, float, float, float, float]]:
    """Rows of `gt_train.csv` as (image_id, class, xmin, ymin, xmax, ymax). The image ID stays a string."""
    rows = []
    with open(csv_path, newline="") as f:
        for line, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(f"{csv_path}:{line}: expected 6 columns, got {len(row)}")
            image_id, name = row[0].strip(), row[1].strip()
            if name not in CLASSES:
                raise ValueError(f"{csv_path}:{line}: unknown class {name!r}")
            rows.append((image_id, name, *(float(v) for v in row[2:])))
    return rows


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) from the image header; the pixels are not decoded."""
    with Image.open(path) as im:
        return im.size


def _coco_ids(image_ids: list[str]) -> dict[str, int]:
    """int(image_id) when every ID is a number and the numbers are distinct; else the position in sorted order."""
    if all(i.isdigit() for i in image_ids) and len({int(i) for i in image_ids}) == len(image_ids):
        return {i: int(i) for i in image_ids}
    return {i: k for k, i in enumerate(image_ids, start=1)}


def convert(csv_path: str | Path, image_dir: str | Path, workers: int = 8) -> tuple[dict, dict]:
    """`gt_train.csv` and the image folder -> (COCO dict, counts).

    Image sizes come from the JPEG headers. Boxes are clipped to the image; a box with zero area after
    clipping is dropped and counted. An image of the CSV with no file is left out and counted."""
    image_dir = Path(image_dir)
    rows = read_rows(csv_path)
    by_image: dict[str, list] = {}
    for image_id, name, *box in rows:
        by_image.setdefault(image_id, []).append((name, box))
    image_ids = sorted(by_image)
    present = [i for i in image_ids if (image_dir / f"{i}.jpg").exists()]
    with ThreadPoolExecutor(workers) as pool:
        sizes = dict(zip(present, pool.map(lambda i: image_size(image_dir / f"{i}.jpg"), present)))
    coco_id = _coco_ids(image_ids)

    images, annotations = [], []
    dropped, clipped = 0, 0
    per_class: Counter = Counter()
    for image_id in present:
        width, height = sizes[image_id]
        images.append({"id": coco_id[image_id], "miotcd_id": image_id, "file_name": f"{image_id}.jpg",
                       "width": width, "height": height})
        for name, (xmin, ymin, xmax, ymax) in by_image[image_id]:
            x0, y0 = min(max(xmin, 0.0), width), min(max(ymin, 0.0), height)
            x1, y1 = min(max(xmax, 0.0), width), min(max(ymax, 0.0), height)
            if (x0, y0, x1, y1) != (xmin, ymin, xmax, ymax):
                clipped += 1
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                dropped += 1
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": coco_id[image_id],
                                "category_id": category_id(name), "bbox": [x0, y0, w, h], "area": w * h,
                                "iscrowd": 0})
            per_class[name] += 1

    files = {p.stem for p in image_dir.glob("*.jpg")}
    counts = {
        "csv_rows": len(rows),
        "csv_images": len(image_ids),
        "csv_images_with_motorized_vehicle": sum(
            any(name == MOTORIZED_VEHICLE for name, _ in boxes) for boxes in by_image.values()
        ),
        "images": len(images),
        "boxes": len(annotations),
        "boxes_clipped": clipped,
        "boxes_dropped_zero_area": dropped,
        "images_missing_file": len(image_ids) - len(present),
        "files_not_in_csv": len(files - set(image_ids)),
        "boxes_per_class": {name: per_class[name] for name in CLASSES},
    }
    return {"images": images, "annotations": annotations, "categories": categories()}, counts


def count_differences(counts: dict) -> list[str]:
    """The checks of design §4.5, "Counts": one line for each reference count that the converter does not give,
    with what the converter knows about the cause."""
    out = []
    pairs = [("images", "csv_images"), ("boxes", "csv_rows"),
             ("images_with_motorized_vehicle", "csv_images_with_motorized_vehicle")]
    for reference, key in pairs:
        if counts[key] != REFERENCE_COUNTS[reference]:
            out.append(f"{reference}: the CSV has {counts[key]:,}, the reference count is {REFERENCE_COUNTS[reference]:,}")
    if counts["images_missing_file"]:
        out.append(f"{counts['images_missing_file']:,} images of the CSV have no file and are left out")
    if counts["boxes_dropped_zero_area"]:
        out.append(f"{counts['boxes_dropped_zero_area']:,} boxes have zero area after clipping and are dropped: "
                   f"{counts['boxes']:,} boxes remain of {counts['csv_rows']:,} rows")
    return out


def official_ignore_classes(evaluation_code: str | Path) -> tuple[str, ...]:
    """`motorized_vehicle_classes` as written in the dataset's `localization_evaluation.py`."""
    tree = ast.parse(Path(evaluation_code).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "motorized_vehicle_classes" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"{evaluation_code} does not assign motorized_vehicle_classes")


def check_ignore_classes(evaluation_code: str | Path) -> None:
    """Design §4.4: confirm the ignore classes against the dataset's own evaluation code."""
    official = official_ignore_classes(evaluation_code)
    if set(official) != set(IGNORE_UNDER):
        raise ValueError(f"{evaluation_code} lists {sorted(official)}; IGNORE_UNDER is {sorted(IGNORE_UNDER)}")


def load_annotations(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def subset(coco: dict, image_ids) -> dict:
    """The COCO dict with only the images in `image_ids` (COCO ids) and their annotations."""
    keep = {int(i) for i in image_ids}
    return {
        "images": [im for im in coco["images"] if im["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }


def boxes_per_class(coco: dict, image_ids=None) -> dict[str, int]:
    keep = None if image_ids is None else {int(i) for i in image_ids}
    counts = Counter(a["category_id"] for a in coco["annotations"] if keep is None or a["image_id"] in keep)
    return {name: counts[category_id(name)] for name in CLASSES}


def ids_by_miotcd_id(coco: dict) -> dict[str, int]:
    return {im["miotcd_id"]: im["id"] for im in coco["images"]}


def write_ids(path: str | Path, miotcd_ids) -> None:
    """A split file: one MIO-TCD image ID per line, sorted."""
    Path(path).write_text("".join(f"{i}\n" for i in sorted(miotcd_ids)))


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def split_image_ids(coco: dict, split_file: str | Path) -> list[int]:
    """The COCO ids of the images that a split file lists."""
    lookup = ids_by_miotcd_id(coco)
    ids = read_ids(split_file)
    unknown = [i for i in ids if i not in lookup]
    if unknown:
        raise KeyError(f"{split_file} lists {len(unknown)} images that are not in the annotations, e.g. {unknown[:3]}")
    return [lookup[i] for i in ids]
