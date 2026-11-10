import csv
import json

import pytest
from PIL import Image

from tests.tiny import write_synthetic_miotcd
from truckdet.data import miotcd
from truckdet.data.miotcd import CLASSES, IGNORE_UNDER, SPECIFIC_CLASSES, TRUCK_FAMILY


def test_class_lists():
    assert len(CLASSES) == 11 and list(CLASSES) == sorted(CLASSES)  # the dataset's order is alphabetical
    assert len(SPECIFIC_CLASSES) == 10 and "motorized_vehicle" not in SPECIFIC_CLASSES
    assert set(TRUCK_FAMILY) <= set(SPECIFIC_CLASSES) and len(TRUCK_FAMILY) == 4
    assert set(IGNORE_UNDER) <= set(SPECIFIC_CLASSES)
    assert miotcd.category_id("articulated_truck") == 1 and miotcd.category_id("work_van") == 11


def write_case(root, rows, sizes):
    (root / "train").mkdir(parents=True)
    for image_id, (w, h) in sizes.items():
        Image.new("RGB", (w, h)).save(root / "train" / f"{image_id}.jpg")
    with open(root / "gt_train.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_convert_keeps_leading_zeros_clips_and_drops(tmp_path):
    write_case(tmp_path, [
        ["00000007", "car", 10, 20, 50, 60],
        ["00000007", "bus", -5, 30, 120, 300],  # outside the image on three sides
        ["00000007", "pedestrian", 40, 40, 40, 70],  # zero width
        ["00000123", "motorized_vehicle", 1, 2, 3, 4],
        ["00000999", "car", 0, 0, 5, 5],  # no file
    ], {"00000007": (100, 80), "00000123": (64, 48), "00000500": (10, 10)})
    coco, counts = miotcd.convert(tmp_path / "gt_train.csv", tmp_path / "train")
    assert [(im["id"], im["miotcd_id"], im["file_name"], im["width"], im["height"]) for im in coco["images"]] == [
        (7, "00000007", "00000007.jpg", 100, 80), (123, "00000123", "00000123.jpg", 64, 48)]
    boxes = {(a["image_id"], a["category_id"]): a for a in coco["annotations"]}
    assert boxes[7, miotcd.category_id("car")]["bbox"] == [10, 20, 40, 40]
    assert boxes[7, miotcd.category_id("bus")]["bbox"] == [0, 30, 100, 50]  # clipped to 100 x 80
    assert boxes[7, miotcd.category_id("bus")]["area"] == 5000
    assert (7, miotcd.category_id("pedestrian")) not in boxes
    assert all(a["iscrowd"] == 0 for a in coco["annotations"]) and [a["id"] for a in coco["annotations"]] == [1, 2, 3]
    assert [c["name"] for c in coco["categories"]] == list(CLASSES)
    assert counts["csv_rows"] == 5 and counts["csv_images"] == 3 and counts["images"] == 2 and counts["boxes"] == 3
    assert counts["boxes_dropped_zero_area"] == 1 and counts["boxes_clipped"] == 1
    assert counts["images_missing_file"] == 1 and counts["files_not_in_csv"] == 1
    assert counts["csv_images_with_motorized_vehicle"] == 1
    assert counts["boxes_per_class"]["car"] == 1 and counts["boxes_per_class"]["motorized_vehicle"] == 1


def test_non_numeric_ids_get_positions(tmp_path):
    write_case(tmp_path, [["b", "car", 0, 0, 5, 5], ["a", "car", 0, 0, 5, 5]], {"a": (10, 10), "b": (10, 10)})
    coco, _ = miotcd.convert(tmp_path / "gt_train.csv", tmp_path / "train")
    assert [(im["id"], im["miotcd_id"]) for im in coco["images"]] == [(1, "a"), (2, "b")]


def test_unknown_class_and_bad_rows_raise(tmp_path):
    write_case(tmp_path, [["1", "truck", 0, 0, 5, 5]], {"1": (10, 10)})
    with pytest.raises(ValueError, match="unknown class"):
        miotcd.read_rows(tmp_path / "gt_train.csv")
    (tmp_path / "bad.csv").write_text("1,car,0,0,5\n")
    with pytest.raises(ValueError, match="6 columns"):
        miotcd.read_rows(tmp_path / "bad.csv")


def test_count_differences():
    reference = {"csv_images": 110_000, "csv_rows": 351_549, "csv_images_with_motorized_vehicle": 20_299,
                 "images": 110_000, "boxes": 351_549, "boxes_dropped_zero_area": 0, "images_missing_file": 0}
    assert miotcd.count_differences(reference) == []
    assert miotcd.count_notes(reference) == []
    off = {**reference, "csv_rows": 351_000, "boxes": 350_990, "boxes_dropped_zero_area": 10, "images_missing_file": 3}
    lines = miotcd.count_differences(off)
    assert len(lines) == 2 and "351,000" in lines[0] and "3 images" in lines[1]
    # What MIO-TCD gave (stage 1): the three reference counts, and 5 rows with zero area. The converter drops and
    # records them (design §4.2); the check of the reference counts passes.
    real = {**reference, "boxes": 351_544, "boxes_dropped_zero_area": 5, "boxes_clipped": 0, "files_not_in_csv": 0}
    assert miotcd.count_differences(real) == []
    assert miotcd.count_notes(real) == ["5 boxes have zero area after clipping and are dropped: 351,544 boxes remain "
                                        "of 351,549 rows"]


def test_ignore_classes_are_read_from_the_official_code(tmp_path):
    code = tmp_path / "localization_evaluation.py"
    code.write_text("classes = ['car']\nmotorized_vehicle_classes = ['articulated_truck', 'bus', 'car',\n"
                    "                             'pickup_truck', 'single_unit_truck', 'work_van']\n")
    assert miotcd.official_ignore_classes(code) == IGNORE_UNDER
    miotcd.check_ignore_classes(code)
    code.write_text("motorized_vehicle_classes = ['car', 'bus', 'motorcycle']\n")
    with pytest.raises(ValueError, match="motorcycle"):
        miotcd.check_ignore_classes(code)
    code.write_text("x = 1\n")
    with pytest.raises(ValueError):
        miotcd.official_ignore_classes(code)


def test_subset_split_files_and_counts(tmp_path):
    root = write_synthetic_miotcd(tmp_path / "data", n_images=12)
    coco, counts = miotcd.convert(root / "gt_train.csv", root / "train")
    assert counts["images"] == 12 and counts["boxes"] == 16 and counts["csv_images_with_motorized_vehicle"] == 5  # images 0, 3, 6, 9, and image 5 whose own class it is
    part = miotcd.subset(coco, [0, 3])
    assert [im["id"] for im in part["images"]] == [0, 3] and {a["image_id"] for a in part["annotations"]} == {0, 3}
    assert miotcd.boxes_per_class(coco, [0, 3])["motorized_vehicle"] == 2
    miotcd.write_ids(tmp_path / "val.txt", ["00000003", "00000000"])
    assert (tmp_path / "val.txt").read_text() == "00000000\n00000003\n"
    assert miotcd.split_image_ids(coco, tmp_path / "val.txt") == [0, 3]
    miotcd.write_ids(tmp_path / "bad.txt", ["00000099"])
    with pytest.raises(KeyError):
        miotcd.split_image_ids(coco, tmp_path / "bad.txt")
    assert json.loads(json.dumps(coco)) == coco  # the JSON on disk is the same dict
