import numpy as np
import pytest

from truckdet import errors
from truckdet.data.miotcd import CLASSES, SPECIFIC_CLASSES, TRUCK_FAMILY, categories, category_id

PICKUP, SUT, CAR, MV = (CLASSES.index(c) for c in ("pickup_truck", "single_unit_truck", "car", "motorized_vehicle"))


def one_hot(label: int, score: float = 0.9) -> np.ndarray:
    s = np.full(len(CLASSES), 0.01, dtype=np.float32)
    s[label] = score
    return s


def test_representative_query_is_the_best_box():
    gt = np.array([10.0, 10.0, 50.0, 40.0])
    boxes = np.array([[12, 10, 50, 40], [10, 10, 50, 40], [100, 100, 120, 120], [10, 10, 50, 40]], dtype=float)
    scores = np.stack([one_hot(CAR, 0.99), one_hot(PICKUP, 0.5), one_hot(SUT, 0.9), one_hot(SUT, 0.6)])
    assert errors.representative_query(gt, boxes, scores) == 3  # highest IoU; the tie of 1 and 3 goes to the higher score
    assert errors.representative_query(gt, boxes[:1], scores[:1]) == 0  # IoU 0.95: localized, not the best possible
    assert errors.representative_query(gt, boxes[2:3], scores[2:3]) == errors.NOT_LOCALIZED
    assert errors.representative_query(gt, boxes[:0], scores[:0]) == errors.NOT_LOCALIZED
    half = np.array([[10, 10, 30, 40]], dtype=float)  # IoU exactly 0.5 counts
    assert errors.representative_query(gt, half, scores[:1]) == 0


def case():
    """3 images. Image 1: a pickup found and called pickup; a truck found and called car. Image 2: a pickup
    with no box near it; a car found. Image 3: a motorized_vehicle box (in no error set) and a pickup whose only
    good box is the 3rd query."""
    def ann(i, image, name, box):
        return {"id": i, "image_id": image, "category_id": category_id(name), "bbox": box, "area": box[2] * box[3], "iscrowd": 0}

    gt = {"images": [{"id": i, "width": 200, "height": 150} for i in (1, 2, 3)], "categories": categories(),
          "annotations": [ann(1, 1, "pickup_truck", [10, 10, 40, 30]), ann(2, 1, "single_unit_truck", [100, 50, 60, 40]),
                          ann(3, 2, "pickup_truck", [10, 10, 40, 30]), ann(4, 2, "car", [100, 50, 60, 40]),
                          ann(5, 3, "motorized_vehicle", [5, 5, 10, 8]), ann(6, 3, "pickup_truck", [50, 50, 40, 30])]}
    far = [150, 120, 160, 130]
    predictions = {
        "image_ids": np.array([3, 1, 2]),  # not in sorted order: rows are found by image id
        "scores": np.stack([
            np.stack([one_hot(CAR), one_hot(CAR, 0.8), one_hot(PICKUP, 0.7)]),
            np.stack([one_hot(PICKUP), one_hot(CAR, 0.8), one_hot(SUT, 0.3)]),
            np.stack([one_hot(CAR), one_hot(CAR, 0.2), one_hot(CAR, 0.1)]),
        ]),
        "boxes": np.array([
            [far, far, [50, 50, 90, 80]],
            [[10, 10, 50, 40], [100, 50, 160, 90], far],
            [[100, 50, 160, 90], far, far],
        ], dtype=np.float32),
    }
    return gt, predictions


def test_box_table_and_error_split():
    gt, predictions = case()
    table = errors.box_table(gt, predictions)
    assert table.image_ids.tolist() == [1, 2, 3] and table.image.tolist() == [0, 0, 1, 1, 2, 2]
    assert table.classes.tolist() == [PICKUP, SUT, PICKUP, CAR, MV, PICKUP]
    assert table.predicted.tolist() == [PICKUP, CAR, errors.NOT_LOCALIZED, CAR, errors.NOT_LOCALIZED, PICKUP]
    # truck family: 4 boxes, 3 localized, 2 of the 3 with the correct label
    assert errors.error_split(table, TRUCK_FAMILY) == pytest.approx((3 / 4, 2 / 3))
    # specific classes: the car is added (found, correct); the motorized_vehicle box is in no set
    assert errors.error_split(table, SPECIFIC_CLASSES) == pytest.approx((4 / 5, 3 / 4))
    share, accuracy = errors.error_split(table, ["bus"])
    assert np.isnan(share) and np.isnan(accuracy)


def test_candidates_limit_the_queries():
    gt, predictions = case()
    table = errors.box_table(gt, predictions, candidates=2)  # image 3's good box is its 3rd query
    assert table.predicted.tolist()[-1] == errors.NOT_LOCALIZED
    assert errors.CANDIDATES == 50 and errors.IOU_THRESHOLD == 0.5


def test_bootstrap_equals_the_direct_calculation():
    gt, predictions = case()
    table = errors.box_table(gt, predictions)
    weights = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0], [0, 0, 0]])
    boot = errors.bootstrap_error_split(table, TRUCK_FAMILY, weights)
    assert boot.shape == (4, 2)
    for row, w in zip(boot[:3], weights[:3]):
        assert row == pytest.approx(errors.error_split(table, TRUCK_FAMILY, w), nan_ok=True)
    assert boot[1] == pytest.approx((5 / 5, 3 / 5))  # image 1 two times and image 3: 5 boxes, all found, 3 correct
    assert boot[2, 0] == 0.0 and np.isnan(boot[2, 1])  # only image 2: its pickup is not found
    assert np.isnan(boot[3]).all()


def test_confusion_counts():
    gt, predictions = case()
    counts = errors.confusion_counts(errors.box_table(gt, predictions), TRUCK_FAMILY)
    assert set(counts) == set(TRUCK_FAMILY)
    assert dict(counts["pickup_truck"]) == {"pickup_truck": 2, errors.NOT_LOCALIZED_NAME: 1}
    assert dict(counts["single_unit_truck"]) == {"car": 1} and not counts["work_van"]


def test_missing_prediction_is_an_error():
    gt, predictions = case()
    predictions = {k: v[:2] for k, v in predictions.items()}
    with pytest.raises(KeyError, match="image 2"):
        errors.box_table(gt, predictions)
