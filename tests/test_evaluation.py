import contextlib
import io

import numpy as np
import pytest
import torch
from pycocotools.cocoeval import COCOeval

from tests.tiny import TINY_SIZE, tiny_loader, tiny_processor
from truckdet import evaluation as ev
from truckdet.data.miotcd import CLASSES, IGNORE_UNDER, SPECIFIC_CLASSES, TRUCK_FAMILY, categories, category_id
from truckdet.model import arms

CAR, BUS, MV, PED = (category_id(c) for c in ("car", "bus", "motorized_vehicle", "pedestrian"))


def gt_of(boxes_by_image: dict[int, list[tuple[int, list[float]]]], size=(200, 150)) -> dict:
    """{image id: [(category id, [x, y, w, h]), ...]} -> COCO dict."""
    images = [{"id": i, "width": size[0], "height": size[1]} for i in boxes_by_image]
    anns = []
    for i, boxes in boxes_by_image.items():
        for cat, box in boxes:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cat, "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": categories()}


def det(image_id, cat, box, score):
    return {"image_id": image_id, "category_id": cat, "bbox": box, "score": score}


def random_case(n_images=120, seed=0):
    rng = np.random.default_rng(seed)
    boxes, dets = {}, []
    for i in range(1, n_images + 1):
        boxes[i] = []
        for _ in range(rng.integers(0, 5)):
            w, h = rng.uniform(8, 80), rng.uniform(8, 60)
            x, y = rng.uniform(0, 200 - w), rng.uniform(0, 150 - h)
            cat = int(rng.integers(1, 12))
            boxes[i].append((cat, [x, y, w, h]))
            if rng.random() < 0.8:
                j = rng.normal(0, 4, 4)
                wrong = cat if rng.random() < 0.7 else int(rng.integers(1, 12))
                dets.append(det(i, wrong, [x + j[0], y + j[1], max(2, w + j[2]), max(2, h + j[3])], float(rng.random())))
        for _ in range(rng.integers(0, 6)):
            dets.append(det(i, int(rng.integers(1, 12)), [rng.uniform(0, 120), rng.uniform(0, 90), rng.uniform(8, 80),
                                                           rng.uniform(8, 60)], float(rng.random())))
    return gt_of(boxes), dets


def pycocotools_stats(gt, dets):
    coco = ev.coco_from_dict(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(coco, coco.loadRes(dets), "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    precision = e.eval["precision"][:, :, :, 0, 2]
    per_class = [precision[:, :, k][precision[:, :, k] > -1].mean() if (precision[:, :, k] > -1).any() else np.nan
                 for k in range(precision.shape[2])]
    return e.stats, np.array(per_class)


# --- AP: equal to pycocotools, and weights are resamples ------------------------------------


def test_ap_equals_pycocotools():
    gt, dets = random_case()
    gt = ev.add_ignore_regions(gt)  # with crowd boxes in the ground truth
    stats, per_class = pycocotools_stats(gt, dets)
    matches = ev.match_detections(gt, dets)
    mine = ev.summarize(ev.ap_table(matches), matches.classes)
    assert matches.classes == list(CLASSES)
    assert mine["AP"] == pytest.approx(stats[0], abs=1e-12)
    assert mine["AP50"] == pytest.approx(stats[1], abs=1e-12)
    assert mine["AP75"] == pytest.approx(stats[2], abs=1e-12)
    assert np.allclose(list(mine["per_class"].values()), per_class, atol=1e-12, equal_nan=True)
    small = ev.match_detections(gt, dets, area="small")
    assert ev.summarize(ev.ap_table(small), small.classes)["AP"] == pytest.approx(stats[3], abs=1e-12)


def test_image_weights_are_a_resample():
    """Weights (2, 0, 1, ...) give the AP of a dataset with the images copied that many times."""
    gt, dets = random_case(n_images=80, seed=3)
    matches = ev.match_detections(gt, dets)
    ids = sorted(im["id"] for im in gt["images"])
    rng = np.random.default_rng(1)
    for _ in range(3):
        picks = rng.choice(ids, size=len(ids), replace=True)
        weights = np.bincount(np.searchsorted(ids, picks), minlength=len(ids))
        by_image_a, by_image_d = {}, {}
        for a in gt["annotations"]:
            by_image_a.setdefault(a["image_id"], []).append(a)
        for d in dets:
            by_image_d.setdefault(d["image_id"], []).append(d)
        images, anns, copies = [], [], []
        for k, old in enumerate(picks.tolist(), start=1):  # the truck project's physical resample
            images.append({"id": k, "width": 200, "height": 150})
            anns += [{**a, "id": len(anns) + n + 1, "image_id": k} for n, a in enumerate(by_image_a.get(old, []))]
            copies += [{**d, "image_id": k} for d in by_image_d.get(old, [])]
        _, per_class = pycocotools_stats({"images": images, "annotations": anns, "categories": gt["categories"]}, copies)
        assert np.allclose(ev.ap_table(matches, weights).mean(axis=1), per_class, atol=1e-12, equal_nan=True)
    assert np.array_equal(ev.ap_table(matches, np.ones(len(ids), dtype=int)), ev.ap_table(matches))


def test_class_without_boxes_is_left_out_and_no_detections_is_zero():
    gt = gt_of({1: [(CAR, [10, 10, 40, 30])], 2: []})
    matches = ev.match_detections(gt, [det(1, CAR, [10, 10, 40, 30], 0.9), det(2, BUS, [0, 0, 20, 20], 0.8)])
    table = ev.ap_table(matches)
    summary = ev.summarize(table, matches.classes)
    assert summary["per_class"]["car"] == pytest.approx(1.0) and np.isnan(summary["per_class"]["bus"])
    assert summary["AP"] == pytest.approx(1.0)  # the mean is over classes that have boxes
    empty = ev.match_detections(gt, [])
    assert ev.summarize(ev.ap_table(empty), empty.classes)["AP"] == 0.0
    assert np.isnan(ev.ap_table(matches, np.array([0, 1]))[matches.classes.index("car")]).all()  # its only image has weight 0


# --- Ignore regions: the synthetic checks of stage 2 ---------------------------------------


def stage2_case():
    gt = gt_of({1: [(CAR, [10, 10, 40, 30]), (MV, [120, 80, 20, 14])], 2: [(CAR, [50, 50, 60, 40]), (BUS, [5, 5, 30, 30])]})
    dets = [det(1, CAR, [10, 10, 40, 30], 0.9), det(2, CAR, [50, 50, 60, 40], 0.8), det(2, BUS, [5, 5, 30, 30], 0.7)]
    return gt, dets


def primary_ap(gt, dets) -> dict:
    matches = ev.match_detections(ev.add_ignore_regions(gt), dets, SPECIFIC_CLASSES)
    return ev.summarize(ev.ap_table(matches), matches.classes)


def same(a: dict, b: dict) -> bool:
    """Equal summaries; a class without boxes is NaN in both."""
    flat = lambda d: [d["AP"], d["AP50"], d["AP75"], *d["per_class"].values()]  # noqa: E731
    return np.array_equal(flat(a), flat(b), equal_nan=True)


def test_add_ignore_regions():
    gt, _ = stage2_case()
    out = ev.add_ignore_regions(gt)
    extra = out["annotations"][len(gt["annotations"]):]
    assert len(extra) == len(IGNORE_UNDER) and all(a["iscrowd"] == 1 and a["bbox"] == [120, 80, 20, 14] for a in extra)
    assert {a["category_id"] for a in extra} == {category_id(c) for c in IGNORE_UNDER}
    assert len({a["id"] for a in out["annotations"]}) == len(out["annotations"]) and min(a["id"] for a in out["annotations"]) >= 1
    assert out["annotations"][: len(gt["annotations"])] == gt["annotations"] and len(gt["annotations"]) == 4  # input unchanged


def test_car_detection_on_a_motorized_vehicle_box_changes_no_ap():
    gt, dets = stage2_case()
    before = primary_ap(gt, dets)
    after = primary_ap(gt, dets + [det(1, CAR, [120, 80, 20, 14], 0.99)])  # the highest score of the image set
    assert before["AP"] == pytest.approx(1.0)
    assert same(after, before)


def test_the_same_detection_on_background_decreases_car_ap():
    gt, dets = stage2_case()
    before = primary_ap(gt, dets)
    after = primary_ap(gt, dets + [det(1, CAR, [150, 20, 20, 14], 0.99)])
    assert after["per_class"]["car"] < before["per_class"]["car"]
    assert after["per_class"]["bus"] == before["per_class"]["bus"]


def test_without_ignore_regions_the_detection_is_a_false_positive():
    gt, dets = stage2_case()
    plain = ev.match_detections(gt, dets + [det(1, CAR, [120, 80, 20, 14], 0.99)], SPECIFIC_CLASSES)
    assert ev.summarize(ev.ap_table(plain), plain.classes)["per_class"]["car"] < 1.0


def test_a_class_outside_the_ignore_list_is_still_penalised():
    gt, dets = stage2_case()
    gt["annotations"].append({"id": 99, "image_id": 2, "category_id": PED, "bbox": [150, 100, 10, 20], "area": 200, "iscrowd": 0})
    dets = dets + [det(2, PED, [150, 100, 10, 20], 0.6)]
    assert "pedestrian" not in IGNORE_UNDER
    after = primary_ap(gt, dets + [det(1, PED, [120, 80, 20, 14], 0.99)])
    assert after["per_class"]["pedestrian"] < primary_ap(gt, dets)["per_class"]["pedestrian"]


# --- Readings of the saved predictions -----------------------------------------------------


def tiny_outputs(n=3, seed=0):
    # The raw tiny model: every class row is different. (In the start model the three truck classes are copies
    # of one COCO row and tie, and torch.topk and numpy order ties differently.)
    model = tiny_loader("tiny").eval()
    pixels = torch.randn(n, 3, TINY_SIZE, TINY_SIZE, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return model, pixels, model(pixel_values=pixels)


def test_pairs_reading_is_the_librarys_post_processing():
    """Design §7.1: the detections are `post_process_object_detection(threshold=0, top_k=100)`."""
    _, _, out = tiny_outputs()
    sizes = torch.tensor([[80, 80], [72, 96], [150, 200]])
    library = tiny_processor().post_process_object_detection(out, threshold=0, top_k=100, target_sizes=sizes)
    scores, boxes = ev._queries(out.logits, out.pred_boxes, sizes, keep=100)
    predictions = {"image_ids": np.array([5, 6, 7]), "scores": scores.numpy(), "boxes": boxes.numpy()}
    mine = ev.pairs_detections(predictions)
    assert len(mine) == 300 and out.logits.shape[1] * out.logits.shape[2] == 120  # 100 of the 120 pairs per image
    for k, image_id in enumerate([5, 6, 7]):
        rows = [d for d in mine if d["image_id"] == image_id]
        assert [d["score"] for d in rows] == pytest.approx(library[k]["scores"].tolist(), abs=1e-7)
        assert [d["category_id"] - 1 for d in rows] == library[k]["labels"].tolist()
        xyxy = [[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in rows]
        assert np.allclose(xyxy, library[k]["boxes"].numpy(), atol=1e-4)


def test_keeping_100_queries_loses_no_detection():
    """The 100 best pairs of 300 queries are among the pairs of the 100 queries with the highest top score."""
    g = torch.Generator().manual_seed(0)
    logits, boxes = torch.randn(4, 300, 11, generator=g) * 3, torch.rand(4, 300, 4, generator=g)
    sizes = torch.tensor([[100, 200]] * 4)
    ids = np.arange(4)
    full = dict(zip(("scores", "boxes"), (t.numpy() for t in ev._queries(logits, boxes, sizes, keep=300))), image_ids=ids)
    kept = dict(zip(("scores", "boxes"), (t.numpy() for t in ev._queries(logits, boxes, sizes, keep=100))), image_ids=ids)
    assert kept["scores"].shape == (4, 100, 11)
    assert (np.diff(kept["scores"].max(-1), axis=1) <= 0).all()  # queries in the order of decreasing top score
    assert ev.pairs_detections(full) == ev.pairs_detections(kept)
    for image_id in ids.tolist():  # box AP reads at most 100 detections per image: the 100 best queries
        best = lambda p: sorted((d for d in ev.box_detections(p) if d["image_id"] == image_id),  # noqa: E731
                                key=lambda d: -d["score"])[:100]
        assert best(full) == best(kept)


def test_box_detections_merge_the_classes():
    predictions = {"image_ids": np.array([1]), "scores": np.array([[[0.1, 0.7, 0.2], [0.6, 0.3, 0.5]]], dtype=np.float32),
                   "boxes": np.array([[[0, 0, 10, 20], [5, 5, 25, 15]]], dtype=np.float32)}
    out = ev.box_detections(predictions)
    assert [(d["category_id"], round(d["score"], 4), d["bbox"]) for d in out] == [(1, 0.7, [0, 0, 10, 20]), (1, 0.6, [5, 5, 20, 10])]
    gt = ev.merged_ground_truth(gt_of({1: [(CAR, [0, 0, 10, 20]), (MV, [5, 5, 20, 10])]}))
    assert {a["category_id"] for a in gt["annotations"]} == {1} and gt["categories"] == [{"id": 1, "name": "object"}]
    matches = ev.match_detections(gt, out)
    assert ev.summarize(ev.ap_table(matches), matches.classes)["AP"] == pytest.approx(1.0)


def test_predictions_file_round_trip(tmp_path):
    predictions = {"image_ids": np.array([3, 9]), "scores": np.random.default_rng(0).random((2, 5, 11)).astype(np.float32),
                   "boxes": np.random.default_rng(1).random((2, 5, 4)).astype(np.float32)}
    ev.save_predictions(tmp_path / "pred_val.npz", predictions, {"run": "b_lr3e-4_s0", "merge_max_abs_diff": 1e-12})
    loaded, meta = ev.load_predictions(tmp_path / "pred_val.npz")
    assert all(np.array_equal(loaded[k], predictions[k]) for k in predictions)
    assert meta == {"run": "b_lr3e-4_s0", "merge_max_abs_diff": 1e-12}


def perfect_predictions(gt: dict, keep: int = 4) -> dict:
    """Predictions whose first queries are the ground-truth boxes with a score of 0.9 for the right class."""
    ids = sorted(im["id"] for im in gt["images"])
    scores = np.full((len(ids), keep, len(CLASSES)), 0.01, dtype=np.float32)
    boxes = np.tile(np.array([0, 0, 1, 1], dtype=np.float32), (len(ids), keep, 1))
    for k, i in enumerate(ids):
        for q, a in enumerate([a for a in gt["annotations"] if a["image_id"] == i]):
            x, y, w, h = a["bbox"]
            boxes[k, q] = [x, y, x + w, y + h]
            scores[k, q, a["category_id"] - 1] = 0.9 - 0.01 * q
    return {"image_ids": np.array(ids), "scores": scores, "boxes": boxes}


def test_score_predictions_reports_the_metrics_of_the_design():
    gt, _ = stage2_case()
    result = ev.score_predictions(gt, perfect_predictions(gt))
    assert result["AP"] == pytest.approx(1.0) and result["AP50"] == pytest.approx(1.0) and result["images"] == 2
    assert set(result["per_class"]) == set(SPECIFIC_CLASSES)
    for key in ("AP75", "AP small", "AP truck family", "AP motorized_vehicle", "AP 11 classes, no ignore regions",
                "box AP", "box AP50"):
        assert key in result
    assert result["AP motorized_vehicle"] == pytest.approx(1.0) and result["box AP"] == pytest.approx(1.0)
    assert np.isnan(result["AP truck family"])  # no truck-family box in this case
    vector = ev.metric_vector(ev.primary_matches(gt, perfect_predictions(gt)))
    assert len(vector) == len(ev.metric_names()) == 4 + len(SPECIFIC_CLASSES) and vector[0] == pytest.approx(1.0)
    assert ev.metric_names()[3] == "AP truck family" and set(TRUCK_FAMILY) <= set(ev.metric_names())


def test_predictions_must_cover_the_split():
    gt, _ = stage2_case()
    predictions = perfect_predictions(gt)
    predictions = {k: v[:1] for k, v in predictions.items()}
    with pytest.raises(ValueError, match="1 have no prediction"):
        ev.score_predictions(gt, predictions)


def test_predict_gives_pixel_boxes_of_the_original_image(data_root):
    from truckdet.data.dataset import MioTcdDetection, collate
    from truckdet.data.miotcd import load_annotations

    coco = load_annotations(data_root / "annotations.json")
    dataset = MioTcdDetection(coco, data_root / "train", [0, 1, 2], tiny_processor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate)
    model = arms.build_start_model("tiny", tiny_loader)
    predictions = ev.predict(model, loader, torch.device("cpu"), keep=5)
    assert predictions["image_ids"].tolist() == [0, 1, 2]
    assert predictions["scores"].shape == (3, 5, 11) and predictions["boxes"].shape == (3, 5, 4)
    with torch.no_grad():
        out = model.eval()(pixel_values=torch.stack([dataset[1][0]]))
    best = out.logits[0].sigmoid().max(-1).values.argmax()
    cx, cy, w, h = out.pred_boxes[0, best].tolist()
    expected = [(cx - w / 2) * 96, (cy - h / 2) * 72, (cx + w / 2) * 96, (cy + h / 2) * 72]  # image 1 is 96 x 72
    assert np.allclose(predictions["boxes"][1, 0], expected, atol=1e-3)
    assert not model.training
