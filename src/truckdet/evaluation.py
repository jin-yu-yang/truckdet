"""Predictions, and COCO AP from saved predictions.

**Predictions.** For each image the model gives Q queries, each with a box and one sigmoid score
per class. `predict` keeps the `keep` queries with the highest top score, with all their class
scores. That is enough for every reading below when `keep` is 100: a (query, class) pair of a
query that is not among the 100 best queries cannot be among the 100 best pairs, and pycocotools
reads at most 100 detections per image and class. The file is the input of all analysis (design
§3, decision 10), which then runs on a CPU.

**Readings.**
- `pairs_detections`: the 100 best (query, class) pairs of an image by score. This is the
  library's `post_process_object_detection(threshold=0, top_k=100)`, and the "pairs" reading of
  the truck project.
- `box_detections`: one detection per query, with its highest class score, all classes merged.

**Ground truth.** `add_ignore_regions` copies each `motorized_vehicle` box, with `iscrowd = 1`,
under each class of `IGNORE_UNDER`. pycocotools matches a detection to a crowd box by
intersection / detection area and then leaves it out: it is not a true positive and not a false
positive. The detection must lie in the crowd box to the same degree as the IoU threshold asks,
so at the strict thresholds a detection that is larger than the small box still counts as false.

**AP.** pycocotools does the matching (`match_detections`). The step after it, from matches to
AP (`average_precision`), is written here so that it takes a weight for each image: a bootstrap
resample is then a weight vector, and the matching runs one time for each run, not one time for
each resample. With weights of 1 it gives the numbers of `COCOeval.accumulate` exactly.
"""

import contextlib
import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data.miotcd import CLASSES, IGNORE_UNDER, MOTORIZED_VEHICLE, SPECIFIC_CLASSES, TRUCK_FAMILY, category_id
from .model.lora import LoRALinear, merge_lora

KEEP_QUERIES = 100
TOP_K = 100
AREAS = {"all": [0.0, 1e10], "small": [0.0, 32.0**2]}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


# --- Predictions ---------------------------------------------------------------------------


@torch.no_grad()
def predict(model, loader, device: torch.device, keep: int = KEEP_QUERIES) -> dict:
    """Run the model over `loader`. Returns {"image_ids": [N], "scores": [N, K, C] sigmoid scores,
    "boxes": [N, K, 4] xyxy in pixels of the original image}, K = min(keep, Q), the queries of an image
    in the order of decreasing top score."""
    model.eval()
    ids, scores, boxes = [], [], []
    for pixel_values, targets in loader:
        out = model(pixel_values=pixel_values.to(device))
        s, b = _queries(out.logits, out.pred_boxes, torch.stack([t["orig_size"] for t in targets]), keep)
        ids.extend(t["image_id"] for t in targets)
        scores.append(s.cpu().numpy())
        boxes.append(b.cpu().numpy())
    if not ids:
        return {"image_ids": np.zeros(0, np.int64), "scores": np.zeros((0, 0, 0), np.float32),
                "boxes": np.zeros((0, 0, 4), np.float32)}
    return {"image_ids": np.array(ids, dtype=np.int64), "scores": np.concatenate(scores).astype(np.float32),
            "boxes": np.concatenate(boxes).astype(np.float32)}


def _queries(logits: torch.Tensor, pred_boxes: torch.Tensor, orig_sizes: torch.Tensor, keep: int):
    """[B, Q, C] logits and [B, Q, 4] normalised cxcywh boxes -> the `keep` best queries' scores and pixel boxes."""
    scores = logits.float().sigmoid()
    cx, cy, w, h = pred_boxes.float().unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    heights, widths = orig_sizes.to(boxes).unbind(-1)
    boxes = boxes * torch.stack([widths, heights, widths, heights], dim=-1)[:, None, :]
    top = scores.max(-1).values.topk(min(keep, scores.shape[1]), dim=1).indices
    return (torch.gather(scores, 1, top[..., None].expand(-1, -1, scores.shape[-1])),
            torch.gather(boxes, 1, top[..., None].expand(-1, -1, 4)))


def merge_difference(model, pixel_values: torch.Tensor) -> float:
    """The largest change of any class score or normalised box coordinate when the adapters are merged into
    the base weights, on `pixel_values`. 0.0 for a model without adapters. The model is not changed.

    Calculated in float64 on a CPU copy. In float32 the check is not a check: the merged weights differ
    from the sum W x + B A x by rounding, the proposal stage keeps the 300 best of several thousand
    positions, and a rounding difference in the backbone features changes which positions are kept. The
    queries of the two models are then not the same queries, and their difference shows nothing about
    the merge. (Measured on arm B: up to 0.44 in float32, 3e-15 in float64.)"""
    if not any(isinstance(m, LoRALinear) for m in model.modules()):
        return 0.0
    probe = copy.deepcopy(model).to("cpu").double().eval()
    x = pixel_values.detach().to("cpu").double()
    with torch.no_grad():
        before = probe(pixel_values=x)
        merge_lora(probe)
        after = probe(pixel_values=x)
    return max(
        (before.logits.sigmoid() - after.logits.sigmoid()).abs().max().item(),
        (before.pred_boxes - after.pred_boxes).abs().max().item(),
    )


def merge_check(model, pixel_values: torch.Tensor) -> float:
    """`merge_difference`, and then merge the adapters of `model` itself, in place."""
    difference = merge_difference(model, pixel_values)
    merge_lora(model)
    return difference


def save_predictions(path, predictions: dict, meta: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta, sort_keys=True)), **predictions)
    tmp.replace(path)


def load_predictions(path) -> tuple[dict, dict]:
    """(predictions, meta) of a file written by `save_predictions`."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        return {k: z[k] for k in ("image_ids", "scores", "boxes")}, meta


# --- Readings ------------------------------------------------------------------------------


def _results(image_ids, scores, labels, boxes) -> list[dict]:
    """COCO-format detections from [N, K] scores and 0-based labels and [N, K, 4] xyxy boxes."""
    xywh = np.concatenate([boxes[..., :2], boxes[..., 2:] - boxes[..., :2]], axis=-1).astype(np.float64)
    out = []
    for image_id, s, l, b in zip(image_ids.tolist(), scores.tolist(), labels.tolist(), xywh.tolist()):
        out.extend({"image_id": image_id, "category_id": label + 1, "bbox": box, "score": score}
                   for score, label, box in zip(s, l, b))
    return out


def pairs_detections(predictions: dict, top_k: int = TOP_K) -> list[dict]:
    """The `top_k` best (query, class) pairs of each image, as COCO detections (category id = class index + 1)."""
    scores, boxes = predictions["scores"], predictions["boxes"]
    n, k, c = scores.shape
    if n == 0:
        return []
    flat = scores.reshape(n, k * c)
    order = np.argsort(-flat, axis=1, kind="stable")[:, :top_k]
    queries, labels = order // c, order % c
    rows = np.arange(n)[:, None]
    return _results(predictions["image_ids"], flat[rows, order], labels, boxes[rows, queries])


def box_detections(predictions: dict) -> list[dict]:
    """One detection per query, with its highest class score, under the one category of `merged_ground_truth`."""
    scores, boxes = predictions["scores"], predictions["boxes"]
    if scores.shape[0] == 0:
        return []
    return _results(predictions["image_ids"], scores.max(-1), np.zeros(scores.shape[:2], dtype=np.int64), boxes)


# --- Ground truth --------------------------------------------------------------------------


def add_ignore_regions(gt: dict, under: tuple[str, ...] = IGNORE_UNDER) -> dict:
    """`gt` and, for each `motorized_vehicle` box, a copy with iscrowd = 1 under each class of `under`."""
    source = category_id(MOTORIZED_VEHICLE)
    next_id = max((a["id"] for a in gt["annotations"]), default=0) + 1
    extra = []
    for a in gt["annotations"]:
        if a["category_id"] == source and not a.get("iscrowd", 0):
            for name in under:
                extra.append({**a, "id": next_id, "category_id": category_id(name), "iscrowd": 1})
                next_id += 1
    return {**gt, "annotations": gt["annotations"] + extra}


def merged_ground_truth(gt: dict) -> dict:
    """Every box under one category, `object`: the ground truth of box AP."""
    return {**gt, "annotations": [{**a, "category_id": 1} for a in gt["annotations"]],
            "categories": [{"id": 1, "name": "object"}]}


def coco_from_dict(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


# --- Matching (pycocotools) and AP (here) --------------------------------------------------


@dataclass
class ClassMatches:
    """The detections of one class after matching, in the order of decreasing score."""

    image: np.ndarray  # [n] position of the detection's image in Matches.image_ids
    tp: np.ndarray  # [T, n] bool: matched to a box that counts, at each IoU threshold
    fp: np.ndarray  # [T, n] bool: unmatched, and not ignored
    gt_per_image: np.ndarray  # [I] boxes of the class that count (not crowd, in the area range), per image


@dataclass
class Matches:
    image_ids: np.ndarray  # [I], sorted
    per_class: dict[str, ClassMatches]  # class name -> its detections

    @property
    def classes(self) -> list[str]:
        return list(self.per_class)


def match_detections(gt: dict, results: list[dict], class_names=None, area: str = "all") -> Matches:
    """Match `results` to `gt` with pycocotools, for the classes `class_names` (default: all categories of
    `gt`) and one area range. An image counts at most 100 detections of a class."""
    names = {c["name"]: c["id"] for c in gt["categories"]}
    class_names = list(names) if class_names is None else list(class_names)
    class_names.sort(key=lambda c: names[c])
    coco_gt = coco_from_dict(gt)
    if results:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(results)
    else:
        coco_dt = coco_from_dict({"images": gt["images"], "annotations": [], "categories": gt["categories"]})
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(im["id"] for im in gt["images"])
    evaluator.params.catIds = [names[c] for c in class_names]
    evaluator.params.areaRng = [AREAS[area]]
    evaluator.params.areaRngLbl = [area]
    evaluator.params.maxDets = [TOP_K]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    image_ids = np.array(evaluator.params.imgIds, dtype=np.int64)
    n_images, n_thresholds = len(image_ids), len(evaluator.params.iouThrs)
    per_class = {}
    for k, name in enumerate(class_names):
        scores, image, matched, ignored = [], [], [], []
        gt_per_image = np.zeros(n_images, dtype=np.int64)
        for i, e in enumerate(evaluator.evalImgs[k * n_images : (k + 1) * n_images]):
            if e is None:
                continue
            gt_per_image[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
            if len(e["dtScores"]):
                scores.append(np.asarray(e["dtScores"], dtype=np.float64))
                image.append(np.full(len(e["dtScores"]), i, dtype=np.int64))
                matched.append(np.asarray(e["dtMatches"]) > 0)
                ignored.append(np.asarray(e["dtIgnore"], dtype=bool))
        if scores:
            order = np.argsort(-np.concatenate(scores), kind="mergesort")  # pycocotools' order, ties included
            matched = np.concatenate(matched, axis=1)[:, order]
            ignored = np.concatenate(ignored, axis=1)[:, order]
            per_class[name] = ClassMatches(np.concatenate(image)[order], matched & ~ignored, ~matched & ~ignored,
                                           gt_per_image)
        else:
            empty = np.zeros((n_thresholds, 0), dtype=bool)
            per_class[name] = ClassMatches(np.zeros(0, dtype=np.int64), empty, empty, gt_per_image)
    return Matches(image_ids, per_class)


def average_precision(m: ClassMatches, weights: np.ndarray | None = None) -> np.ndarray:
    """[T] AP of one class at each IoU threshold: the 101-point interpolated precision of pycocotools.

    `weights` ([I], default 1) says how many times each image counts. A weight of 2 is the same as a
    second copy of the image and its detections; 0 leaves the image out. NaN if no box counts."""
    if weights is None:
        per_detection = np.ones(len(m.image), dtype=np.float64)
        n_gt = float(m.gt_per_image.sum())
    else:
        per_detection = weights[m.image].astype(np.float64)
        n_gt = float((m.gt_per_image * weights).sum())
    n_thresholds, n = m.tp.shape
    if n_gt == 0:
        return np.full(n_thresholds, np.nan)
    if n == 0:
        return np.zeros(n_thresholds)
    tp = np.cumsum(m.tp * per_detection, axis=1)
    fp = np.cumsum(m.fp * per_detection, axis=1)
    recall = tp / n_gt
    precision = tp / (fp + tp + np.spacing(1))
    precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]  # best precision at this recall or more
    ap = np.empty(n_thresholds)
    for t in range(n_thresholds):
        at = np.searchsorted(recall[t], RECALL_THRESHOLDS, side="left")
        ap[t] = np.where(at < n, precision[t, np.minimum(at, n - 1)], 0.0).mean()
    return ap


def ap_table(matches: Matches, weights: np.ndarray | None = None) -> np.ndarray:
    """[C, T] AP of each class (in the order of `matches.classes`) at each IoU threshold."""
    return np.stack([average_precision(m, weights) for m in matches.per_class.values()])


def _mean(values) -> float:
    """Mean over the classes that have boxes (as pycocotools: a class without boxes is left out); NaN if none."""
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def summarize(table: np.ndarray, classes: list[str]) -> dict:
    """AP (IoU 0.50 to 0.95), AP50 and AP75 as means over the classes, and AP for each class."""
    per_class = table.mean(axis=1)
    return {"AP": _mean(per_class), "AP50": _mean(table[:, 0]), "AP75": _mean(table[:, 5]),
            "per_class": {c: float(v) for c, v in zip(classes, per_class)}}


# --- The metrics of one run ----------------------------------------------------------------


def primary_matches(gt: dict, predictions: dict) -> Matches:
    """The matches of the primary metric: the pairs reading, the 10 specific classes, with ignore regions."""
    return match_detections(add_ignore_regions(gt), pairs_detections(predictions), SPECIFIC_CLASSES)


def metric_names() -> list[str]:
    return ["AP", "AP50", "AP75", "AP truck family", *SPECIFIC_CLASSES]


def metric_vector(matches: Matches, weights: np.ndarray | None = None) -> np.ndarray:
    """The values of `metric_names()` from the matches of the primary metric."""
    table = ap_table(matches, weights)
    per_class = dict(zip(matches.classes, table.mean(axis=1)))
    return np.array([_mean(table.mean(axis=1)), _mean(table[:, 0]), _mean(table[:, 5]),
                     _mean([per_class[c] for c in TRUCK_FAMILY]), *[per_class[c] for c in SPECIFIC_CLASSES]])


def check_same_images(gt: dict, predictions: dict) -> None:
    gt_ids = {im["id"] for im in gt["images"]}
    predicted = set(predictions["image_ids"].tolist())
    if gt_ids != predicted:
        raise ValueError(f"predictions cover {len(predicted)} images, the ground truth has {len(gt_ids)}; "
                         f"{len(gt_ids - predicted)} have no prediction and {len(predicted - gt_ids)} are not in the split")


def score_predictions(gt: dict, predictions: dict, primary: Matches | None = None) -> dict:
    """All detection metrics of design §7.1 and box AP (§7.2) for the images of `gt`.

    `gt` is the plain ground truth of the split (no ignore regions); every image of it must have predictions.
    `primary`: the result of `primary_matches(gt, predictions)`, if the caller already has it."""
    check_same_images(gt, predictions)
    pairs = pairs_detections(predictions)
    with_ignore = add_ignore_regions(gt)
    if primary is None:
        primary = match_detections(with_ignore, pairs, SPECIFIC_CLASSES)
    out = summarize(ap_table(primary), primary.classes)
    out["AP truck family"] = _mean([out["per_class"][c] for c in TRUCK_FAMILY])
    small = match_detections(with_ignore, pairs, SPECIFIC_CLASSES, area="small")
    out["AP small"] = summarize(ap_table(small), small.classes)["AP"]
    plain = match_detections(gt, pairs, CLASSES)
    no_ignore = summarize(ap_table(plain), plain.classes)
    out["AP 11 classes, no ignore regions"] = no_ignore["AP"]
    out[f"AP {MOTORIZED_VEHICLE}"] = no_ignore["per_class"][MOTORIZED_VEHICLE]
    boxes = match_detections(merged_ground_truth(gt), box_detections(predictions))
    box = summarize(ap_table(boxes), boxes.classes)
    out["box AP"], out["box AP50"] = box["AP"], box["AP50"]
    out["images"] = len(gt["images"])
    return out
