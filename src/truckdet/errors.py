"""The error split: are the errors in the boxes or in the labels? (design §7.2; from `separation.py`
of the truck project)

- *Candidates*: the 50 queries of an image with the highest class score.
- A ground-truth box is *localized* if a candidate overlaps it with IoU >= 0.5. The candidate with
  the highest IoU is the *representative query* of the box (a tie goes to the higher score).
- *Localized share*: the share of ground-truth boxes that are localized.
- *Label accuracy*: of the localized boxes, the share whose representative query has the correct
  top class.

Both numbers are for the boxes of a set of classes: all specific classes, or the truck family. The
top class of a query is over all 11 classes, so a truck that the model calls `motorized_vehicle`
has a wrong label. `motorized_vehicle` boxes themselves are not in either set.

The truck project took the most confident overlapping query as the representative, over all
queries. Here the candidates are already the confident queries, and the design takes the best box
among them: "is there a good box, and what does the model call it".
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data.miotcd import CLASSES

CANDIDATES = 50
IOU_THRESHOLD = 0.5
NOT_LOCALIZED = -1
NOT_LOCALIZED_NAME = "not localized"


@dataclass
class BoxTable:
    """One row per ground-truth box."""

    image: np.ndarray  # [G] position of the box's image in `image_ids`
    classes: np.ndarray  # [G] 0-based class of the box
    predicted: np.ndarray  # [G] top class of the representative query, NOT_LOCALIZED if none
    image_ids: np.ndarray  # [I] the images, sorted (also those without a box)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box with [Q, 4] xyxy boxes."""
    x0, y0 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x1, y1 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area - inter, 1e-12)


def representative_query(gt_xyxy: np.ndarray, boxes: np.ndarray, scores: np.ndarray, iou_thr: float = IOU_THRESHOLD) -> int:
    """Index of the candidate with the highest IoU, if that IoU is at least `iou_thr`; a tie goes to the
    higher top score. NOT_LOCALIZED if no candidate overlaps enough. `boxes` [Q, 4], `scores` [Q, C]."""
    if len(boxes) == 0:
        return NOT_LOCALIZED
    ious = _iou(gt_xyxy, boxes)
    best = ious.max()
    if best < iou_thr:
        return NOT_LOCALIZED
    tied = np.flatnonzero(ious == best)
    return int(tied[np.argmax(scores[tied].max(-1))])


def box_table(gt: dict, predictions: dict, candidates: int = CANDIDATES, iou_thr: float = IOU_THRESHOLD) -> BoxTable:
    """`predictions`: the queries of each image in the order of decreasing top score (as `evaluation.predict`
    writes them), so the candidates are the first `candidates` of them. Covers every box of `gt`; category
    ids are 1..C. Crowd boxes are left out."""
    image_ids = np.array(sorted(im["id"] for im in gt["images"]), dtype=np.int64)
    position = {int(i): k for k, i in enumerate(image_ids)}
    row = {int(i): k for k, i in enumerate(predictions["image_ids"])}
    image, classes, predicted = [], [], []
    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if a["image_id"] not in row:
            raise KeyError(f"image {a['image_id']} has no prediction")
        k = row[a["image_id"]]
        scores, boxes = predictions["scores"][k][:candidates], predictions["boxes"][k][:candidates]
        x, y, w, h = a["bbox"]
        q = representative_query(np.array([x, y, x + w, y + h]), boxes, scores, iou_thr)
        image.append(position[a["image_id"]])
        classes.append(a["category_id"] - 1)
        predicted.append(NOT_LOCALIZED if q == NOT_LOCALIZED else int(scores[q].argmax()))
    return BoxTable(np.array(image, dtype=np.int64), np.array(classes, dtype=np.int64),
                    np.array(predicted, dtype=np.int64), image_ids)


def class_indices(names) -> np.ndarray:
    return np.array([CLASSES.index(n) for n in names], dtype=np.int64)


def error_split(table: BoxTable, names, weights: np.ndarray | None = None) -> tuple[float, float]:
    """(localized share, label accuracy) over the boxes of the classes `names`. `weights` ([I], default 1)
    says how many times each image counts. NaN where the share has no boxes."""
    w = np.ones(len(table.image)) if weights is None else weights[table.image].astype(float)
    in_set = np.isin(table.classes, class_indices(names))
    localized = in_set & (table.predicted != NOT_LOCALIZED)
    correct = localized & (table.predicted == table.classes)
    total, n_localized = w[in_set].sum(), w[localized].sum()
    share = n_localized / total if total > 0 else float("nan")
    accuracy = w[correct].sum() / n_localized if n_localized > 0 else float("nan")
    return float(share), float(accuracy)


def bootstrap_error_split(table: BoxTable, names, weights: np.ndarray) -> np.ndarray:
    """`error_split` for each row of `weights` [n_boot, I]: an array [n_boot, 2] of (localized share, label accuracy)."""
    in_set = np.isin(table.classes, class_indices(names))
    localized = in_set & (table.predicted != NOT_LOCALIZED)
    correct = localized & (table.predicted == table.classes)

    def per_image(mask: np.ndarray) -> np.ndarray:  # boxes of `mask` in each image
        return np.bincount(table.image[mask], minlength=len(table.image_ids)).astype(float)

    total, n_localized, n_correct = (weights @ per_image(m) for m in (in_set, localized, correct))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([np.where(total > 0, n_localized / total, np.nan),
                         np.where(n_localized > 0, n_correct / n_localized, np.nan)], axis=1)


def confusion_counts(table: BoxTable, names) -> dict[str, Counter]:
    """Ground-truth class (of `names`) -> Counter of the top class of its representative query, or "not localized"."""
    out: dict[str, Counter] = {name: Counter() for name in names}
    for c, p in zip(table.classes, table.predicted):
        if CLASSES[c] in out:
            out[CLASSES[c]][NOT_LOCALIZED_NAME if p == NOT_LOCALIZED else CLASSES[p]] += 1
    return out
