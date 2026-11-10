"""Scene groups and the split by scene (design §4.3).

Frames from one camera show the same background, and a random image split puts the same camera in
train and test. `gt_train.csv` has no camera ID, so the project estimates one:

1. One embedding for each labelled image: the mean of the last feature map of the frozen backbone,
   at 224 px.
2. Embeddings normalised to unit length, clustered with k-means (k = 1,000). A cluster is a *scene
   group*.
3. The groups in a random order (fixed seed). Whole groups go to the test split until it has 10,000
   images or more, then whole groups go to the val split until it has 2,000 or more. The remaining
   images are the train pool.
4. The train subset is 8,000 images from the train pool (fixed seed), the same for all runs.

The splits are files of MIO-TCD image IDs (`splits/*.txt`); `scene_groups.csv` has the group of
every image.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .miotcd import SPECIFIC_CLASSES, boxes_per_class, category_id, write_ids

EMBED_SIZE = 224
N_GROUPS = 1000
TEST_IMAGES = 10_000
VAL_IMAGES = 2_000
TRAIN_IMAGES = 8_000
SPLIT_SEED = 0
MIN_TEST_BOXES = 100
MIN_VAL_BOXES = 30


class _Images(Dataset):
    def __init__(self, paths, size: int, mean, std):
        self.paths, self.size = list(paths), size
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            pixels = np.asarray(im.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))
        x = pixels.astype(np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((x - self.mean) / self.std)


@torch.no_grad()
def embed_images(paths, backbone, mean, std, device="cpu", batch_size: int = 128, num_workers: int = 2,
                 size: int = EMBED_SIZE, log_every: int = 50) -> np.ndarray:
    """[N, D] float32: for each image, the mean over positions of the last feature map of `backbone` (the
    detector's backbone, `model.model.backbone.backbone`), at `size` x `size` px. Not normalised."""
    backbone = backbone.to(device).eval()
    loader = DataLoader(_Images(paths, size, mean, std), batch_size=batch_size, num_workers=num_workers)
    out = []
    for i, batch in enumerate(loader):
        features = backbone(batch.to(device)).feature_maps[-1]
        out.append(features.float().mean(dim=(2, 3)).cpu().numpy())
        if log_every and i % log_every == 0:
            print(f"[scenes] embedded {min((i + 1) * batch_size, len(loader.dataset)):,} of {len(loader.dataset):,}", flush=True)
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cluster_scenes(embeddings: np.ndarray, k: int = N_GROUPS, seed: int = SPLIT_SEED) -> np.ndarray:
    """[N] scene group of each image: k-means on the unit-length embeddings."""
    from sklearn.cluster import KMeans  # imported here: only the split needs scikit-learn

    k = min(k, len(embeddings))
    return KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(unit_rows(embeddings)).astype(np.int64)


def split_by_group(groups: np.ndarray, n_test: int = TEST_IMAGES, n_val: int = VAL_IMAGES, seed: int = SPLIT_SEED,
                   class_boxes: np.ndarray | None = None,
                   min_boxes: tuple[int, int] = (MIN_TEST_BOXES, MIN_VAL_BOXES)) -> dict[str, np.ndarray]:
    """{"test", "val", "train_pool"} -> sorted image positions. Whole groups, in a random order, go to test
    until it has `n_test` images or more, then to val until it has `n_val` or more; the rest is the train pool.

    With `class_boxes` ([image, class] -> boxes, from `class_box_counts`) a split is also not complete before
    each class has its minimum of boxes (`min_boxes`: test, val). This is the rule of design §4.5 for a split
    with too few boxes of a class, "make the splits larger", with the smallest change: the order of the groups
    is the same, so the first groups of a split are those of the split without the rule."""
    groups = np.asarray(groups)
    group_ids, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(group_ids)
    row = {g: k for k, g in enumerate(group_ids.tolist())}
    if class_boxes is None:
        boxes_of_group = np.zeros((len(group_ids), 0), dtype=np.int64)
    else:
        boxes_of_group = np.zeros((len(group_ids), class_boxes.shape[1]), dtype=np.int64)
        np.add.at(boxes_of_group, inverse, class_boxes)
    targets = {"test": (n_test, min_boxes[0]), "val": (n_val, min_boxes[1])}
    images = {name: 0 for name in targets}
    boxes = {name: np.zeros(boxes_of_group.shape[1], dtype=np.int64) for name in targets}

    def complete(name: str) -> bool:
        n_images, minimum = targets[name]
        return images[name] >= n_images and bool((boxes[name] >= minimum).all())

    assigned = {}
    for g in order.tolist():
        name = "test" if not complete("test") else "val" if not complete("val") else "train_pool"
        assigned[g] = name
        if name in targets:
            images[name] += int(sizes[row[g]])
            boxes[name] += boxes_of_group[row[g]]
    names = np.array([assigned[g] for g in groups.tolist()])
    return {name: np.flatnonzero(names == name) for name in ("test", "val", "train_pool")}


def class_box_counts(coco: dict, image_ids) -> np.ndarray:
    """[image, specific class] -> boxes, with the images in the order of `image_ids` (COCO ids) and the classes
    in the order of `SPECIFIC_CLASSES`. The counts are those of `boxes_per_class`."""
    position = {int(i): k for k, i in enumerate(image_ids)}
    column = {category_id(name): j for j, name in enumerate(SPECIFIC_CLASSES)}
    out = np.zeros((len(position), len(SPECIFIC_CLASSES)), dtype=np.int64)
    for a in coco["annotations"]:
        k, j = position.get(a["image_id"]), column.get(a["category_id"])
        if k is not None and j is not None:
            out[k, j] += 1
    return out


def random_split(n: int, n_test: int = TEST_IMAGES, n_val: int = VAL_IMAGES, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """A split of single images with the sizes of the scene split: the comparison of the scene check."""
    order = np.random.default_rng(seed).permutation(n)
    return {"test": np.sort(order[:n_test]), "val": np.sort(order[n_test : n_test + n_val]),
            "train_pool": np.sort(order[n_test + n_val :])}


def train_subset(pool: np.ndarray, n: int = TRAIN_IMAGES, seed: int = SPLIT_SEED) -> np.ndarray:
    """`n` images of the train pool (all of it if it is smaller), sorted."""
    pool = np.asarray(pool)
    if len(pool) <= n:
        return np.sort(pool)
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def overlap_errors(splits: dict[str, np.ndarray], groups: np.ndarray) -> list[str]:
    """The "No overlap" check: no image and no scene group is in two of test, val and train pool."""
    errors = []
    names = [n for n in ("test", "val", "train_pool") if n in splits]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            images = np.intersect1d(splits[a], splits[b])
            shared = np.intersect1d(np.unique(groups[splits[a]]), np.unique(groups[splits[b]]))
            if len(images):
                errors.append(f"{len(images)} images are in {a} and in {b}")
            if len(shared):
                errors.append(f"{len(shared)} scene groups are in {a} and in {b}")
    if "train" in splits and len(np.setdiff1d(splits["train"], splits["train_pool"])):
        errors.append("the train subset has images that are not in the train pool")
    return errors


def class_count_errors(coco: dict, image_ids_by_split: dict[str, list[int]]) -> list[str]:
    """The "Class counts" check: each specific class has 100 boxes or more in test and 30 or more in val."""
    errors = []
    for split, minimum in (("test", MIN_TEST_BOXES), ("val", MIN_VAL_BOXES)):
        counts = boxes_per_class(coco, image_ids_by_split[split])
        for name in SPECIFIC_CLASSES:
            if counts[name] < minimum:
                errors.append(f"{split}: {name} has {counts[name]} boxes, {minimum} are needed; make the split larger")
    return errors


def nearest_neighbours(queries: np.ndarray, pool: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `queries`, (index of the most similar row of `pool`, their cosine similarity)."""
    queries, pool = unit_rows(queries), unit_rows(pool)
    index, similarity = [], []
    for start in range(0, len(queries), chunk):
        sims = queries[start : start + chunk] @ pool.T
        index.append(sims.argmax(axis=1))
        similarity.append(sims.max(axis=1))
    return np.concatenate(index), np.concatenate(similarity)


def scene_check_pairs(embeddings: np.ndarray, splits: dict[str, np.ndarray], n: int = 200,
                      seed: int = SPLIT_SEED) -> list[tuple[int, int, float]]:
    """The "Scene check": for `n` random test images, (test image, nearest train-pool image, cosine similarity),
    as image positions. A person looks at the pairs and counts those that show the same camera view."""
    test = np.asarray(splits["test"])
    picks = np.sort(np.random.default_rng(seed).choice(test, size=min(n, len(test)), replace=False))
    index, similarity = nearest_neighbours(embeddings[picks], embeddings[splits["train_pool"]])
    return [(int(q), int(splits["train_pool"][i]), float(s)) for q, i, s in zip(picks, index, similarity)]


def neighbour_similarity(embeddings: np.ndarray, seed: int = SPLIT_SEED) -> dict[str, float]:
    """Does the order of the image IDs show the camera? The mean cosine similarity of images that are next to
    each other in ID order, against that of random pairs. If the first is much higher, runs of IDs are scenes,
    and the design says to use that as the group and not to cluster."""
    unit = unit_rows(embeddings)
    adjacent = float((unit[:-1] * unit[1:]).sum(axis=1).mean())
    shuffled = unit[np.random.default_rng(seed).permutation(len(unit))]
    return {"adjacent_ids": adjacent, "random_pairs": float((unit * shuffled).sum(axis=1).mean())}


def write_groups(path, miotcd_ids: list[str], groups: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "scene_group"])
        writer.writerows(zip(miotcd_ids, np.asarray(groups).tolist()))


def read_groups(path) -> dict[str, int]:
    with open(path, newline="") as f:
        return {row["image_id"]: int(row["scene_group"]) for row in csv.DictReader(f)}


def write_split_files(out_dir, miotcd_ids: list[str], groups: np.ndarray, splits: dict[str, np.ndarray], meta: dict) -> None:
    """`test.txt`, `val.txt`, `train_pool.txt`, `train.txt`, `scene_groups.csv` and `split_meta.json` in `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array(miotcd_ids)
    for name, positions in splits.items():
        write_ids(out_dir / f"{name}.txt", ids[positions].tolist())
    write_groups(out_dir / "scene_groups.csv", miotcd_ids, groups)
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
