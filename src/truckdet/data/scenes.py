"""Scene groups and the split by scene (design §4.3, as changed by design change 1).

Frames from one camera show the same background, and a random image split puts the same camera in
train and test. `gt_train.csv` has no camera ID, so the project estimates one:

1. One embedding for each labelled image: the mean of the last feature map of the frozen backbone,
   at 224 px, with unit length. *Similarity* is the cosine similarity; *distance* is 1 - similarity.
2. Two images have a link if their similarity is at or above a limit. A *scene group* is a connected
   set of linked images (`link_groups`). So near images are in one group by construction.
3. An automatic search sets the limit (`search_limit`): of 17 candidates between the distance to the
   nearest image and the distance of random pairs, the lowest one before the groups join into one
   very large group.
4. The groups in a random order (fixed seed) go to the test split, then to the val split
   (`split_with_caps`). A group gives at most 300 images to test and at most 100 to val; its other
   images are not used. All other groups are the train pool, and the train subset is 8,000 images
   of it (fixed seed), the same for all runs.

Two automatic checks come before the scene check by a person: `group_check` (no image has an image
outside its group at or above the limit) and `split_check` (no test or val image has a train-pool
image at or above the limit).

The first method, k-means with k = 1,000, cut the frames of one camera into more than one group and
failed the scene check on MIO-TCD: `docs/build_notes.md` §8.

Each pass over all pairs of images calculates `unit[start:stop] @ unit.T` with the same blocks of
rows, so a pair has the same similarity in the search and in the checks. The groups are a function
of the embeddings and the limit; the float32 rounding of a similarity that is equal to the limit is
the one part that can depend on the machine.

The splits are files of MIO-TCD image IDs (`splits/*.txt`); `scene_groups.csv` has the group of
every image.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader, Dataset

from .miotcd import SPECIFIC_CLASSES, boxes_per_class, category_id, write_ids

EMBED_SIZE = 224
GRID = 4  # the second embedding (design change 1 §6): the mean in each cell of a 4 x 4 grid
TEST_IMAGES = 10_000
VAL_IMAGES = 2_000
TRAIN_IMAGES = 8_000
SPLIT_SEED = 0
MIN_TEST_BOXES = 100
MIN_VAL_BOXES = 30
NEIGHBOURS = 100  # the nearest images of each image that the first links come from
CANDIDATES = 17  # candidate limits, at equal steps of log distance
MAX_GROUP_SHARE = 0.20  # a limit is over-linked if the largest group has more than this share of the images
TEST_CAP = 300  # images that one group gives to the test split
VAL_CAP = 100  # images that one group gives to the val split
BLOCK = 2048  # rows of one block of a pass over all pairs
SPLIT_FILES = ("test.txt", "val.txt", "train_pool.txt", "train.txt", "unused.txt", "scene_groups.csv")


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
                 size: int = EMBED_SIZE, log_every: int = 50, grid: int | None = None) -> np.ndarray:
    """[N, D] for each image, from the last feature map of `backbone` (the detector's backbone,
    `model.model.backbone.backbone`) at `size` x `size` px. Not normalised.

    Default: the mean over all positions (D = channels, float32). With `grid` (the second embedding): the mean
    in each cell of a `grid` x `grid` grid, the cells joined (D = channels x grid x grid, float16 for storage).
    The mean tells what is in a picture; the grid also tells where it is."""
    backbone = backbone.to(device).eval()
    loader = DataLoader(_Images(paths, size, mean, std), batch_size=batch_size, num_workers=num_workers)
    dtype = np.float16 if grid else np.float32
    out = []
    for i, batch in enumerate(loader):
        features = backbone(batch.to(device)).feature_maps[-1].float()
        if grid:
            features = torch.nn.functional.adaptive_avg_pool2d(features, grid).flatten(1)
        else:
            features = features.mean(dim=(2, 3))
        out.append(features.cpu().numpy().astype(dtype))
        if log_every and i % log_every == 0:
            print(f"[scenes] embedded {min((i + 1) * batch_size, len(loader.dataset)):,} of {len(loader.dataset):,}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 0), dtype)


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prepare_embeddings(raw: np.ndarray, centre: bool = False) -> np.ndarray:
    """[N, D] float32 rows of unit length. With `centre` (the grid embedding) the mean of all images is
    subtracted first. One copy, changed in place: the grid embedding of MIO-TCD is 2.7 GB."""
    x = np.array(raw, dtype=np.float32)
    if centre:
        x -= x.mean(axis=0, keepdims=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def as_tensor(unit, device="cpu") -> torch.Tensor:
    """The prepared embeddings on `device`. The functions below take this tensor or the numpy array."""
    if torch.is_tensor(unit):
        return unit
    return torch.from_numpy(np.ascontiguousarray(unit, dtype=np.float32)).to(device)


# --- Passes over all pairs of images ---------------------------------------------------------


@torch.no_grad()
def _best_match(unit: torch.Tensor, exclude, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """For each image, the most similar image that is not excluded: (position [N], similarity [N]); (-1, -inf)
    if all are excluded. `exclude(start, stop)` gives the pairs to leave out for the rows start..stop: a bool
    tensor [stop - start, N], or [N] for the same columns in each row."""
    n = len(unit)
    index, best = np.full(n, -1, dtype=np.int64), np.full(n, -np.inf, dtype=np.float32)
    for start in range(0, n, block):
        sims = unit[start : start + block] @ unit.T
        stop = start + len(sims)
        sims.masked_fill_(exclude(start, stop), float("-inf"))
        value, position = sims.max(dim=1)
        best[start:stop], index[start:stop] = value.cpu().numpy(), position.cpu().numpy()
    index[np.isneginf(best)] = -1
    return index, best


@torch.no_grad()
def neighbour_lists(unit, k: int = NEIGHBOURS, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """For each image, its `k` most similar other images, the most similar first: (position [N, k],
    similarity [N, k]). k is at most N - 1."""
    unit = as_tensor(unit)
    n = len(unit)
    k = max(min(k, n - 1), 0)
    index, similarity = np.empty((n, k), dtype=np.int32), np.empty((n, k), dtype=np.float32)
    for start in range(0, n, block):
        sims = unit[start : start + block] @ unit.T
        rows = torch.arange(len(sims), device=sims.device)
        sims[rows, rows + start] = float("-inf")  # not the image itself; an exact copy of it stays
        value, position = sims.topk(k, dim=1)
        index[start : start + len(sims)], similarity[start : start + len(sims)] = position.cpu().numpy(), value.cpu().numpy()
    return index, similarity


def nearest_outside(unit, groups: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """For each image, its most similar image outside its group: (position, similarity); (-1, -inf) if there
    is one group only."""
    unit = as_tensor(unit)
    g = torch.as_tensor(np.asarray(groups), device=unit.device)
    return _best_match(unit, lambda start, stop: g[start:stop, None] == g[None, :], block)


@torch.no_grad()
def row_dots(unit, a, b, block: int = 8192) -> np.ndarray:
    """The similarity of image a[k] and image b[k], for each k."""
    unit = as_tensor(unit)
    a, b = np.asarray(a), np.asarray(b)
    out = [np.zeros(0, dtype=np.float32)]
    for start in range(0, len(a), block):
        i = torch.as_tensor(a[start : start + block], device=unit.device)
        j = torch.as_tensor(b[start : start + block], device=unit.device)
        out.append((unit[i] * unit[j]).sum(dim=1).cpu().numpy())
    return np.concatenate(out)


# --- Scene groups for a limit (design change 1 §3) ----------------------------------------------


def canonical_groups(labels: np.ndarray) -> np.ndarray:
    """The same groups, numbered in the order of their first image: the numbers do not depend on the code that
    found the groups."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[inverse]


def link_groups(unit, lists: tuple[np.ndarray, np.ndarray], limit: float, links: np.ndarray | None = None,
                block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """([N] scene group of each image, [M, 2] links that the lists did not hold).

    A link joins two images with a similarity of `limit` or more, and a scene group is a connected set of the
    linked images. The method (design change 1 §3.6):

    1. `lists` (`neighbour_lists`) gives the first links.
    2. The connected sets of the links.
    3. Each image whose most similar image outside its group is at or above the limit gets that link; then
       step 2 again, until there is no such image. The lists do not hold all links of an image in a large
       group, so this step is necessary; it ends with a pass that finds nothing, which is the group check.

    `links` are the added links of a higher limit: they are links at this limit too."""
    unit = as_tensor(unit)
    index, similarity = lists
    n = len(index)
    limit = np.float32(limit)
    rows, columns = np.nonzero(similarity >= limit)
    first = np.stack([rows, index[rows, columns]], axis=1).astype(np.int64)
    links = np.zeros((0, 2), dtype=np.int64) if links is None else np.asarray(links, dtype=np.int64).reshape(-1, 2)
    while True:
        pairs = np.concatenate([first, links])
        graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        labels = connected_components(graph, directed=False)[1]
        position, value = nearest_outside(unit, labels, block)
        more = np.flatnonzero(value >= limit)
        if not len(more):
            return canonical_groups(labels), links
        links = np.concatenate([links, np.stack([more, position[more]], axis=1)])


def link_distances(unit, lists: tuple[np.ndarray, np.ndarray], seed: int = SPLIT_SEED) -> tuple[float, float]:
    """(d_near, d_rand): the median, for all images, of the distance to the nearest image, and the median
    distance of N random pairs (N images)."""
    unit = as_tensor(unit)
    n = len(unit)
    d_near = float(np.median(1.0 - lists[1][:, 0].astype(np.float64)))
    i = np.arange(n)
    j = (i + np.random.default_rng(seed).integers(1, n, size=n)) % n  # never the image itself
    d_rand = float(np.median(1.0 - row_dots(unit, i, j).astype(np.float64)))
    return d_near, d_rand


def candidate_limits(d_near: float, d_rand: float, n: int = CANDIDATES) -> list[float]:
    """The candidate limits, the highest first: distances at equal steps of log distance from d_near to d_rand,
    L_k = 1 - d_near x (d_rand / d_near)^(k / (n - 1)). Each limit is a float32 number, as the similarities are.
    A d_near below 1e-6 (most images have an exact copy) counts as 1e-6."""
    d_near = max(d_near, 1e-6)
    if not d_rand > d_near:
        raise ValueError(f"the nearest image is not nearer than a random image (distances {d_near:.2g} and "
                         f"{d_rand:.2g}): this embedding has no near images to link")
    distances = d_near * (d_rand / d_near) ** (np.arange(n) / (n - 1))
    return [float(x) for x in (1.0 - distances).astype(np.float32)]


def search_limit(unit, lists: tuple[np.ndarray, np.ndarray], limits: list[float], try_split,
                 max_share: float = MAX_GROUP_SHARE, block: int = BLOCK, log=None) -> dict:
    """The search of design change 1 §3.5: go from the highest candidate to the lowest, stop at the first
    over-linked limit, and choose the candidate before it. A lower limit can only join groups, so the search
    has one stop point.

    A limit is over-linked if the largest group has more than `max_share` of the images, or if the split is
    not possible with its groups: `try_split(groups)` gives (the split, the causes; no cause: possible).

    Returns {"candidates": one row for each limit that was tried, "chosen": None or {"limit", "candidate",
    "groups", "split"}}. No chosen limit: the first candidate is over-linked."""
    unit = as_tensor(unit)
    rows, chosen, links = [], None, None
    for k, limit in enumerate(limits):
        groups, links = link_groups(unit, lists, limit, links, block)
        sizes = np.bincount(groups)
        share = float(sizes.max() / len(groups))
        split, cause = None, None
        if share > max_share:
            cause = f"the largest group has {share:.1%} of the images"
        else:
            split, problems = try_split(groups)
            if problems:
                cause = "the split is not possible: " + "; ".join(problems)
        rows.append({"limit": limit, "groups": int(len(sizes)), "largest_group_share": share, "over_linked": cause})
        if log:
            log(rows[-1])
        if cause:
            break
        chosen = {"limit": limit, "candidate": k, "groups": groups, "split": split}
    return {"candidates": rows, "chosen": chosen}


# --- The split (design change 1 §4) ------------------------------------------------------------


def split_with_caps(groups: np.ndarray, n_test: int = TEST_IMAGES, n_val: int = VAL_IMAGES,
                    caps: tuple[int | None, int | None] = (TEST_CAP, VAL_CAP), seed: int = SPLIT_SEED,
                    class_boxes: np.ndarray | None = None,
                    min_boxes: tuple[int, int] = (MIN_TEST_BOXES, MIN_VAL_BOXES)) -> dict[str, np.ndarray]:
    """{"test", "val", "train_pool", "unused"} -> sorted image positions.

    The groups in a random order go to the test split until it has `n_test` images or more and each class has
    `min_boxes[0]` boxes or more, then to the val split in the same way. A group gives at most `caps` images
    (test, val), drawn at random; its other images are "unused", in no split. All images of all other groups
    are the train pool. Images and boxes are counted on the images that a split uses.

    `class_boxes`: [image, class] -> boxes (`class_box_counts`); None: no class rule. The draw of a group
    depends on the seed and on the group only, not on the groups before it. A split can be not complete (the
    groups end): `split_problems` tells."""
    groups = np.asarray(groups)
    group_ids, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    by_group = np.argsort(inverse, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    if class_boxes is None:
        class_boxes = np.zeros((len(groups), 0), dtype=np.int64)
    targets = {"test": (n_test, caps[0], min_boxes[0]), "val": (n_val, caps[1], min_boxes[1])}
    images = {name: 0 for name in targets}
    boxes = {name: np.zeros(class_boxes.shape[1], dtype=np.int64) for name in targets}
    parts = {"test": [], "val": [], "unused": []}
    taken = np.zeros(len(group_ids), dtype=bool)

    def complete(name: str) -> bool:
        n_images, _, minimum = targets[name]
        return images[name] >= n_images and bool((boxes[name] >= minimum).all())

    for row in np.random.default_rng(seed).permutation(len(group_ids)).tolist():
        name = "test" if not complete("test") else "val" if not complete("val") else None
        if name is None:
            break
        members, cap = by_group[bounds[row] : bounds[row + 1]], targets[name][1]
        used = members
        if cap is not None and len(members) > cap:
            used = np.sort(np.random.default_rng([seed, row]).choice(members, size=cap, replace=False))
            parts["unused"].append(np.setdiff1d(members, used))
        parts[name].append(used)
        taken[row] = True
        images[name] += len(used)
        boxes[name] += class_boxes[used].sum(axis=0)
    out = {name: np.sort(np.concatenate(p)).astype(np.int64) if p else np.zeros(0, dtype=np.int64)
           for name, p in parts.items()}
    out["train_pool"] = np.flatnonzero(~taken[inverse])
    return {name: out[name] for name in ("test", "val", "train_pool", "unused")}


def split_problems(splits: dict[str, np.ndarray], n_test: int = TEST_IMAGES, n_val: int = VAL_IMAGES,
                   n_train: int = TRAIN_IMAGES, class_boxes: np.ndarray | None = None,
                   min_boxes: tuple[int, int] = (MIN_TEST_BOXES, MIN_VAL_BOXES)) -> list[str]:
    """Why the split is not possible with these groups; an empty list: it is possible. Not possible: the groups
    end before the val split is complete, or the train pool has fewer than `n_train` images."""
    def complete(name: str, n_images: int, minimum: int) -> bool:
        enough = class_boxes is None or bool((class_boxes[splits[name]].sum(axis=0) >= minimum).all())
        return len(splits[name]) >= n_images and enough

    if not complete("test", n_test, min_boxes[0]):
        return ["the groups end before the test split is complete"]
    problems = [] if complete("val", n_val, min_boxes[1]) else ["the groups end before the val split is complete"]
    if len(splits["train_pool"]) < n_train:
        problems.append(f"the train pool has {len(splits['train_pool']):,} images, {n_train:,} are needed")
    return problems


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


# --- Checks ------------------------------------------------------------------------------------


def overlap_errors(splits: dict[str, np.ndarray], groups: np.ndarray) -> list[str]:
    """The "No overlap" check: no image and no scene group is in two of test, val and train pool. The unused
    images (over the cap of their group) are in no split."""
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
        if "unused" in splits and len(np.intersect1d(splits[a], splits["unused"])):
            errors.append(f"{len(np.intersect1d(splits[a], splits['unused']))} unused images are in {a}")
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


def group_check(unit, groups: np.ndarray, limit: float, block: int = BLOCK) -> dict:
    """The group check (design change 1 §5.1): for each image, its most similar image outside its group. No
    such similarity is at or above the limit: "failures" must be 0. This check decides if the groups are
    correct, whatever code made them."""
    _, value = nearest_outside(unit, groups, block)
    highest = float(value.max()) if len(value) and np.isfinite(value.max()) else None
    return {"failures": int((value >= np.float32(limit)).sum()), "highest_similarity_outside_a_group": highest}


def split_check(unit, splits: dict[str, np.ndarray], groups: np.ndarray, limit: float,
                min_groups: tuple[int, int], block: int = BLOCK) -> tuple[list[str], dict]:
    """The split check (design change 1 §5.2): (errors, numbers).

    - No test image and no val image has a train-pool image at or above the limit.
    - No test image has a val image at or above the limit.
    - The test split has `min_groups[0]` groups or more, and the val split has `min_groups[1]` or more.

    The first two lines follow from the group check; they are a guard for the split code."""
    unit = as_tensor(unit)
    groups = np.asarray(groups)

    def best_in(name: str) -> np.ndarray:  # for each image, its highest similarity to an image of this split
        outside = torch.ones(len(unit), dtype=torch.bool, device=unit.device)
        outside[torch.as_tensor(np.asarray(splits[name], dtype=np.int64), device=unit.device)] = False
        return _best_match(unit, lambda start, stop: outside, block)[1]

    to_pool, to_val = best_in("train_pool"), best_in("val")
    errors, numbers = [], {}
    for a, b, value in (("test", "train_pool", to_pool), ("val", "train_pool", to_pool), ("test", "val", to_val)):
        v = value[splits[a]]
        count = int((v >= np.float32(limit)).sum())
        highest = float(v.max()) if len(v) and np.isfinite(v.max()) else None
        numbers[f"{a} to {b}"] = {"at_or_above_the_limit": count, "highest_similarity": highest}
        if count:
            errors.append(f"{count} {a} images have a {b} image at or above the limit")
    for name, minimum in zip(("test", "val"), min_groups):
        n_groups = int(len(np.unique(groups[splits[name]])))
        numbers[f"{name} groups"] = n_groups
        if n_groups < minimum:
            errors.append(f"the {name} split has {n_groups} groups, {minimum} are needed")
    return errors, numbers


@torch.no_grad()
def scene_check_pairs(unit, splits: dict[str, np.ndarray], n: int = 200, seed: int = SPLIT_SEED) -> list[tuple[int, int, float]]:
    """The "Scene check": for `n` random test images, (test image, nearest train-pool image, cosine similarity),
    as image positions. A person looks at the pairs and counts those that show the same camera view."""
    unit = as_tensor(unit)
    test, pool = np.asarray(splits["test"]), np.asarray(splits["train_pool"])
    picks = np.sort(np.random.default_rng(seed).choice(test, size=min(n, len(test)), replace=False))
    sims = unit[torch.as_tensor(picks, device=unit.device)] @ unit.T
    value, position = sims[:, torch.as_tensor(pool, device=unit.device)].max(dim=1)
    return [(int(q), int(pool[i]), float(s)) for q, i, s in zip(picks, position.cpu().numpy(), value.cpu().numpy())]


def neighbour_similarity(unit, seed: int = SPLIT_SEED) -> dict[str, float]:
    """Does the order of the image IDs show the camera? The mean cosine similarity of images that are next to
    each other in ID order, against that of random pairs (`unit`: the prepared embeddings). If the first is
    much higher, runs of IDs are scenes, and the design says to use that as the group and not to link."""
    unit = as_tensor(unit)
    n = len(unit)
    adjacent = float(row_dots(unit, np.arange(n - 1), np.arange(1, n)).astype(np.float64).mean())
    shuffled = np.random.default_rng(seed).permutation(n)
    return {"adjacent_ids": adjacent, "random_pairs": float(row_dots(unit, np.arange(n), shuffled).astype(np.float64).mean())}


# --- Files -------------------------------------------------------------------------------------


def write_groups(path, miotcd_ids: list[str], groups: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "scene_group"])
        writer.writerows(zip(miotcd_ids, np.asarray(groups).tolist()))


def read_groups(path) -> dict[str, int]:
    with open(path, newline="") as f:
        return {row["image_id"]: int(row["scene_group"]) for row in csv.DictReader(f)}


def write_split_files(out_dir, miotcd_ids: list[str], groups: np.ndarray, splits: dict[str, np.ndarray], meta: dict) -> None:
    """`<split>.txt` for each split (`test`, `val`, `train_pool`, `train`, `unused`), `scene_groups.csv` and
    `split_meta.json` in `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array(miotcd_ids)
    for name, positions in splits.items():
        write_ids(out_dir / f"{name}.txt", ids[positions].tolist())
    write_groups(out_dir / "scene_groups.csv", miotcd_ids, groups)
    write_meta(out_dir, meta)


def write_meta(out_dir, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def remove_split_files(out_dir) -> list[str]:
    """Remove the split lists of an earlier run from `out_dir` (the names of `SPLIT_FILES` only), so that a run
    that gives no split leaves no old split behind. Returns the names that were removed."""
    removed = []
    for name in SPLIT_FILES:
        path = Path(out_dir) / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed
