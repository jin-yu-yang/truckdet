import json
import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tests.tiny import TINY_SIZE, make_tiny_rfdetr
from truckdet.data import scenes
from truckdet.data.miotcd import SPECIFIC_CLASSES, boxes_per_class, load_annotations, read_ids


def cameras(n_cam: int = 10, per: int = 20, dim: int = 8, spread: float = 0.05, seed: int = 0):
    """(unit-length frames [n_cam * per, dim] float32, the camera of each frame): frames near the centre of their camera."""
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(n_cam), per)
    x = rng.normal(size=(n_cam, dim))[truth] * 5 + rng.normal(size=(n_cam * per, dim)) * spread
    return scenes.unit_rows(x).astype(np.float32), truth


def brute_force_groups(unit: np.ndarray, limit: float) -> np.ndarray:
    """Design §4.3: link each pair of images with a similarity of the limit or more; a group is a connected set."""
    linked = (unit @ unit.T) >= limit
    return scenes.canonical_groups(connected_components(csr_matrix(linked), directed=False)[1])


def clear_limits(unit: np.ndarray, quantiles) -> list[float]:
    """Limits that are not near the similarity of a pair: the middle of two neighbours in the sorted similarities."""
    sims = np.sort((unit @ unit.T)[np.triu_indices(len(unit), 1)].astype(np.float64))
    out = []
    for q in quantiles:
        k = int(q * (len(sims) - 2))
        while k + 2 < len(sims) and sims[k + 1] - sims[k] < 2e-6:  # a gap that float32 rounding cannot close
            k += 1
        out.append(float(np.float32((sims[k] + sims[k + 1]) / 2)))
    return out


# --- Passes over all pairs -------------------------------------------------------------------


def test_neighbour_lists_are_the_nearest_images():
    rng = np.random.default_rng(0)
    unit = scenes.unit_rows(rng.normal(size=(150, 8))).astype(np.float32)
    index, similarity = scenes.neighbour_lists(unit, k=5, block=64)  # 3 blocks of rows
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    expected = np.argsort(-sims, axis=1)[:, :5]
    assert index.shape == similarity.shape == (150, 5) and np.array_equal(index, expected)
    assert np.allclose(similarity, np.take_along_axis(sims, expected, axis=1), atol=1e-6)
    assert (np.diff(similarity, axis=1) <= 0).all()  # the most similar first
    assert scenes.neighbour_lists(unit[:4], k=100)[0].shape == (4, 3)  # k is at most the number of other images
    copies = np.concatenate([unit[:3], unit[:3]])
    assert np.allclose(scenes.neighbour_lists(copies, k=1)[1], 1.0, atol=1e-6)  # an exact copy is a neighbour


def test_nearest_outside_is_the_most_similar_image_of_another_group():
    unit, truth = cameras(n_cam=5, per=12)
    position, value = scenes.nearest_outside(unit, truth, block=16)
    sims = unit @ unit.T
    sims[truth[:, None] == truth[None, :]] = -np.inf
    assert np.array_equal(position, sims.argmax(axis=1)) and np.allclose(value, sims.max(axis=1), atol=1e-6)
    assert (truth[position] != truth).all()
    position, value = scenes.nearest_outside(unit, np.zeros(60, dtype=int))  # one group: no image is outside
    assert (position == -1).all() and np.isneginf(value).all()


# --- Scene groups for a limit ---------------------------------------------------------------


def test_link_groups_are_the_connected_sets_of_the_links():
    """Against a brute-force result on a small set, with neighbour lists that are too short to hold all links."""
    rng = np.random.default_rng(1)
    truth = np.repeat(np.arange(12), 25)
    centres = rng.normal(size=(12, 6)) * np.linspace(0.5, 3, 12)[:, None]  # near cameras and far cameras
    unit = scenes.unit_rows(centres[truth] + rng.normal(size=(300, 6)) * 0.15).astype(np.float32)
    lists = scenes.neighbour_lists(unit, k=3, block=64)
    limits = clear_limits(unit, (0.999, 0.995, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6))
    assert len(limits) >= 6
    added, short = 0, 0
    for limit in limits:
        groups, links = scenes.link_groups(unit, lists, limit, block=64)
        expected = brute_force_groups(unit, limit)
        assert np.array_equal(groups, expected), limit
        assert scenes.group_check(unit, groups, limit, block=64)["failures"] == 0
        rows, columns = np.nonzero(lists[1] >= np.float32(limit))
        from_lists = csr_matrix((np.ones(len(rows), dtype=bool), (rows, lists[0][rows, columns])), shape=(300, 300))
        short += not np.array_equal(scenes.canonical_groups(connected_components(from_lists, directed=False)[1]), expected)
        added += len(links) > 0
    assert short >= 2 and added >= 2  # the lists alone give other groups: the added links are necessary
    # a group number is the rank of the group's first image
    first = [int(np.flatnonzero(groups == g)[0]) for g in range(groups.max() + 1)]
    assert groups[0] == 0 and first == sorted(first)
    # the added links of a higher limit are links at a lower limit too
    high, low = limits[1], limits[4]
    carried = scenes.link_groups(unit, lists, low, scenes.link_groups(unit, lists, high, block=64)[1], block=64)[0]
    assert np.array_equal(carried, brute_force_groups(unit, low))
    # a lower limit can only join groups
    fine, coarse = brute_force_groups(unit, high), brute_force_groups(unit, low)
    assert all(len(set(coarse[fine == g].tolist())) == 1 for g in range(fine.max() + 1))


def test_canonical_groups():
    assert scenes.canonical_groups(np.array([7, 7, 3, 9, 3, 7])).tolist() == [0, 0, 1, 2, 1, 0]
    assert scenes.canonical_groups(np.array([5, 4, 5])).tolist() == scenes.canonical_groups(np.array([0, 9, 0])).tolist()


def test_candidate_limits_are_equal_steps_of_log_distance():
    limits = scenes.candidate_limits(0.004, 0.086)  # the numbers of MIO-TCD: similarities 0.996 and 0.914
    assert len(limits) == scenes.CANDIDATES == 17
    assert limits[0] == pytest.approx(0.996, abs=1e-6) and limits[-1] == pytest.approx(0.914, abs=1e-6)
    distances = 1 - np.array(limits)
    assert np.allclose(distances[1:] / distances[:-1], (0.086 / 0.004) ** (1 / 16), rtol=1e-3)  # each step: x 1.21
    assert all(float(np.float32(v)) == v for v in limits) and limits == sorted(limits, reverse=True)
    with pytest.raises(ValueError, match="no near images to link"):
        scenes.candidate_limits(0.09, 0.086)
    assert scenes.candidate_limits(0.0, 0.01)[0] == pytest.approx(1 - 1e-6, abs=1e-7)  # exact copies: a floor


def test_link_distances_are_the_two_medians():
    unit, truth = cameras(n_cam=10, per=20)
    lists = scenes.neighbour_lists(unit)
    d_near, d_rand = scenes.link_distances(unit, lists, seed=0)
    sims = (unit @ unit.T).astype(np.float64)
    np.fill_diagonal(sims, -np.inf)
    assert d_near == pytest.approx(float(np.median(1 - sims.max(axis=1))), abs=1e-6)
    assert d_near < 1e-3 < 0.3 < d_rand  # a frame of the same camera is near; a random frame is far
    assert (d_near, d_rand) == scenes.link_distances(unit, lists, seed=0)


def two_levels():
    """6 cameras of 20 frames, in 2 regions: frames of a camera are very near, cameras of a region are near."""
    rng = np.random.default_rng(0)
    region = rng.normal(size=(2, 16)) * 5
    centre = region[np.repeat([0, 1], 3)] + rng.normal(size=(6, 16)) * 0.6
    truth = np.repeat(np.arange(6), 20)
    unit = scenes.unit_rows(centre[truth] + rng.normal(size=(120, 16)) * 0.01).astype(np.float32)
    sims = unit @ unit.T
    same_camera = truth[:, None] == truth[None, :]
    same_region = (truth[:, None] // 3) == (truth[None, :] // 3)
    cameras_limit = float(np.float32((sims[same_camera].min() + sims[same_region & ~same_camera].max()) / 2))
    regions_limit = float(np.float32((sims[same_region].min() + sims[~same_region].max()) / 2))
    return unit, truth, cameras_limit, regions_limit


def test_search_stops_at_the_first_over_linked_limit():
    unit, truth, cameras_limit, regions_limit = two_levels()
    lists = scenes.neighbour_lists(unit, k=5)
    asked = []

    def try_split(groups):  # possible with 3 groups or more
        asked.append(int(groups.max()) + 1)
        return {"groups": int(groups.max()) + 1}, [] if groups.max() + 1 >= 3 else ["the groups end before the val split is complete"]

    rows = []
    result = scenes.search_limit(unit, lists, [cameras_limit, regions_limit, 0.0], try_split, log=rows.append)
    chosen = result["chosen"]
    assert chosen["limit"] == cameras_limit and chosen["candidate"] == 0 and chosen["split"] == {"groups": 6}
    assert np.array_equal(chosen["groups"], truth)  # the 6 cameras, numbered in the order of their first image
    assert rows == result["candidates"] and len(rows) == 2  # the search stops: the third limit is not tried
    assert rows[0] == {"limit": cameras_limit, "groups": 6, "largest_group_share": pytest.approx(1 / 6), "over_linked": None}
    assert rows[1]["groups"] == 2 and rows[1]["over_linked"] == "the largest group has 50.0% of the images"
    assert asked == [6]  # a limit with a very large group does not get a split

    # the second cause: the largest group is small enough, and the split is not possible
    result = scenes.search_limit(unit, lists, [cameras_limit, regions_limit], try_split, max_share=0.6)
    assert result["chosen"]["limit"] == cameras_limit
    assert result["candidates"][1]["over_linked"] == "the split is not possible: the groups end before the val split is complete"

    # the first candidate is over-linked: there is no limit
    none = scenes.search_limit(unit, lists, [regions_limit, 0.0], try_split)
    assert none["chosen"] is None and len(none["candidates"]) == 1
    # no candidate is over-linked: the lowest one is chosen
    last = scenes.search_limit(unit, lists, [0.99999, cameras_limit], try_split)
    assert last["chosen"]["limit"] == cameras_limit and last["chosen"]["candidate"] == 1


# --- The split -------------------------------------------------------------------------------


def group_sizes(sizes) -> np.ndarray:
    groups = np.repeat(np.arange(len(sizes)), sizes)
    return groups[np.random.default_rng(3).permutation(len(groups))]  # the images of a group are not next to each other


def test_split_with_caps():
    sizes = [500, 400, 50, 30, 20, 350, 10, 120, 80, 60, 45, 35, 250, 15, 90, 70, 25, 130, 40, 55]
    groups = group_sizes(sizes)
    n = len(groups)
    splits = scenes.split_with_caps(groups, n_test=600, n_val=150, caps=(300, 100), seed=0)
    assert list(splits) == ["test", "val", "train_pool", "unused"]
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(n))  # every image in one place
    assert not scenes.overlap_errors(splits, groups)
    assert all(np.array_equal(v, np.sort(v)) for v in splits.values())

    order = np.random.default_rng(0).permutation(len(sizes)).tolist()
    in_test, in_val = np.unique(groups[splits["test"]]).tolist(), np.unique(groups[splits["val"]]).tolist()
    assert set(in_test) == set(order[: len(in_test)])  # a run of the random order, then the next run
    assert set(in_val) == set(order[len(in_test) : len(in_test) + len(in_val)])
    used = {"test": np.bincount(groups[splits["test"]], minlength=len(sizes)),
            "val": np.bincount(groups[splits["val"]], minlength=len(sizes))}
    for name, members, cap, target in (("test", in_test, 300, 600), ("val", in_val, 100, 150)):
        assert all(used[name][g] == min(sizes[g], cap) for g in members)  # a group gives all its images, or the cap
        assert len(splits[name]) >= target > len(splits[name]) - used[name][order[len(in_test) - 1] if name == "test"
                                                                           else order[len(in_test) + len(in_val) - 1]]
    assert any(sizes[g] > 300 for g in in_test) or any(sizes[g] > 100 for g in in_val)  # the case has a cap at work
    # unused: the images of a test or val group that are over its cap; the train pool: all images of all other groups
    over = sum(sizes[g] - 300 for g in in_test if sizes[g] > 300) + sum(sizes[g] - 100 for g in in_val if sizes[g] > 100)
    assert len(splits["unused"]) == over > 0 and set(np.unique(groups[splits["unused"]]).tolist()) <= set(in_test + in_val)
    assert np.array_equal(splits["train_pool"], np.flatnonzero(~np.isin(groups, in_test + in_val)))

    again = scenes.split_with_caps(groups, 600, 150, (300, 100), seed=0)
    assert all(np.array_equal(splits[name], again[name]) for name in splits)
    assert not np.array_equal(scenes.split_with_caps(groups, 600, 150, (300, 100), seed=1)["test"], splits["test"])
    # the draw of a group depends on the seed and the group only: a larger test split keeps the images of the smaller
    larger = scenes.split_with_caps(groups, 900, 150, (300, 100), seed=0)
    assert set(splits["test"].tolist()) < set(larger["test"].tolist())
    # no cap: whole groups, and no image is unused
    whole = scenes.split_with_caps(groups, 600, 150, (None, None), seed=0)
    assert len(whole["unused"]) == 0 and all(used == sizes[g] for g, used in zip(*np.unique(groups[whole["test"]], return_counts=True)))


def test_split_takes_groups_until_each_class_has_its_minimum():
    """Design §4.3: a split is complete when it has its images and each class has its boxes, counted on the images
    that the split uses. On MIO-TCD a val split of 2,016 images had 5 `bicycle` boxes."""
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=1000)
    class_boxes = np.zeros((1000, 2), dtype=np.int64)
    class_boxes[:, 0] = 1  # a common class: one box in every image
    class_boxes[np.isin(groups, [3, 8, 17, 21, 29, 33]), 1] = 1  # a rare class: only six scenes have it
    plain = scenes.split_with_caps(groups, 200, 60, (None, None), seed=0)
    assert class_boxes[plain["val"], 1].sum() < 20  # the case: too few boxes of the rare class with no class rule
    full = scenes.split_with_caps(groups, 200, 60, (None, None), seed=0, class_boxes=class_boxes, min_boxes=(20, 20))
    for name, target in (("test", 200), ("val", 60)):
        assert len(full[name]) >= target and (class_boxes[full[name]].sum(axis=0) >= 20).all()
    assert set(plain["test"].tolist()) <= set(full["test"].tolist())  # the same order of groups, and more of them
    assert not scenes.split_problems(full, 200, 60, 100, class_boxes, (20, 20))
    # with a cap the boxes are counted on the images that the split uses
    capped = scenes.split_with_caps(groups, 100, 30, (10, 10), seed=0, class_boxes=class_boxes, min_boxes=(20, 20))
    assert (class_boxes[capped["test"]].sum(axis=0) >= 20).all() and (class_boxes[capped["val"]].sum(axis=0) >= 20).all()
    assert np.bincount(groups[capped["test"]]).max() == 10 and len(capped["unused"]) > 0


def test_split_problems():
    groups = group_sizes([50, 40, 30, 20, 10])
    ok = scenes.split_with_caps(groups, 40, 20, (None, None), seed=0)
    assert scenes.split_problems(ok, 40, 20, n_train=10) == []
    assert scenes.split_problems(ok, 40, 20, n_train=1000) == [f"the train pool has {len(ok['train_pool'])} images, 1,000 are needed"]
    short_val = scenes.split_with_caps(groups, 100, 60, (None, None), seed=0)
    assert scenes.split_problems(short_val, 100, 60, n_train=0)[0] == "the groups end before the val split is complete"
    short_test = scenes.split_with_caps(groups, 500, 20, (None, None), seed=0)
    assert scenes.split_problems(short_test, 500, 20, n_train=10) == ["the groups end before the test split is complete"]
    capped = scenes.split_with_caps(groups, 100, 20, (20, 20), seed=0)  # 5 groups give 100 images at most: none is left
    assert len(capped["test"]) == 90 and scenes.split_problems(capped, 100, 20, 10) == ["the groups end before the test split is complete"]
    boxes = np.ones((150, 1), dtype=np.int64)
    assert scenes.split_problems(ok, 40, 20, 10, boxes, (10, 10)) == []
    assert scenes.split_problems(ok, 40, 20, 10, boxes, (10, 1000)) == ["the groups end before the val split is complete"]


def test_class_box_counts_are_those_of_boxes_per_class(data_root):
    coco = load_annotations(data_root / "annotations.json")
    ids = [im["id"] for im in coco["images"]][::-1]
    counts = scenes.class_box_counts(coco, ids)
    assert counts.shape == (len(ids), 10)
    total = boxes_per_class(coco)
    assert counts.sum(axis=0).tolist() == [total[name] for name in SPECIFIC_CLASSES]
    one = boxes_per_class(coco, ids[:1])
    assert counts[0].tolist() == [one[name] for name in SPECIFIC_CLASSES]


def test_overlap_errors_are_found():
    groups = np.array([0, 0, 1, 1, 2, 2])
    bad = {"test": np.array([0, 2]), "val": np.array([1]), "train_pool": np.array([2, 3, 4, 5])}
    errors = scenes.overlap_errors(bad, groups)
    assert any("1 images are in test and in train_pool" in e for e in errors)
    assert any("scene groups are in test and in val" in e for e in errors)
    good = {"test": np.array([0]), "val": np.array([2, 3]), "train_pool": np.array([4, 5]), "train": np.array([4]),
            "unused": np.array([1])}  # image 1 is over the cap of its test group: in no split
    assert scenes.overlap_errors(good, groups) == []
    assert scenes.overlap_errors({**good, "train": np.array([0])}, groups) == ["the train subset has images that are not in the train pool"]
    assert scenes.overlap_errors({**good, "unused": np.array([0, 1])}, groups) == ["1 unused images are in test"]


def test_train_subset():
    pool = np.arange(100, 400)
    subset = scenes.train_subset(pool, n=50, seed=0)
    assert len(subset) == 50 == len(set(subset.tolist())) and set(subset.tolist()) <= set(pool.tolist())
    assert np.array_equal(subset, scenes.train_subset(pool, 50, 0)) and np.array_equal(subset, np.sort(subset))
    assert np.array_equal(scenes.train_subset(pool, n=1000), pool)  # a small pool is used whole


# --- The two automatic checks, and the scene check -------------------------------------------------


def test_group_check_finds_a_planted_failure():
    unit, truth = cameras(n_cam=8, per=15)
    sims = unit @ unit.T
    limit = float(np.float32((sims[truth[:, None] == truth[None, :]].min() + sims[truth[:, None] != truth[None, :]].max()) / 2))
    good = scenes.group_check(unit, truth, limit, block=32)
    assert good["failures"] == 0 and good["highest_similarity_outside_a_group"] < limit
    planted = truth.copy()
    planted[0] = 99  # a frame of camera 0 in a group of its own: it and its camera have a near image outside
    bad = scenes.group_check(unit, planted, limit, block=32)
    assert bad["failures"] >= 2 and bad["highest_similarity_outside_a_group"] >= limit
    assert scenes.group_check(unit, np.zeros(120, dtype=int), limit) == {"failures": 0, "highest_similarity_outside_a_group": None}


def test_split_check_finds_a_planted_failure():
    unit, truth = cameras(n_cam=12, per=10)
    sims = unit @ unit.T
    limit = float(np.float32((sims[truth[:, None] == truth[None, :]].min() + sims[truth[:, None] != truth[None, :]].max()) / 2))
    splits = scenes.split_with_caps(truth, n_test=40, n_val=20, caps=(8, 8), seed=0)
    errors, numbers = scenes.split_check(unit, splits, truth, limit, min_groups=(5, 3), block=32)
    assert errors == [] and numbers["test groups"] == 5 and numbers["val groups"] == 3
    for pair in ("test to train_pool", "val to train_pool", "test to val"):
        assert numbers[pair]["at_or_above_the_limit"] == 0 and numbers[pair]["highest_similarity"] < limit

    def moved(source: str, target: str) -> dict:  # one image of `source` goes to `target`: its camera is now in both
        out = {name: v.copy() for name, v in splits.items()}
        out[source], out[target] = out[source][1:], np.sort(np.append(out[target], out[source][0]))
        return out

    errors, numbers = scenes.split_check(unit, moved("test", "train_pool"), truth, limit, (5, 3))
    assert errors == ["7 test images have a train_pool image at or above the limit"]
    assert numbers["test to train_pool"]["highest_similarity"] >= limit
    assert scenes.split_check(unit, moved("val", "train_pool"), truth, limit, (5, 3))[0] == [
        "7 val images have a train_pool image at or above the limit"]
    assert scenes.split_check(unit, moved("test", "val"), truth, limit, (5, 3))[0] == [
        "7 test images have a val image at or above the limit"]
    # an unused image (over the cap) is in no split: its test group has no error from it
    assert len(splits["unused"]) > 0
    assert scenes.split_check(unit, splits, truth, limit, min_groups=(6, 4))[0] == [
        "the test split has 5 groups, 6 are needed", "the val split has 3 groups, 4 are needed"]


def test_scene_check_pairs():
    unit, truth = cameras(n_cam=10, per=20)
    by_scene = scenes.split_with_caps(truth, n_test=40, n_val=20, caps=(None, None), seed=0)
    by_image = scenes.random_split(200, n_test=40, n_val=20, seed=0)
    assert len(by_image["test"]) == 40 and len(by_image["val"]) == 20 and len(by_image["train_pool"]) == 140
    same_scene = lambda split: np.mean([truth[a] == truth[b] for a, b, _ in scenes.scene_check_pairs(unit, split, n=30)])  # noqa: E731
    assert same_scene(by_scene) == 0.0  # no test image has its camera in the train pool
    assert same_scene(by_image) == 1.0  # a random image split puts every camera in both
    pairs = scenes.scene_check_pairs(unit, by_scene, n=30)
    assert len(pairs) == 30 and all(a in by_scene["test"] and b in by_scene["train_pool"] for a, b, _ in pairs)
    sims = unit @ unit.T
    for a, b, s in pairs:  # the nearest train-pool image, and its similarity
        assert s == pytest.approx(float(sims[a, by_scene["train_pool"]].max()), abs=1e-6) and sims[a, b] == pytest.approx(s, abs=1e-6)
    assert pairs == scenes.scene_check_pairs(scenes.as_tensor(unit), by_scene, n=30)  # the tensor or the array


def test_neighbour_similarity_shows_ordered_ids():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 8))
    truth = np.repeat(np.arange(10), 20)
    embeddings = scenes.unit_rows(centres[truth] + rng.normal(size=(200, 8)) * 0.05)  # prepared: unit length
    ordered = scenes.neighbour_similarity(embeddings)
    shuffled = scenes.neighbour_similarity(embeddings[rng.permutation(200)])
    assert ordered["adjacent_ids"] > 0.9 > ordered["random_pairs"]
    assert abs(shuffled["adjacent_ids"] - shuffled["random_pairs"]) < 0.2


def test_class_count_errors(data_root):
    coco = load_annotations(data_root / "annotations.json")
    ids = [im["id"] for im in coco["images"]]
    errors = scenes.class_count_errors(coco, {"test": ids, "val": ids})
    assert any(e.startswith("test: car has 2 boxes, 100 are needed") for e in errors) and len(errors) == 20
    assert scenes.MIN_TEST_BOXES == 100 and scenes.MIN_VAL_BOXES == 30


# --- Embeddings ------------------------------------------------------------------------------


def test_embeddings_are_mean_features_of_the_backbone(data_root):
    coco = load_annotations(data_root / "annotations.json")
    paths = [data_root / "train" / im["file_name"] for im in coco["images"]]
    backbone = make_tiny_rfdetr().model.backbone.backbone
    emb = scenes.embed_images(paths, backbone, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], batch_size=5,
                              num_workers=0, size=TINY_SIZE, log_every=0)
    assert emb.shape == (24, 24) and emb.dtype == np.float32 and np.isfinite(emb).all()
    again = scenes.embed_images(paths[:3], backbone, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], batch_size=2,
                                num_workers=0, size=TINY_SIZE, log_every=0)
    assert np.allclose(emb[:3], again, atol=1e-5)  # does not depend on the batch
    unit = scenes.unit_rows(emb)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-5)


def test_grid_embedding_keeps_the_position(data_root):
    """The second embedding (design §4.3): the mean in each cell of a grid, the cells joined."""
    coco = load_annotations(data_root / "annotations.json")
    paths = [data_root / "train" / im["file_name"] for im in coco["images"]][:6]
    backbone = make_tiny_rfdetr().model.backbone.backbone
    arguments = dict(batch_size=4, num_workers=0, size=TINY_SIZE, log_every=0)
    mean = scenes.embed_images(paths, backbone, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], **arguments)
    grid = scenes.embed_images(paths, backbone, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], grid=2, **arguments)
    assert grid.shape == (6, 24 * 4) and grid.dtype == np.float16 and np.isfinite(grid).all()
    side = backbone(np_images(paths, TINY_SIZE)).feature_maps[-1].shape[-1]
    if side % 2 == 0:  # cells of equal size: the mean of the cells is the mean of the map
        assert np.allclose(grid.astype(np.float32).reshape(6, 24, 4).mean(axis=2), mean, atol=2e-3)
    assert not np.allclose(grid.astype(np.float32).reshape(6, 24, 4)[:, :, 0], grid.astype(np.float32).reshape(6, 24, 4)[:, :, 3], atol=1e-4)


def np_images(paths, size):
    import torch
    from PIL import Image

    return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.float32)
                                         .transpose(2, 0, 1) / 255.0) for p in paths])


def test_prepare_embeddings():
    rng = np.random.default_rng(0)
    raw = (rng.normal(size=(50, 12)) + 3).astype(np.float16)
    before = raw.copy()
    plain = scenes.prepare_embeddings(raw)
    assert plain.dtype == np.float32 and np.allclose(np.linalg.norm(plain, axis=1), 1.0, atol=1e-5)
    assert np.allclose(plain, scenes.unit_rows(raw.astype(np.float32)), atol=1e-6) and np.array_equal(raw, before)
    centred = scenes.prepare_embeddings(raw, centre=True)
    expected = raw.astype(np.float32) - raw.astype(np.float32).mean(axis=0)
    assert np.allclose(centred, scenes.unit_rows(expected), atol=1e-5)
    assert (plain @ plain.T).mean() > 0.8 > abs((centred @ centred.T).mean())  # the common part is gone
    tensor = scenes.as_tensor(plain)
    assert scenes.as_tensor(tensor) is tensor and tensor.dtype.is_floating_point and tuple(tensor.shape) == (50, 12)


# --- Files -----------------------------------------------------------------------------------


def test_split_files(tmp_path):
    ids = [f"{k:08d}" for k in range(10)]
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    splits = {"test": np.array([0]), "val": np.array([2, 3]), "train_pool": np.arange(4, 10), "train": np.array([4, 5, 8]),
              "unused": np.array([1])}
    scenes.write_split_files(tmp_path, ids, groups, splits, {"seed": 0})
    assert read_ids(tmp_path / "train.txt") == ["00000004", "00000005", "00000008"]
    assert read_ids(tmp_path / "test.txt") == ["00000000"] and read_ids(tmp_path / "unused.txt") == ["00000001"]
    assert scenes.read_groups(tmp_path / "scene_groups.csv") == dict(zip(ids, groups.tolist()))
    assert json.loads((tmp_path / "split_meta.json").read_text()) == {"seed": 0}
    assert {p.name for p in tmp_path.iterdir()} == {*scenes.SPLIT_FILES, "split_meta.json"}
    # a run that gives no split removes the lists of the run before it, and nothing else
    (tmp_path / "notes.txt").write_text("mine")
    assert scenes.remove_split_files(tmp_path) == list(scenes.SPLIT_FILES)
    assert {p.name for p in tmp_path.iterdir()} == {"split_meta.json", "notes.txt"} and scenes.remove_split_files(tmp_path) == []
    scenes.write_meta(tmp_path, {"limit": None})
    assert json.loads((tmp_path / "split_meta.json").read_text()) == {"limit": None}


def test_design_sizes():
    assert (scenes.TEST_IMAGES, scenes.VAL_IMAGES, scenes.TRAIN_IMAGES, scenes.EMBED_SIZE) == (10_000, 2_000, 8_000, 224)
    assert (scenes.NEIGHBOURS, scenes.CANDIDATES, scenes.MAX_GROUP_SHARE, scenes.GRID) == (100, 17, 0.20, 4)
    assert (scenes.TEST_CAP, scenes.VAL_CAP, scenes.MIN_TEST_BOXES, scenes.MIN_VAL_BOXES) == (300, 100, 100, 30)
    # the caps give the minimum number of groups of the split check
    assert (math.ceil(scenes.TEST_IMAGES / scenes.TEST_CAP), math.ceil(scenes.VAL_IMAGES / scenes.VAL_CAP)) == (34, 20)
