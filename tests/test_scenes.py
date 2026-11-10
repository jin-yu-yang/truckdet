import json

import numpy as np

from tests.tiny import TINY_SIZE, make_tiny_rfdetr
from truckdet.data import scenes
from truckdet.data.miotcd import SPECIFIC_CLASSES, boxes_per_class, load_annotations, read_ids


def test_split_by_group_keeps_groups_whole():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=1000)
    splits = scenes.split_by_group(groups, n_test=200, n_val=60, seed=0)
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(1000))  # every image in one split
    assert not scenes.overlap_errors(splits, groups)
    sizes = np.bincount(groups)
    for name, target in (("test", 200), ("val", 60)):
        n = len(splits[name])
        last = max(sizes[g] for g in np.unique(groups[splits[name]]))
        assert target <= n < target + last  # whole groups until the target is reached, and no more
    again = scenes.split_by_group(groups, 200, 60, seed=0)
    assert all(np.array_equal(splits[name], again[name]) for name in splits)
    other = scenes.split_by_group(groups, 200, 60, seed=1)
    assert not np.array_equal(other["test"], splits["test"])


def test_split_grows_until_each_class_has_its_minimum():
    """Design §4.5: "each specific class has 100 boxes or more in the test split and 30 or more in the val split.
    If not, make the splits larger." On MIO-TCD the val split of 2,016 images had 5 `bicycle` boxes."""
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=1000)
    class_boxes = np.zeros((1000, 2), dtype=np.int64)
    class_boxes[:, 0] = 1  # a common class: one box in every image
    class_boxes[np.isin(groups, [3, 8, 17, 21, 29, 33]), 1] = 1  # a rare class: only six scenes have it
    plain = scenes.split_by_group(groups, 200, 60, seed=0)
    assert class_boxes[plain["val"], 1].sum() < 20  # the case: too few boxes of the rare class at the plain size
    larger = scenes.split_by_group(groups, 200, 60, seed=0, class_boxes=class_boxes, min_boxes=(20, 20))

    assert sorted(np.concatenate(list(larger.values())).tolist()) == list(range(1000))
    assert not scenes.overlap_errors(larger, groups)
    for name in ("test", "val"):
        assert len(larger[name]) >= {"test": 200, "val": 60}[name]
        assert (class_boxes[larger[name]].sum(axis=0) >= 20).all()
    # The groups keep their order: a split is a run of the random order, the plain test split is the start of the
    # larger one, and a split with its last group taken away is not complete.
    order = np.random.default_rng(0).permutation(np.unique(groups)).tolist()
    in_test, in_val = set(np.unique(groups[larger["test"]]).tolist()), set(np.unique(groups[larger["val"]]).tolist())
    assert in_test == set(order[: len(in_test)]) and in_val == set(order[len(in_test) : len(in_test) + len(in_val)])
    assert set(plain["test"].tolist()) <= set(larger["test"].tolist())
    for run, n_images in ((order[: len(in_test) - 1], 200), (order[len(in_test) : len(in_test) + len(in_val) - 1], 60)):
        positions = np.flatnonzero(np.isin(groups, run))
        assert len(positions) < n_images or (class_boxes[positions].sum(axis=0) < 20).any()
    # Minimums that the plain split has change nothing.
    same = scenes.split_by_group(groups, 200, 60, seed=0, class_boxes=class_boxes, min_boxes=(0, 0))
    assert all(np.array_equal(plain[name], same[name]) for name in plain)


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
    good = {"test": np.array([0, 1]), "val": np.array([2, 3]), "train_pool": np.array([4, 5]), "train": np.array([4])}
    assert scenes.overlap_errors(good, groups) == []
    assert scenes.overlap_errors({**good, "train": np.array([0])}, groups) == ["the train subset has images that are not in the train pool"]


def test_train_subset():
    pool = np.arange(100, 400)
    subset = scenes.train_subset(pool, n=50, seed=0)
    assert len(subset) == 50 == len(set(subset.tolist())) and set(subset.tolist()) <= set(pool.tolist())
    assert np.array_equal(subset, scenes.train_subset(pool, 50, 0)) and np.array_equal(subset, np.sort(subset))
    assert np.array_equal(scenes.train_subset(pool, n=1000), pool)  # a small pool is used whole


def test_clusters_recover_separate_scenes():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(6, 16)) * 5
    truth = np.repeat(np.arange(6), 30)
    embeddings = centres[truth] + rng.normal(size=(180, 16)) * 0.1
    groups = scenes.cluster_scenes(embeddings, k=6, seed=0)
    assert all(len(set(groups[truth == t].tolist())) == 1 for t in range(6)) and len(set(groups.tolist())) == 6
    assert len(set(scenes.cluster_scenes(embeddings[:4], k=1000).tolist())) == 4  # k is at most the number of images


def test_nearest_neighbours_and_scene_check():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 8)) * 5
    truth = np.repeat(np.arange(10), 20)
    embeddings = (centres[truth] + rng.normal(size=(200, 8)) * 0.05).astype(np.float32)
    index, similarity = scenes.nearest_neighbours(embeddings[:5] * 3.0, embeddings)  # the length does not matter
    assert index.tolist() == [0, 1, 2, 3, 4] and np.allclose(similarity, 1.0, atol=1e-5)
    by_scene = scenes.split_by_group(truth, n_test=40, n_val=20, seed=0)
    by_image = scenes.random_split(200, n_test=40, n_val=20, seed=0)
    assert len(by_image["test"]) == 40 and len(by_image["val"]) == 20 and len(by_image["train_pool"]) == 140
    same_scene = lambda split: np.mean([truth[a] == truth[b] for a, b, _ in scenes.scene_check_pairs(embeddings, split, n=30)])  # noqa: E731
    assert same_scene(by_scene) == 0.0  # no test image has its camera in the train pool
    assert same_scene(by_image) == 1.0  # a random image split puts every camera in both
    pairs = scenes.scene_check_pairs(embeddings, by_scene, n=30)
    assert len(pairs) == 30 and all(a in by_scene["test"] and b in by_scene["train_pool"] for a, b, _ in pairs)


def test_neighbour_similarity_shows_ordered_ids():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 8))
    truth = np.repeat(np.arange(10), 20)
    embeddings = centres[truth] + rng.normal(size=(200, 8)) * 0.05
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


def test_split_files(tmp_path):
    ids = [f"{k:08d}" for k in range(10)]
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    splits = {"test": np.array([0, 1]), "val": np.array([2, 3]), "train_pool": np.arange(4, 10), "train": np.array([4, 5, 8])}
    scenes.write_split_files(tmp_path, ids, groups, splits, {"seed": 0})
    assert read_ids(tmp_path / "train.txt") == ["00000004", "00000005", "00000008"]
    assert read_ids(tmp_path / "test.txt") == ["00000000", "00000001"]
    assert scenes.read_groups(tmp_path / "scene_groups.csv") == dict(zip(ids, groups.tolist()))
    assert json.loads((tmp_path / "split_meta.json").read_text()) == {"seed": 0}


def test_design_sizes():
    assert (scenes.N_GROUPS, scenes.TEST_IMAGES, scenes.VAL_IMAGES, scenes.TRAIN_IMAGES, scenes.EMBED_SIZE) == (1000, 10_000, 2_000, 8_000, 224)
