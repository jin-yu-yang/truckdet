"""The accepted split of the project (design §4.3, "The accepted split"): the files of `splits/`.

The split is the output of the fourth run on MIO-TCD, with no change. No split is made again, so these
tests keep the files as they are.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from truckdet import report
from truckdet.data.miotcd import read_ids
from truckdet.data.scenes import SPLIT_FILES, read_groups

SPLITS = Path(__file__).resolve().parents[1] / "splits"
# The first 16 characters of the SHA-256 of each file, from the design change that accepted the split.
SHA256 = {
    "test.txt": "0e6c23878dd86420",
    "val.txt": "3b04e94308d1e7b6",
    "train.txt": "35570634ff86b82a",
    "train_pool.txt": "7eed9ef168e457f8",
    "unused.txt": "1207b12341cf8e7a",
    "scene_groups.csv": "fa9415a232f91b3b",
    "split_meta.json": "f33f827175f90a8f",
}


@pytest.fixture(scope="module")
def split():
    ids = {name: read_ids(SPLITS / f"{name}.txt") for name in ("test", "val", "train_pool", "train", "unused")}
    return ids, read_groups(SPLITS / "scene_groups.csv"), json.loads((SPLITS / "split_meta.json").read_text())


def test_the_files_are_those_of_the_fourth_run():
    assert set(SHA256) == set(SPLIT_FILES) | {"split_meta.json"}
    for name, start in SHA256.items():
        assert hashlib.sha256((SPLITS / name).read_bytes()).hexdigest()[:16] == start, name


def test_the_split_is_consistent(split):
    ids, group, meta = split
    assert {name: len(v) for name, v in ids.items()} == meta["images"] == {
        "test": 10_001, "val": 2_017, "train_pool": 67_785, "train": 8_000, "unused": 30_197}
    assert meta["embedding"] == "grid" and meta["limit"] == pytest.approx(0.7214082, abs=1e-7)
    assert meta["checks"]["passed"] and meta["checks"]["group_check"]["failures"] == 0
    assert meta["checks"]["split_check"]["test to train_pool"]["highest_similarity"] < meta["limit"]
    parts = {name: set(ids[name]) for name in ("test", "val", "train_pool", "unused")}
    assert sum(len(v) for v in parts.values()) == len(group) == 110_000 and set().union(*parts.values()) == set(group)
    assert set(ids["train"]) <= parts["train_pool"]
    groups = {name: {group[i] for i in v} for name, v in parts.items()}
    assert not groups["test"] & groups["val"] and not (groups["test"] | groups["val"]) & groups["train_pool"]
    assert groups["unused"] <= groups["test"] | groups["val"]
    assert {name: len(groups[name]) for name in ("test", "val", "train_pool")} == meta["groups_per_split"] == {
        "test": 1_096, "val": 513, "train_pool": 1_452}
    assert len({group[i] for i in ids["train"]}) == 291
    # no group gives more than its cap
    assert max(Counter(group[i] for i in ids["test"]).values()) == meta["caps"]["test"] == 300
    assert max(Counter(group[i] for i in ids["val"]).values()) == meta["caps"]["val"] == 100


def test_the_composition_is_that_of_the_design(split):
    """The table of design §11, and the groups at the cap."""
    _, _, meta = split
    composition = report.split_composition(SPLITS, meta["caps"])
    shares = {key: [round(100 * composition[key][part], 1) for part in ("dataset", "train", "test", "val")]
              for key in ("small_groups", "large_groups")}
    assert shares == {"small_groups": [6.4, 5.1, 23.2, 65.3], "large_groups": [74.2, 78.4, 33.0, 19.8]}
    assert composition["at_cap"]["test"] == {"cap": 300, "groups": 22, "images": 6_600}


def test_the_scene_measurement_is_the_count_of_the_person():
    """A measurement for the report, not a pass rule (design §4.5)."""
    measurement = json.loads((SPLITS / report.SCENE_MEASUREMENT).read_text())
    pairs = measurement["scene_pairs"]
    assert (pairs["looked_at"], pairs["same_camera_view"], pairs["different_camera_views"]) == (100, 60, 40)
    assert sum(sheet["same_camera_view"] for sheet in pairs["sheets"].values()) == 60
    assert sum(sum(sheet.values()) for sheet in pairs["sheets"].values()) == 100
    largest = measurement["largest_group"]
    assert (largest["images_looked_at"], largest["cameras"]) == (50, 5) and sum(largest["images_of_each_camera"]) == 50
    assert measurement["random_image_split"] == {**measurement["random_image_split"], "looked_at": 50, "same_camera_view": 50}
    meta = json.loads((SPLITS / "split_meta.json").read_text())
    scope, limit = report._scope(meta, measurement)
    assert scope.startswith("The test images have no near image in the training data: no training image has a similarity "
                            "of 0.7214 or more to a test image. The cameras are not separated. For 60 of 100 test images")
    assert "a 95% interval of 50% to 69% (100 pairs)" in limit and "It is not an answer for new cameras." in limit
