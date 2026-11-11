import json

import numpy as np
import pytest

from tests.conftest import tiny_config
from truckdet import report, runs, stats
from truckdet.config import for_run
from truckdet.data.miotcd import SPECIFIC_CLASSES, TRUCK_FAMILY
from truckdet.evaluation import metric_names
from truckdet.training import run_training

RATES = {"H": (1e-3, None), "D": (1e-3, None), "B": (1e-3, None), "F": (1e-3, 1e-4)}
SEEDS = (0, 1)


@pytest.fixture
def runs_root(data_root, splits_dir, tmp_path):
    """Two seeds of each arm, trained for one epoch on the tiny model and scored on the test split."""
    root = tmp_path / "runs"
    for arm, (lr, body_lr) in RATES.items():
        for seed in SEEDS:
            cfg = for_run(tiny_config(arm, epochs=1), lr, seed, body_lr)
            run_dir = run_training(cfg, data_root, splits_dir, root, device="cpu", log_every=0)
            runs.evaluate_run(run_dir, data_root, splits_dir, "test", device="cpu", num_workers=0)
    selection = {arm: {"lr": lr, "body_lr": body_lr, "val_ap": {"0.001": 0.1}} for arm, (lr, body_lr) in RATES.items()}
    (root / runs.SELECTION).write_text(json.dumps(selection))
    return root


def test_report_end_to_end_and_the_same_on_a_second_run(runs_root, data_root, splits_dir, tmp_path):
    out = tmp_path / "results"
    text = report.analyse(runs_root, data_root, splits_dir, out, n_boot=20, seeds=SEEDS)
    assert (out / "report.md").read_text() == text and (out / "recovered_share.png").stat().st_size > 5000
    # the first screen: the question, the predictions and their outcomes
    head = text[: text.index("## Results")]
    assert report.QUESTION in head and all(p in head for p in report.PREDICTIONS.values())
    # the scope of the answer is in the first screen too (design §4.3): scene groups are not cameras
    assert "*The scope of the answer.*" in head and "not an answer for new cameras" in head
    outcomes = [line for line in head.splitlines() if line.startswith(("| P1 |", "| P2 |", "| P3 |"))]
    assert len(outcomes) == 3 and all(any(f"**{w}**" in line for w in ("confirmed", "refuted", "not shown")) for line in outcomes)
    for section in ("## Results", "### Differences in AP", "## Error split: boxes and labels", "### Truck family",
                    "### Confusion of the truck family, B: LoRA in the backbone", "## Other metrics",
                    "## Learning rates and the gate", "## Checks", "## Limits"):
        assert section in text
    assert "6 images" in text and "mean of 2 seeds (0, 1)" in text
    assert "20 paired bootstrap resamples of the 3 scene groups of the split" in text
    assert "*The unit of the bootstrap.* AP_B − AP_D with resamples of scene groups:" in text
    assert "With resamples of single images:" in text
    # the composition of the splits, from the split files: each group of the fixture has 2 images
    assert "| Share of the images that come from | Dataset | Train subset | Test | Val |" in text
    assert "| groups with fewer than 100 images | 100.0% | 100.0% | 100.0% | 100.0% |" in text
    assert "| groups with 1,000 images or more | 0.0% | 0.0% | 0.0% | 0.0% |" in text
    assert "b_lr1e-3_s0, b_lr1e-3_s1" in text and "f_lr1e-4_h1e-3_s0" in text
    numbers = json.loads((out / "numbers.json").read_text())
    assert set(numbers["arms"]) == set("HDBF") and set(numbers["readings"]) == {"P1", "P2", "P3"}
    assert set(numbers["readings"].values()) <= {"confirmed", "refuted", "not shown"}
    assert len(numbers["arms"]["B"]["metrics"]["AP"]["by_seed"]) == 2
    by_image, by_group = numbers["image_bootstrap"]["B - D"], numbers["differences"]["B - D"]
    assert by_image["value"] == by_group["value"] and by_image["low"] <= by_image["value"] <= by_image["high"]
    # a second run reads the saved predictions (and the cache) and gives the same text
    assert all((runs_root / name / "analysis_test.npz").exists() for name in ("h_lr1e-3_s0", "f_lr1e-4_h1e-3_s1"))
    assert report.analyse(runs_root, data_root, splits_dir, out, n_boot=20, seeds=SEEDS) == text
    assert report.analyse(runs_root, data_root, splits_dir, tmp_path / "fresh", n_boot=20, seeds=SEEDS, cache=False) == text


def test_missing_predictions_are_listed(runs_root, data_root, splits_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="h_lr1e-3_s2/pred_test.npz"):
        report.analyse(runs_root, data_root, splits_dir, tmp_path / "results", n_boot=5, seeds=(0, 1, 2))


def fake_analysis(name: str, ap: float, noise: np.ndarray, share: float, accuracy: float) -> report.RunAnalysis:
    """An analysis whose metrics are all `ap` (+ noise on each resample) and whose error split is given."""
    m = len(metric_names())
    n_boot = len(noise)
    errors = {s: np.array([share, accuracy]) for s in report.ERROR_SETS}
    error_samples = {s: np.tile([share, accuracy], (n_boot, 1)) + noise[:, None] * 0.1 for s in report.ERROR_SETS}
    metrics = {"AP small": ap, "AP 11 classes, no ignore regions": ap, "AP motorized_vehicle": ap, "box AP": ap, "box AP50": ap}
    return report.RunAnalysis(name, metrics, np.full(m, ap), np.full((n_boot, m), ap) + noise[:, None], errors, error_samples,
                              {c: {c: 10} for c in TRUCK_FAMILY}, {"merge_max_abs_diff": 1e-12, "skipped_batches": 0})


def numbers_for(ap: dict, share: dict, accuracy: dict, spread: float = 0.002) -> dict:
    noise = np.random.default_rng(0).normal(0, spread, 200)  # the same images move every arm the same way: paired
    analyses = {arm: [fake_analysis(f"{arm}{s}", ap[arm] + 0.001 * s, noise, share[arm], accuracy[arm]) for s in range(3)]
                for arm in "HDBF"}
    return report.build_numbers(analyses)


def test_readings_follow_the_numbers():
    flat = {arm: 0.9 for arm in "HDBF"}
    numbers = numbers_for({"H": 0.40, "D": 0.44, "B": 0.56, "F": 0.60}, flat, {"H": 0.6, "D": 0.6, "B": 0.8, "F": 0.8})
    assert numbers["readings"] == {"P1": "confirmed", "P2": "confirmed", "P3": "confirmed"}
    assert numbers["recovered_share"]["B"]["value"] == pytest.approx(0.8) and numbers["recovered_share"]["D"]["value"] == pytest.approx(0.2)
    assert numbers["differences"]["B - D"]["value"] == pytest.approx(0.12)
    assert numbers["differences"]["B - D"]["seed_pairs_in_agreement"] == 9 and numbers["differences"]["B - D"]["seed_pairs"] == 9
    assert numbers["errors"]["truck family"]["B - D"]["label accuracy"]["value"] == pytest.approx(0.2)
    assert numbers["arms"]["B"]["metrics"]["AP"]["by_seed"] == pytest.approx([0.56, 0.561, 0.562])

    reverse = numbers_for({"H": 0.40, "D": 0.56, "B": 0.44, "F": 0.60}, flat, {"H": 0.6, "D": 0.8, "B": 0.6, "F": 0.8})
    assert reverse["readings"] == {"P1": "refuted", "P2": "refuted", "P3": "refuted"}

    equal = numbers_for({"H": 0.40, "D": 0.50, "B": 0.50, "F": 0.60}, flat, flat, spread=0.05)
    assert equal["readings"]["P1"] == "not shown" and equal["readings"]["P2"] == "not shown"

    boxes_too = numbers_for({"H": 0.40, "D": 0.44, "B": 0.56, "F": 0.60}, {"H": 0.7, "D": 0.7, "B": 0.9, "F": 0.9},
                            {"H": 0.6, "D": 0.6, "B": 0.8, "F": 0.8})
    assert boxes_too["readings"]["P2"] == "not shown"  # the backbone is better in the boxes too


def test_render_handles_missing_classes_and_shows_the_outcomes():
    numbers = numbers_for({"H": 0.40, "D": 0.44, "B": 0.56, "F": 0.60}, {arm: 0.9 for arm in "HDBF"},
                          {"H": 0.6, "D": 0.6, "B": 0.8, "F": 0.8})
    numbers["arms"]["H"]["metrics"]["bicycle"] = {"value": float("nan"), "low": float("nan"), "high": float("nan"), "by_seed": []}
    context = {"images": 10_000, "seeds": [0, 1, 2], "n_boot": 400, "split": "test",
               "trainable": {"H": 1_896_146, "D": 2_048_210, "B": 2_048_210, "F": 31_886_690}, "selection": {},
               "gate": None, "figure": "recovered_share.png", "samples": []}
    text = report.render(numbers, context)
    assert "| P1 | LoRA in the backbone is better than LoRA in the decoder. | **confirmed** | AP_B − AP_D = +0.120 [" in text
    assert "| B: LoRA in the backbone | 2,048,210 | 0.561 [" in text and "1 (by definition)" in text
    assert "| bicycle | – |" in text and "10,000 images" in text and "![Recovered share" in text
    assert "resamples of the scene groups of the split" in text and "The unit of the bootstrap" not in text
    # no split files: the text has no count and no table, and it does not claim the guarantee of the split check
    assert "The split keeps the images of one scene group in one split." in text and "at most 300 images" in text
    assert "no near image in the training data" not in text and "Share of the images that come from" not in text
    assert "this rule set the chosen limit" in text and "at the cap of" not in text
    split = {
        "groups": 41,
        "split_meta": {"limit": 0.97531, "caps": {"test": 250, "val": 80}, "checks": {"passed": True}},
        "label_measurement": {"images": 100, "boxes": 297, "limit": 0.05, "clear_objects_with_no_box": 6,
                              "all_objects_with_no_box_best_guess": 35},
        "scene_measurement": {"scene_pairs": {"looked_at": 100, "same_camera_view": 60},
                              "largest_group": {"images_looked_at": 50, "cameras": 5},
                              "random_image_split": {"looked_at": 50, "same_camera_view": 50}},
        "composition": {"images": {"dataset": 110_000, "train": 8_000, "test": 10_000, "val": 2_000},
                        "small_groups": {"dataset": 0.064, "train": 0.051, "test": 0.232, "val": 0.653},
                        "large_groups": {"dataset": 0.742, "train": 0.784, "test": 0.330, "val": 0.198},
                        "at_cap": {"test": {"cap": 250, "groups": 22, "images": 5_500}}},
    }
    with_split = report.render({**numbers, "image_bootstrap": {"B - D": {"value": 0.12, "low": 0.11, "high": 0.13}}},
                               {**context, **split})
    assert "resamples of the 41 scene groups of the split" in with_split and "With resamples of single images: +0.120 [+0.110, +0.130]" in with_split
    head = with_split[: with_split.index("## Results")]
    assert ("*The scope of the answer.* The test images have no near image in the training data: no training image has "
            "a similarity of 0.9753 or more to a test image. The cameras are not separated. For 60 of 100 test images") in head
    assert "It is not an answer for new cameras." in head
    limits = with_split[with_split.index("## Limits"):]
    assert "- **The cameras are not separated.**" in limits and "a 95% interval of 50% to 69% (100 pairs)" in limits
    assert "It is a lower limit" in limits and "In a random image split the count was 50 of 50." in limits
    assert "So a confirmed P1 is the weaker result, and a refuted P1 is the stronger result." in limits
    assert "at most 250 images to the test split and at most 80 to the val split" in limits
    assert "| groups with fewer than 100 images | 6.4% | 5.1% | 23.2% | 65.3% |" in limits
    assert "| groups with 1,000 images or more | 74.2% | 78.4% | 33.0% | 19.8% |" in limits
    assert "- **The val split is different from the test split.** The val cap of 80 is the cause." in limits
    assert ("- **A few groups give most of the test split.** 22 groups are at the cap of 250 and give 5,500 of the "
            "10,000 images. One group can hold more than one camera: 50 images of the largest group show 5 cameras.") in limits
    assert "this rule set 0.9753" in limits
    # the label check of design §4.5: the clear objects decide it, and the second count is reported
    assert ("- **Some objects have no box.** On 100 random images with 297 boxes, a person found 6 clear objects with "
            "no box: 2.0% of the clear objects (the limit of the label check is 5%).") in limits
    assert "about 35 objects have no box: about 10.5%." in limits and "Some objects have no box" not in text
    # the limits of change 1 that the measurement replaces are gone
    assert "This is safe for the split" not in with_split and "Two parts of one camera" not in with_split
    assert all(c in text for c in SPECIFIC_CLASSES) and stats.P3_THRESHOLD == pytest.approx(0.6667, abs=1e-4)


def test_split_composition_and_the_interval_of_a_count(tmp_path):
    """Design §11: the share of the images of each split that come from small and from large scene groups."""
    from truckdet.data.miotcd import write_ids

    # group 0: 1,200 images; group 1: 150; groups 2 to 51: 1 image each
    group_of = [0] * 1200 + [1] * 150 + list(range(2, 52))
    ids = [f"{k:08d}" for k in range(len(group_of))]
    (tmp_path / "scene_groups.csv").write_text("image_id,scene_group\n" + "".join(f"{i},{g}\n" for i, g in zip(ids, group_of)))
    write_ids(tmp_path / "train.txt", ids[:1000])  # all from the large group
    write_ids(tmp_path / "test.txt", ids[1000:1010] + ids[1200:1210] + ids[1350:1370])  # 10 large, 10 medium, 20 small
    write_ids(tmp_path / "val.txt", ids[1370:1380])  # all from groups of one image
    composition = report.split_composition(tmp_path, {"test": 10, "val": 100})
    assert composition["images"] == {"dataset": 1400, "train": 1000, "test": 40, "val": 10}
    assert composition["small_groups"] == pytest.approx({"dataset": 50 / 1400, "train": 0.0, "test": 0.5, "val": 1.0})
    assert composition["large_groups"] == pytest.approx({"dataset": 1200 / 1400, "train": 1.0, "test": 0.25, "val": 0.0})
    # two groups give exactly the cap of 10 images to the test split; no group gives 100 to the val split
    assert composition["at_cap"] == {"test": {"cap": 10, "groups": 2, "images": 20}, "val": {"cap": 100, "groups": 0, "images": 0}}
    assert report.split_composition(tmp_path)["at_cap"] == {}
    assert report.split_composition(tmp_path / "missing") is None
    low, high = report.wilson_interval(60, 100)
    assert (low, high) == pytest.approx((0.5020, 0.6906), abs=1e-4)
    assert report.wilson_interval(0, 50)[0] == pytest.approx(0.0, abs=1e-12)
