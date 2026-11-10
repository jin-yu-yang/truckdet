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
    assert "linked at a cosine similarity of the chosen limit or more" in text and "at most 300 images" in text
    with_split = report.render({**numbers, "image_bootstrap": {"B - D": {"value": 0.12, "low": 0.11, "high": 0.13}}},
                               {**context, "groups": 41, "split_meta": {"limit": 0.97531, "caps": {"test": 250, "val": 80}}})
    assert "resamples of the 41 scene groups of the split" in with_split and "With resamples of single images: +0.120 [+0.110, +0.130]" in with_split
    assert "linked at a cosine similarity of 0.9753 or more" in with_split and "at most 250 images" in with_split
    assert all(c in text for c in SPECIFIC_CLASSES) and stats.P3_THRESHOLD == pytest.approx(0.6667, abs=1e-4)
