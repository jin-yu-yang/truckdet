"""The results of the project on MIO-TCD: the files of `results/` (stage `report` of the notebook).

The test split was scored one time for each of the 12 final checkpoints, and the predictions are not in the
repository. So these tests do not calculate a result again. They check that the files agree with each other,
that the outcomes follow the rules of design §2, and that the report and the README give the outcomes of
`numbers.json`.
"""

import json
from pathlib import Path

import pytest

from truckdet import report, stats
from truckdet.data.miotcd import read_ids
from truckdet.runs import ARMS, GATE, SEEDS, read_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS, RUNS = ROOT / "results", ROOT / "results" / "runs"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def numbers() -> dict:
    return _json(RESULTS / "numbers.json")


@pytest.fixture(scope="module")
def final() -> dict[str, list[dict]]:
    """Arm -> the test evaluation of its final runs, in the order of the seeds."""
    selection = read_selection(RESULTS)
    out = {}
    for arm in ARMS:
        stem = selection[arm]["run"].rsplit("_s", 1)[0]
        out[arm] = [_json(RUNS / f"{stem}_s{seed}" / "eval_test.json") for seed in SEEDS]
    return out


def test_the_test_split_was_scored_for_the_final_runs_only(final):
    scored = {p.parent.name for p in RUNS.glob("*/eval_test.json")}
    assert scored == {e["run"] for runs in final.values() for e in runs} and len(scored) == len(ARMS) * len(SEEDS)
    images = len(read_ids(ROOT / "splits" / "test.txt"))
    for arm, runs in final.items():
        assert [e["seed"] for e in runs] == list(SEEDS)
        for e in runs:
            assert (e["arm"], e["split"], e["images"]) == (arm, "test", images), e["run"]


def test_each_final_run_passes_the_checks_of_the_design(final, numbers):
    """Design §9, stage 6: no skipped step, and each LoRA checkpoint passes the merge check."""
    for arm, runs in final.items():
        for e in runs:
            assert e["skipped_batches"] == 0, e["run"]
            assert e["merge_max_abs_diff"] < 1e-4, e["run"]
            assert (e["merge_max_abs_diff"] > 0) == (arm in ("D", "B")), e["run"]  # H and F have no adapters
            lines = (RUNS / e["run"] / "metrics.jsonl").read_text().splitlines()
            assert len([line for line in lines if line.strip()]) == 12, e["run"]  # 12 epochs
        saved = numbers["arms"][arm]
        assert saved["runs"] == [e["run"] for e in runs]
        assert saved["skipped_batches"] == [0, 0, 0]
        assert saved["merge_max_abs_diff"] == [e["merge_max_abs_diff"] for e in runs]
        assert saved["scaler_skips"] == [e["scaler_skips"] for e in runs]


def test_the_numbers_are_those_of_the_runs(final, numbers):
    mean = {}
    for arm, runs in final.items():
        ap = numbers["arms"][arm]["metrics"]["AP"]
        assert ap["by_seed"] == pytest.approx([e["AP"] for e in runs], abs=1e-12)
        mean[arm] = sum(e["AP"] for e in runs) / len(runs)
        assert ap["value"] == pytest.approx(mean[arm], abs=1e-12) and ap["low"] < ap["value"] < ap["high"]
    assert set(numbers["differences"]) == set(report.DIFFERENCES)
    for name, d in numbers["differences"].items():
        a, b = name.split(" - ")
        assert d["value"] == pytest.approx(mean[a] - mean[b], abs=1e-12) and d["low"] < d["value"] < d["high"]
    for arm in ("D", "B"):
        share = numbers["recovered_share"][arm]
        assert share["value"] == pytest.approx((mean[arm] - mean["H"]) / (mean["F"] - mean["H"]), abs=1e-12)
    for label in report.ERROR_SETS:
        for arm, runs in final.items():
            for key in ("label accuracy", "localized share"):
                by_seed = [e["errors"][label][key] for e in runs]
                assert numbers["errors"][label][arm][key]["value"] == pytest.approx(sum(by_seed) / len(by_seed), abs=1e-12)


def test_the_outcomes_follow_the_rules_of_the_design(numbers):
    """Design §2: the rules were written before any training run."""
    b_d = numbers["differences"]["B - D"]
    family = numbers["errors"]["truck family"]["B - D"]
    r_b = numbers["recovered_share"]["B"]
    assert numbers["readings"] == {
        "P1": stats.reading_p1((b_d["low"], b_d["high"])),
        "P2": stats.reading_p2((family["label accuracy"]["low"], family["label accuracy"]["high"]),
                               (family["localized share"]["low"], family["localized share"]["high"])),
        "P3": stats.reading_p3((r_b["low"], r_b["high"])),
    } == {"P1": "confirmed", "P2": "refuted", "P3": "confirmed"}
    assert b_d["low"] > 0  # P1: the interval of AP_B − AP_D is above zero
    assert family["label accuracy"]["high"] < 0  # P2: the interval of the label accuracy difference is below zero
    assert family["localized share"]["low"] < 0 < family["localized share"]["high"]
    assert r_b["low"] > stats.P3_THRESHOLD  # P3: the interval of R_B is above 2/3


def test_the_learning_rates_and_the_gate_are_those_of_the_runs():
    selection, gate = read_selection(RESULTS), _json(RESULTS / GATE)
    for arm in ARMS:
        chosen = selection[arm]
        val = _json(RUNS / chosen["run"] / "eval_val.json")
        assert val["AP"] == pytest.approx(max(chosen["val_ap"].values()), abs=1e-12), arm
    assert gate["runs"] == {"H": selection["H"]["run"], "F": selection["F"]["run"]} and gate["passed"]
    for arm in ("H", "F"):
        assert gate[f"AP_{arm}"] == pytest.approx(_json(RUNS / gate["runs"][arm] / "eval_val.json")["AP"], abs=1e-12)
    assert gate["difference"] >= gate["min_difference"] and gate["ratio"] >= gate["min_ratio"]


def _outcome_rows(numbers: dict) -> list[str]:
    """The three rows of "Predictions and outcomes", as `report.render` writes them."""
    readings = numbers["readings"]
    family = numbers["errors"]["truck family"]["B - D"]
    return [
        f"| P1 | {report.PREDICTIONS['P1']} | **{readings['P1']}** | AP_B − AP_D = "
        f"{report._ci(numbers['differences']['B - D'], signed=True)} |",
        f"| P2 | {report.PREDICTIONS['P2']} | **{readings['P2']}** | Truck family, B − D: label accuracy "
        f"{report._pct_ci(family['label accuracy'], signed=True)} points; localized share "
        f"{report._pct_ci(family['localized share'], signed=True)} points |",
        f"| P3 | {report.PREDICTIONS['P3']} | **{readings['P3']}** | Recovered share of B = "
        f"{report._ci(numbers['recovered_share']['B'])}; the line is {stats.P3_THRESHOLD:.3f} |",
    ]


def test_the_report_gives_the_outcomes_in_its_first_screen(numbers):
    text = (RESULTS / "report.md").read_text()
    first_screen = text.split("\n## Results\n")[0]
    assert f"**Question.** {report.QUESTION}" in first_screen
    for row in _outcome_rows(numbers):
        assert row in first_screen
    for arm in ARMS:  # the table of the arms: AP with its interval, and the three seeds
        ap = numbers["arms"][arm]["metrics"]["AP"]
        assert f"| {report._ci(ap)} | {' / '.join(report._f(v) for v in ap['by_seed'])} |" in text, arm
    assert (RESULTS / "recovered_share.png").exists()


def test_the_readme_gives_the_outcomes_of_the_report(numbers):
    text = (ROOT / "README.md").read_text()
    results = text.split("\n## Results\n")[1].split("\n## ")[0]
    for row in _outcome_rows(numbers):
        assert row in results
    for arm in ARMS:
        ap = numbers["arms"][arm]["metrics"]["AP"]
        assert f"| {report._ci(ap)} | {' / '.join(report._f(v) for v in ap['by_seed'])} |" in results, arm
    for arm in ("D", "B"):
        assert report._ci(numbers["recovered_share"][arm]) in results, arm
    assert "To come" not in text
