import json

import numpy as np
import pytest

from tests.conftest import tiny_config
from truckdet import runs
from truckdet.config import for_run
from truckdet.evaluation import load_predictions
from truckdet.training import run_training


def train(arm, lr, seed, data_root, splits_dir, out, body_lr=None):
    cfg = for_run(tiny_config(arm, epochs=1), lr, seed, body_lr)
    return run_training(cfg, data_root, splits_dir, out, device="cpu", log_every=0)


def fake_eval(runs_root, name: str, ap: float) -> None:
    (runs_root / name).mkdir(parents=True, exist_ok=True)
    (runs_root / name / "eval_val.json").write_text(json.dumps({"AP": ap}))


def test_evaluate_run_saves_predictions_and_scores(data_root, splits_dir, tmp_path):
    run_dir = train("B", 1e-3, 0, data_root, splits_dir, tmp_path)
    result = runs.evaluate_run(run_dir, data_root, splits_dir, "val", device="cpu", num_workers=0)
    predictions, meta = load_predictions(run_dir / "pred_val.npz")
    assert predictions["image_ids"].tolist() == list(range(12, 18)) and predictions["scores"].shape == (6, 12, 11)
    assert meta == {"run": "b_lr1e-3_s0", "arm": "B", "seed": 0, "split": "val",
                    "merge_max_abs_diff": meta["merge_max_abs_diff"], "skipped_batches": 0}
    assert 0 < meta["merge_max_abs_diff"] < 1e-9  # a trained adapter, merged exactly
    assert json.loads((run_dir / "eval_val.json").read_text()) == json.loads(json.dumps(result))
    assert result["images"] == 6 and 0.0 <= result["AP"] <= 1.0
    assert set(result["errors"]) == {"specific classes", "truck family"}
    # a second call scores the saved predictions and does not predict again
    stamp = (run_dir / "pred_val.npz").stat().st_mtime_ns
    assert runs.evaluate_run(run_dir, data_root, splits_dir, "val", device="cpu")["AP"] == result["AP"]
    assert (run_dir / "pred_val.npz").stat().st_mtime_ns == stamp


def test_test_split_is_scored_one_time(data_root, splits_dir, tmp_path):
    run_dir = train("H", 1e-3, 0, data_root, splits_dir, tmp_path)
    first = runs.evaluate_run(run_dir, data_root, splits_dir, "test", device="cpu", num_workers=0)
    assert first["images"] == 6 and first["merge_max_abs_diff"] == 0.0
    with pytest.raises(RuntimeError, match="one time"):
        runs.evaluate_run(run_dir, data_root, splits_dir, "test", device="cpu")
    with pytest.raises(ValueError):
        runs.evaluate_run(run_dir, data_root, splits_dir, "train", device="cpu")


def test_unfinished_run_cannot_be_evaluated(data_root, splits_dir, tmp_path):
    (tmp_path / "h_lr1e-3_s0").mkdir()
    with pytest.raises(FileNotFoundError, match="not finished"):
        runs.evaluate_run(tmp_path / "h_lr1e-3_s0", data_root, splits_dir, "val", device="cpu")


def test_start_model_check(data_root, splits_dir, tmp_path):
    result = runs.evaluate_start_model("tiny", data_root, splits_dir, tmp_path / "start", device="cpu", max_images=4, num_workers=0)
    assert result["images"] == 4 and set(result["AP50 per class"]) >= {"car", "bus"}
    assert (tmp_path / "start" / "pred_val.npz").exists()


def test_list_runs_reads_the_names(tmp_path):
    for name in ("b_lr3e-4_s0", "f_lr3e-5_h3e-4_s1", "start_model", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "h_lr1e-3_s0").write_text("a file, not a folder")
    found = runs.list_runs(tmp_path)
    assert [(r["name"], r["arm"], r["lr"], r["body_lr"], r["seed"]) for r in found] == [
        ("b_lr3e-4_s0", "B", 3e-4, None, 0), ("f_lr3e-5_h3e-4_s1", "F", 3e-4, 3e-5, 1)]
    assert runs.list_runs(tmp_path / "nothing") == []


def test_select_lr_asks_for_runs_then_chooses(tmp_path):
    cfg = tiny_config("H")
    assert runs.select_lr(tmp_path, cfg) == {"status": "run", "lr": [1e-4, 3e-4, 1e-3], "val_ap": {}, "head_lr": None}
    fake_eval(tmp_path, "h_lr1e-4_s0", 0.40)
    fake_eval(tmp_path, "h_lr3e-4_s0", 0.45)
    fake_eval(tmp_path, "h_lr3e-4_s1", 0.99)  # another seed: not a learning-rate run
    assert runs.select_lr(tmp_path, cfg)["lr"] == [1e-3]
    fake_eval(tmp_path, "h_lr1e-3_s0", 0.42)
    chosen = runs.select_lr(tmp_path, cfg)
    assert chosen["status"] == "chosen" and chosen["lr"] == 3e-4 and chosen["body_lr"] is None and chosen["run"] == "h_lr3e-4_s0"
    assert runs.read_selection(tmp_path)["H"]["val_ap"] == {"0.0001": 0.40, "0.0003": 0.45, "0.001": 0.42}


def test_select_lr_adds_one_step_at_an_end_one_time(tmp_path):
    cfg = tiny_config("D")
    for lr, ap in (("1e-4", 0.40), ("3e-4", 0.45), ("1e-3", 0.50)):
        fake_eval(tmp_path, f"d_lr{lr}_s0", ap)
    status = runs.select_lr(tmp_path, cfg)
    assert status["status"] == "run" and status["lr"] == [3e-3] and "one time" in status["reason"]
    assert runs.read_selection(tmp_path) == {}
    fake_eval(tmp_path, "d_lr3e-3_s0", 0.55)  # still the best, and still at the end: no second step
    chosen = runs.select_lr(tmp_path, cfg)
    assert chosen["status"] == "chosen" and chosen["lr"] == 3e-3


def test_arm_f_uses_the_rate_of_h_for_its_heads(tmp_path):
    cfg_f = tiny_config("F", lr_grid=[1e-5, 3e-5, 1e-4])
    assert runs.select_lr(tmp_path, cfg_f)["status"] == "blocked"
    for lr, ap in (("1e-4", 0.40), ("3e-4", 0.45), ("1e-3", 0.42)):
        fake_eval(tmp_path, f"h_lr{lr}_s0", ap)
    runs.select_lr(tmp_path, tiny_config("H"))
    status = runs.select_lr(tmp_path, cfg_f)
    assert status["status"] == "run" and status["lr"] == [1e-5, 3e-5, 1e-4] and status["head_lr"] == 3e-4
    fake_eval(tmp_path, "f_lr1e-5_h3e-4_s0", 0.50)
    fake_eval(tmp_path, "f_lr3e-5_h3e-4_s0", 0.58)
    fake_eval(tmp_path, "f_lr1e-4_h3e-4_s0", 0.52)
    fake_eval(tmp_path, "f_lr3e-4_h1e-3_s0", 0.99)  # heads at another rate: not a run of this choice
    chosen = runs.select_lr(tmp_path, cfg_f)
    assert (chosen["status"], chosen["lr"], chosen["body_lr"], chosen["run"]) == ("chosen", 3e-4, 3e-5, "f_lr3e-5_h3e-4_s0")
    final = runs.final_runs(tmp_path, arms=("H", "F"))
    assert [d.name for d in final["H"]] == ["h_lr3e-4_s0", "h_lr3e-4_s1", "h_lr3e-4_s2"]
    assert [d.name for d in final["F"]] == [f"f_lr3e-5_h3e-4_s{s}" for s in range(3)]
    with pytest.raises(KeyError, match="arm D"):  # the first arm without a choice
        runs.final_runs(tmp_path)


def test_gate_reads_the_chosen_runs(data_root, splits_dir, tmp_path):
    for arm, lr, body in (("H", 1e-3, None), ("F", 1e-3, 1e-4)):
        run_dir = train(arm, lr, 0, data_root, splits_dir, tmp_path, body)
        runs.evaluate_run(run_dir, data_root, splits_dir, "val", device="cpu", num_workers=0)
    (tmp_path / runs.SELECTION).write_text(json.dumps({"H": {"lr": 1e-3, "body_lr": None}, "F": {"lr": 1e-3, "body_lr": 1e-4}}))
    result = runs.run_gate(tmp_path, data_root, splits_dir, n_boot=30)
    assert result["runs"] == {"H": "h_lr1e-3_s0", "F": "f_lr1e-4_h1e-3_s0"} and result["val_images"] == 6
    assert result["difference"] == pytest.approx(result["AP_F"] - result["AP_H"]) and isinstance(result["passed"], bool)
    assert json.loads((tmp_path / runs.GATE).read_text())["n_boot"] == 30
    text = runs.gate_text(result)
    assert "AP_F − AP_H" in text and ("The gate passes" in text or "The gate fails: stop. The data cannot separate the arms." in text)
    assert "fails" in runs.gate_text({**result, "passed": False}) and "passes" in runs.gate_text({**result, "passed": True})


def test_gate_needs_the_val_predictions(data_root, splits_dir, tmp_path):
    (tmp_path / runs.SELECTION).write_text(json.dumps({"H": {"lr": 1e-3, "body_lr": None}, "F": {"lr": 1e-3, "body_lr": 1e-4}}))
    with pytest.raises(FileNotFoundError, match="evaluate the run"):
        runs.run_gate(tmp_path, data_root, splits_dir, n_boot=5)
    assert np.isfinite(runs.SEEDS).all() and runs.SEEDS == (0, 1, 2)
