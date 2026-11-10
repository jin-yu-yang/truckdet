import importlib.util
import json
from pathlib import Path

import pytest

from tests.tiny import write_synthetic_miotcd
from truckdet.data.miotcd import read_ids

ROOT = Path(__file__).parent.parent


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_config(tmp_path, arm: str) -> Path:
    """The arm's real config with a recipe that the tiny model runs in a second."""
    text = (ROOT / "configs" / f"arm_{arm.lower()}.yaml").read_text()
    text = text.replace("r: 33", "r: 3").replace("alpha: 66", "alpha: 6").replace("r: 9", "r: 4").replace("alpha: 18", "alpha: 8")
    text += "  epochs: 1\n  batch_size: 2\n  accum_steps: 2\n  num_workers: 0\n  amp: false\n  eval_batch_size: 4\n"
    path = tmp_path / f"arm_{arm.lower()}.yaml"
    path.write_text(text)
    return path


def test_prepare_data_and_make_split(tmp_path, capsys):
    root = write_synthetic_miotcd(tmp_path / "data", n_images=60, scenes=6)
    code = tmp_path / "localization_evaluation.py"
    code.write_text("motorized_vehicle_classes = ['articulated_truck', 'bus', 'car', 'pickup_truck', 'single_unit_truck', 'work_van']\n")
    with pytest.raises(SystemExit) as done:  # the counts are not the reference counts of the real dataset
        script("prepare_data").main(["--data-root", str(root), "--eval-code", str(code), "--sheets-dir", str(tmp_path / "checks"),
                                     "--label-images", "8"])
    out = capsys.readouterr().out
    assert done.value.code == 1 and "DIFFERENT from the reference counts" in out and "lists the same 6 classes" in out
    assert json.loads((root / "counts.json").read_text())["images"] == 60 and (tmp_path / "checks" / "labels_01.jpg").exists()

    with pytest.raises(SystemExit) as done:  # the class counts of a 60-image dataset are too small
        script("make_split").main(["--data-root", str(root), "--splits-dir", str(tmp_path / "splits"), "--device", "cpu",
                                   "--groups", "6", "--test-images", "12", "--val-images", "6", "--train-images", "20",
                                   "--sheets-dir", str(tmp_path / "checks"), "--pairs", "4", "--num-workers", "0"])
    out = capsys.readouterr().out
    assert done.value.code == 1 and "make the split larger" in out and "scene check, random_split" in out
    splits = {name: read_ids(tmp_path / "splits" / f"{name}.txt") for name in ("test", "val", "train_pool", "train")}
    assert len(splits["test"]) >= 12 and len(splits["val"]) >= 6 and len(splits["train"]) == 20
    assert set(splits["train"]) <= set(splits["train_pool"])
    assert sorted(splits["test"] + splits["val"] + splits["train_pool"]) == [f"{k:08d}" for k in range(60)]
    meta = json.loads((tmp_path / "splits" / "split_meta.json").read_text())
    assert meta["groups"] == 6 and meta["images"]["train"] == 20
    assert (tmp_path / "checks" / "scene_split_01.jpg").exists() and (root / "embeddings.npy").exists()


def test_train_select_gate_evaluate_analyse(data_root, splits_dir, tmp_path, capsys):
    """The order of the notebook, on the tiny model: learning-rate runs, the choice, the gate, the other seeds,
    the test split, the report."""
    runs_root = tmp_path / "runs"
    common = ["--data-root", str(data_root), "--splits-dir", str(splits_dir)]
    train, select = script("train").main, script("select_lr").main
    cfg = {arm: str(small_config(tmp_path, arm)) for arm in "HDBF"}

    with pytest.raises(SystemExit, match="arm F needs the rate that arm H chose"):
        train(["--config", cfg["F"], "--lr", "1e-5", "--output-root", str(runs_root), "--device", "cpu", *common])
    for lr in ("1e-4", "3e-4", "1e-3"):
        train(["--config", cfg["H"], "--lr", lr, "--output-root", str(runs_root), "--device", "cpu", *common])
    assert json.loads((runs_root / "h_lr3e-4_s0" / "eval_val.json").read_text())["images"] == 6
    select(["--config", cfg["H"], "--runs-root", str(runs_root), "--json"])
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    if status["status"] == "run":  # the best rate was at an end of the list: one more step
        train(["--config", cfg["H"], "--lr", str(status["lr"][0]), "--output-root", str(runs_root), "--device", "cpu", *common])
        select(["--config", cfg["H"], "--runs-root", str(runs_root), "--json"])
        status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "chosen"

    # arms D, B, F: one rate each is enough here; write their choice by hand
    selection = json.loads((runs_root / "selection.json").read_text())
    for arm, lr in (("D", "3e-4"), ("B", "3e-4"), ("F", "3e-5")):
        train(["--config", cfg[arm], "--lr", lr, "--output-root", str(runs_root), "--device", "cpu", *common])
        selection[arm] = {"lr": selection["H"]["lr"] if arm == "F" else float(lr), "body_lr": float(lr) if arm == "F" else None,
                          "val_ap": {lr: 0.0}}
    (runs_root / "selection.json").write_text(json.dumps(selection))

    with pytest.raises(SystemExit) as done:
        script("gate").main(["--runs-root", str(runs_root), "--bootstrap", "20", *common])
    assert done.value.code in (0, 2) and "AP_F − AP_H" in capsys.readouterr().out

    for arm in "HDBF":
        train(["--config", cfg[arm], "--selected", "--seed", "1", "--output-root", str(runs_root), "--device", "cpu", "--no-eval", *common])
    script("evaluate").main(["--final-runs", str(runs_root), "--seeds", "0", "1", "--split", "test", "--device", "cpu", *common])
    assert "8 final checkpoints to score" in capsys.readouterr().out
    script("evaluate").main(["--final-runs", str(runs_root), "--seeds", "0", "1", "--split", "test", "--device", "cpu", *common])
    assert "0 final checkpoints to score" in capsys.readouterr().out  # the test split: one time for each checkpoint
    script("analyse").main(["--runs-root", str(runs_root), "--out", str(tmp_path / "results"), "--bootstrap", "10",
                            "--seeds", "0", "1", "--samples", "2", *common])
    text = (tmp_path / "results" / "report.md").read_text()
    assert "## Sample images" in text and len(list((tmp_path / "results" / "samples").glob("*.png"))) == 2
    assert "The gate" in text and "| H: heads only |" in text


def test_start_model_check_script(data_root, splits_dir, tmp_path, capsys):
    with pytest.raises(SystemExit):
        script("evaluate").main(["--start-model", "--out", str(tmp_path / "start"), "--device", "cpu", "--data-root", str(data_root),
                                 "--splits-dir", str(splits_dir), "--num-workers", "0"])
    assert "check of stage 2" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="val split"):
        script("evaluate").main(["--start-model", "--out", str(tmp_path / "s"), "--split", "test", "--data-root", str(data_root)])
