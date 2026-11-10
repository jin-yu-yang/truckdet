import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests.tiny import write_synthetic_miotcd
from truckdet.data import scenes
from truckdet.data.miotcd import SPECIFIC_CLASSES, boxes_per_class, read_ids

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
    assert done.value.code == 3 and "DIFFERENT from the reference counts" in out and "lists the same 6 classes" in out
    assert json.loads((root / "counts.json").read_text())["images"] == 60 and (tmp_path / "checks" / "labels_01.jpg").exists()

    # The embeddings of the real code path (the tiny backbone). 60 images cannot give 100 boxes of each class, so no
    # limit has a split: the search stops at its first candidate, and the split lists of an earlier run are removed.
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "test.txt").write_text("00000001\n")
    (splits_dir / "notes.txt").write_text("not a split file\n")
    arguments = ["--data-root", str(root), "--splits-dir", str(splits_dir), "--device", "cpu", "--test-images", "12",
                 "--val-images", "6", "--train-images", "20", "--sheets-dir", str(tmp_path / "checks"), "--pairs", "4",
                 "--num-workers", "0"]
    with pytest.raises(SystemExit) as done:
        script("make_split").main(arguments)
    out = capsys.readouterr().out
    assert done.value.code == 3 and "[check] NO LIMIT: the first candidate is over-linked" in out
    assert "the split is not possible: the groups end before the test split is complete" in out
    assert "removed the split files of an earlier run" in out and "run the second embedding one time: --embedding grid" in out
    assert {p.name for p in splits_dir.iterdir()} == {"split_meta.json", "notes.txt"}
    meta = json.loads((splits_dir / "split_meta.json").read_text())
    assert meta["limit"] is None and meta["embedding"] == "mean" and meta["checks"]["passed"] is False
    assert len(meta["search"]["candidates"]) == 1 and meta["search"]["candidates"][0]["over_linked"]
    assert meta["search"]["d_near"] < meta["search"]["d_rand"] and meta["class_minimums"] == {"test": 100, "val": 30}
    assert np.load(root / "embeddings.npy").shape == (60, 24)

    # The second embedding, one time; after it a decision is needed.
    with pytest.raises(SystemExit) as done:
        script("make_split").main([*arguments, "--embedding", "grid"])
    out = capsys.readouterr().out
    assert done.value.code == 3 and "the second embedding failed too. A decision is needed" in out
    grid = np.load(root / "embeddings_grid.npy")
    assert grid.shape == (60, 24 * 16) and grid.dtype == np.float16
    assert json.loads((splits_dir / "split_meta.json").read_text())["embedding"] == "grid"


def planted_cameras(root, n_images: int = 60, n_cameras: int = 6) -> np.ndarray:
    """Write `embeddings.npy`: image k is a frame of camera k mod n_cameras, near the centre of its camera."""
    rng = np.random.default_rng(0)
    camera = np.arange(n_images) % n_cameras
    centres = np.linalg.qr(rng.normal(size=(16, 16)))[0][:n_cameras] + 3.0  # a large common part, as the real ones
    np.save(root / "embeddings.npy", (centres[camera] + rng.normal(size=(n_images, 16)) * 0.003).astype(np.float32))
    return camera


def test_make_split_links_the_frames_of_a_camera(tmp_path, capsys, monkeypatch):
    """The whole stage on 6 planted cameras of 10 frames: the search, the split with caps, the checks, the files."""
    root = write_synthetic_miotcd(tmp_path / "data", n_images=60, scenes=6)
    with pytest.raises(SystemExit):
        script("prepare_data").main(["--data-root", str(root)])
    camera = planted_cameras(root)
    monkeypatch.setattr(scenes, "MIN_TEST_BOXES", 0)  # 60 images do not have the boxes of the real minimums
    monkeypatch.setattr(scenes, "MIN_VAL_BOXES", 0)
    capsys.readouterr()
    splits_dir, checks = tmp_path / "splits", tmp_path / "checks"
    with pytest.raises(SystemExit) as done:
        script("make_split").main(["--data-root", str(root), "--splits-dir", str(splits_dir), "--device", "cpu",
                                   "--test-images", "12", "--val-images", "6", "--train-images", "20", "--test-cap", "8",
                                   "--val-cap", "4", "--sheets-dir", str(checks), "--pairs", "4", "--num-workers", "0"])
    out = capsys.readouterr().out
    assert done.value.code == 0, out
    for line in ("[search] nearest image: median similarity", "the lowest that is not over-linked",
                 "[check] no overlap and class counts: passed", "[check] group check: passed: 0 images",
                 "[check] split check: passed: test 2 groups (2 needed), val 2 groups (2 needed)",
                 "[check] scene check, scene_split: 4 pairs", "[check] scene check, random_split: 4 pairs",
                 "Each pair of the scene split is below the limit", "16 images of these groups are over the cap"):
        assert line in out, line
    assert "OVER-LINKED: the largest group has" in out  # the search went on until two cameras joined

    ids = [f"{k:08d}" for k in range(60)]
    lists = {name: read_ids(splits_dir / f"{name}.txt") for name in ("test", "val", "train_pool", "train", "unused")}
    assert sorted(lists["test"] + lists["val"] + lists["train_pool"] + lists["unused"]) == ids
    assert len(lists["test"]) == 16 and len(lists["val"]) == 8 and len(lists["unused"]) == 16
    assert len(lists["train_pool"]) == 20 and sorted(lists["train"]) == sorted(lists["train_pool"])
    groups = scenes.read_groups(splits_dir / "scene_groups.csv")
    assert [groups[i] for i in ids] == camera.tolist()  # a scene group is a camera
    cameras_of = {name: {groups[i] for i in lists[name]} for name in lists}
    assert len(cameras_of["test"]) == 2 and len(cameras_of["val"]) == 2 and len(cameras_of["train_pool"]) == 2
    assert not cameras_of["test"] & cameras_of["val"] and not (cameras_of["test"] | cameras_of["val"]) & cameras_of["train_pool"]
    assert cameras_of["unused"] == cameras_of["test"] | cameras_of["val"]  # over the cap of their group: in no split

    meta = json.loads((splits_dir / "split_meta.json").read_text())
    search = meta["search"]
    assert meta["limit"] == search["candidates"][search["chosen_candidate"]]["limit"] and meta["groups"] == 6
    assert [row["over_linked"] is None for row in search["candidates"]] == [True] * (len(search["candidates"]) - 1) + [False]
    assert search["chosen_candidate"] == len(search["candidates"]) - 2  # the candidate before the first over-linked one
    assert meta["checks"]["passed"] is True and meta["checks"]["group_check"]["failures"] == 0
    assert meta["checks"]["split_check"]["errors"] == [] and meta["checks"]["split_check"]["test groups"] == 2
    assert meta["checks"]["group_check"]["highest_similarity_outside_a_group"] < meta["limit"]
    assert meta["images"] == {"test": 16, "val": 8, "train_pool": 20, "train": 20, "unused": 16}
    assert meta["caps"] == {"test": 8, "val": 4} and meta["groups_per_split"] == {"test": 2, "val": 2, "train_pool": 2}
    assert meta["largest_group_share"] == pytest.approx(1 / 6) and meta["embedding"] == "mean"
    assert all((checks / name).exists() for name in ("scene_split_01.jpg", "random_split_01.jpg", "largest_group.jpg"))

    # Groups from a file (design §4.3: "if the image IDs show the camera"): the split, with no limit to check.
    with pytest.raises(SystemExit) as done:
        script("make_split").main(["--data-root", str(root), "--splits-dir", str(tmp_path / "by_file"), "--device", "cpu",
                                   "--groups-csv", str(splits_dir / "scene_groups.csv"), "--test-images", "12",
                                   "--val-images", "6", "--train-images", "20", "--test-cap", "8", "--val-cap", "4"])
    out = capsys.readouterr().out
    assert done.value.code == 0 and "group check and split check: not run" in out
    assert read_ids(tmp_path / "by_file" / "test.txt") == lists["test"]  # the same groups give the same split
    by_file = json.loads((tmp_path / "by_file" / "split_meta.json").read_text())
    assert by_file["limit"] is None and by_file["search"] is None and by_file["group_source"] == "groups from scene_groups.csv"


def test_simulation_of_the_scene_groups():
    """`scripts/scene_split_simulation.py` on small copies of two of its cases: cameras that the links separate, and
    cameras that they do not separate. Both failure signals of design §4.3 must work: no limit, or a limit whose
    split fails the scene check."""
    simulation = script("scene_split_simulation")
    small = dict(n_train=300, quiet=True, n_cam=60, median=40)
    apart = simulation.run("A", **small, **simulation.CASES["A: cameras well apart"])
    assert apart["limit"] is not None and apart["scene_check"] == 0.0
    assert apart["group_check_failures"] == 0 and apart["split_check_errors"] == []
    assert apart["groups"] <= apart["cameras"] == 60  # no camera is cut; some similar cameras are in one group
    rows = apart["candidates"]
    assert rows[-1]["over_linked"] and all(row["over_linked"] is None for row in rows[:-1])
    assert [row["groups"] for row in rows] == sorted((row["groups"] for row in rows), reverse=True)  # a lower limit joins groups

    none = simulation.run("D", **small, **simulation.CASES["D: not separable"])
    assert none["limit"] is None and len(none["candidates"]) == 1 and none["candidates"][0]["over_linked"]
    # With fewer cameras the search accepts its first candidate. The automatic checks pass, and the scene check
    # fails: the check with eyes is the one that decides.
    fewer = simulation.run("D", **{**small, "n_cam": 40, "median": 30}, **simulation.CASES["D: not separable"])
    assert fewer["limit"] is not None and fewer["group_check_failures"] == 0 and fewer["split_check_errors"] == []
    assert fewer["scene_check"] > 0.5 and fewer["groups"] > 10 * fewer["cameras"]  # the cameras are cut into small groups


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
    assert done.value.code in (0, 3) and "AP_F − AP_H" in capsys.readouterr().out

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
