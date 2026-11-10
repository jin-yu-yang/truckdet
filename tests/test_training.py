import json

import pytest
import torch

from tests.conftest import tiny_config
from tests.tiny import make_tiny_rfdetr
from truckdet.checkpoint import _load, read_meta
from truckdet.config import for_run
from truckdet.model import arms
from truckdet.reporting import read_jsonl
from truckdet.training import epoch_seed, run_training


def train(arm, data_root, splits_dir, out, seed=0, **kwargs):
    cfg = for_run(tiny_config(arm), 1e-3, seed, body_lr=1e-4 if arm == "F" else None)
    return run_training(cfg, data_root, splits_dir, out, device="cpu", log_every=0, **kwargs)


@pytest.mark.parametrize("arm", ["H", "D", "B", "F"])
def test_run_writes_the_final_checkpoint(arm, data_root, splits_dir, tmp_path):
    run_dir = train(arm, data_root, splits_dir, tmp_path / "runs")
    assert run_dir.name == {"H": "h_lr1e-3_s0", "D": "d_lr1e-3_s0", "B": "b_lr1e-3_s0", "F": "f_lr1e-4_h1e-3_s0"}[arm]
    assert (run_dir / "final.pt").exists() and not (run_dir / "last.pt").exists() and (run_dir / "curves.png").exists()
    rows = read_jsonl(run_dir / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [0, 1]
    # 12 train images, batch 2 -> 6 batches, accumulation 2 -> 3 optimizer steps per epoch
    assert all(r["steps"] == 3 and r["skipped"] == 0 for r in rows) and rows[-1]["lr"] == pytest.approx(0.0, abs=1e-12)
    state = _load(run_dir / "final.pt")
    params = json.loads((run_dir / "params.json").read_text())
    assert sum(v.numel() for v in state["model"].values()) == params["table"]["total"]["trainable"]
    if arm != "F":  # trainable parameters only: heads and adapters
        assert all(arms.is_head(n) or arms.is_adapter(n) for n in state["model"])
    meta = read_meta(run_dir / "final.pt")
    assert meta["config"]["arm"] == arm and meta["epochs"] == 2 and meta["skipped"] == 0 and meta["scaler_skips"] == 0
    assert "optimizer" not in state


def test_finished_run_trains_nothing(data_root, splits_dir, tmp_path, capsys):
    run_dir = train("H", data_root, splits_dir, tmp_path / "runs")
    stamp = (run_dir / "final.pt").stat().st_mtime_ns
    train("H", data_root, splits_dir, tmp_path / "runs")
    assert (run_dir / "final.pt").stat().st_mtime_ns == stamp and "already finished" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["B", "F"])
def test_resumed_run_is_the_uninterrupted_run(arm, data_root, splits_dir, tmp_path, monkeypatch):
    """Dropout is on (as in the real decoder): the order, the flips and the dropout masks of an epoch are a
    function of (seed, epoch), so stopping after epoch 1 and resuming gives the same weights."""
    monkeypatch.setattr("truckdet.training.load_pretrained", lambda name: make_tiny_rfdetr(dropout=0.1))
    whole = train(arm, data_root, splits_dir, tmp_path / "whole", epochs=3)
    parts = train(arm, data_root, splits_dir, tmp_path / "parts", epochs=3, stop_after=1)
    assert (parts / "last.pt").exists() and not (parts / "final.pt").exists()
    assert [r["epoch"] for r in read_jsonl(parts / "metrics.jsonl")] == [0]
    parts = train(arm, data_root, splits_dir, tmp_path / "parts", epochs=3)
    a, b = _load(whole / "final.pt")["model"], _load(parts / "final.pt")["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    rows_a, rows_b = read_jsonl(whole / "metrics.jsonl"), read_jsonl(parts / "metrics.jsonl")
    assert [r["loss"] for r in rows_a] == [r["loss"] for r in rows_b]  # the loss curve continues


def test_resume_with_another_length_is_refused(data_root, splits_dir, tmp_path):
    train("H", data_root, splits_dir, tmp_path / "runs", epochs=3, stop_after=1)
    with pytest.raises(ValueError, match="schedule"):
        train("H", data_root, splits_dir, tmp_path / "runs", epochs=5)


def test_seed_sets_the_adapters_and_the_order(data_root, splits_dir, tmp_path):
    a = _load(train("D", data_root, splits_dir, tmp_path / "a", seed=0) / "final.pt")["model"]
    again = _load(train("D", data_root, splits_dir, tmp_path / "b", seed=0) / "final.pt")["model"]
    other = _load(train("D", data_root, splits_dir, tmp_path / "c", seed=1) / "final.pt")["model"]
    assert all(torch.equal(a[k], again[k]) for k in a)  # the same seed is the same run
    lora_a = next(k for k in a if "lora_A" in k)
    assert not torch.equal(a[lora_a], other[lora_a])


def test_epoch_seeds_keep_seeds_apart():
    seeds = {epoch_seed(s, e) for s in range(3) for e in range(12)}
    assert len(seeds) == 36 and epoch_seed(0, 1) != epoch_seed(1, 0)


def test_config_without_a_rate_is_refused(data_root, splits_dir, tmp_path):
    with pytest.raises(ValueError, match="for_run"):
        run_training(tiny_config("H"), data_root, splits_dir, tmp_path, device="cpu")


def test_train_ids_replace_the_train_subset(data_root, splits_dir, tmp_path):
    run_dir = train("H", data_root, splits_dir, tmp_path / "runs", train_ids=[0, 1, 2, 3])
    assert json.loads((run_dir / "params.json").read_text())["train_images"] == 4
    assert all(r["steps"] == 1 for r in read_jsonl(run_dir / "metrics.jsonl"))  # 2 batches, accumulation 2
