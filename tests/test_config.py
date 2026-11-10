from pathlib import Path

import pytest

from truckdet.config import (
    Config, LoRAConfig, TrainConfig, config_from_dict, config_to_dict, for_run, format_lr, load_config, make_run_name,
    parse_run_name,
)

CONFIGS = Path(__file__).parent.parent / "configs"


def test_format_lr():
    assert [format_lr(x) for x in (3e-4, 0.001, 1e-5, 2.5e-5, 3e-3)] == ["3e-4", "1e-3", "1e-5", "2.5e-5", "3e-3"]


def test_run_names_round_trip():
    assert make_run_name("B", 3e-4, 0) == "b_lr3e-4_s0"
    assert make_run_name("F", 3e-4, 2, body_lr=3e-5) == "f_lr3e-5_h3e-4_s2"
    assert parse_run_name("b_lr3e-4_s0") == {"arm": "B", "lr": 3e-4, "body_lr": None, "seed": 0}
    assert parse_run_name("f_lr3e-5_h3e-4_s2") == {"arm": "F", "lr": 3e-4, "body_lr": 3e-5, "seed": 2}
    for name in ("start_model", "b_lr3e-4", "f_lr3e-5_s0", "b_lr3e-4_h1e-4_s0", "x_lr1e-3_s0"):
        assert parse_run_name(name) is None
    with pytest.raises(ValueError):
        make_run_name("F", 3e-4, 0)


def test_for_run_sets_rates_seed_and_name():
    cfg = for_run(Config(arm="H"), 1e-3, 1)
    assert (cfg.train.lr, cfg.train.body_lr, cfg.train.seed, cfg.run_name) == (1e-3, None, 1, "h_lr1e-3_s1")
    f = for_run(Config(arm="F"), 3e-4, 0, body_lr=1e-5)
    assert (f.train.lr, f.train.body_lr, f.run_name) == (3e-4, 1e-5, "f_lr1e-5_h3e-4_s0")
    with pytest.raises(ValueError):
        for_run(Config(arm="F"), 3e-4, 0)
    with pytest.raises(ValueError):
        for_run(Config(arm="H"), 3e-4, 0, body_lr=1e-5)


def test_validation():
    with pytest.raises(ValueError):
        Config(arm="X")
    with pytest.raises(ValueError):
        Config(arm="B")  # a LoRA arm with no adapter
    with pytest.raises(ValueError):
        Config(arm="H", lora=LoRAConfig(r=2, alpha=4))
    with pytest.raises(ValueError):
        TrainConfig(accum_steps=0)


def test_dict_round_trip():
    cfg = for_run(Config(arm="D", lora=LoRAConfig(r=33, alpha=66)), 3e-4, 2)
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_arm_configs_follow_the_design():
    """Design §5.3 and §6: ranks 9 and 33 with alpha = 2 x rank; 12 epochs; an effective batch of 16; the rate lists."""
    cfgs = {arm: load_config(CONFIGS / f"arm_{arm.lower()}.yaml") for arm in "HDBF"}
    assert {arm: cfg.arm for arm, cfg in cfgs.items()} == {arm: arm for arm in "HDBF"}
    assert (cfgs["B"].lora.r, cfgs["B"].lora.alpha) == (9, 18)
    assert (cfgs["D"].lora.r, cfgs["D"].lora.alpha) == (33, 66)
    assert cfgs["H"].lora is None and cfgs["F"].lora is None
    for arm in "HDB":
        assert cfgs[arm].train.lr_grid == [1e-4, 3e-4, 1e-3]
    assert cfgs["F"].train.lr_grid == [1e-5, 3e-5, 1e-4]
    for cfg in cfgs.values():
        t = cfg.train
        assert (t.epochs, t.batch_size * t.accum_steps, t.weight_decay, t.clip_max_norm, t.warmup_fraction) == (12, 16, 1e-4, 0.1, 0.05)
        assert t.amp and t.flip and cfg.model_name == "Roboflow/rf-detr-base"
