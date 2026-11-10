"""Run configuration: one YAML file per arm, loaded into dataclasses.

An arm's file holds what is fixed for that arm (the adapter, the recipe, the learning-rate list).
A run adds a learning rate and a seed (`for_run`), and its name records both.
"""

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

ARMS = ("H", "D", "B", "F")
LORA_ARMS = ("D", "B")

# One step up or down from a learning-rate list (design §6: "add one step in that direction, one time").
LR_LADDER = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass
class LoRAConfig:
    r: int
    alpha: float
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 12
    # Effective batch = batch_size * accum_steps = 16 images. Stage 0 sets the split by memory.
    batch_size: int = 4
    accum_steps: int = 4
    # Arms H, D, B: the one learning rate of all trainable parameters. Arm F: the learning rate of the heads
    # (the rate that H chose). None in the arm's file; a run sets it.
    lr: float | None = None
    # Arm F only: the learning rate of all parameters other than the heads.
    body_lr: float | None = None
    # The rates to try with seed 0 (for arm F these are body rates).
    lr_grid: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    clip_max_norm: float = 0.1
    amp: bool = True
    flip: bool = True
    num_workers: int = 2
    eval_batch_size: int = 16
    seed: int = 0

    def __post_init__(self):
        if self.batch_size < 1 or self.accum_steps < 1:
            raise ValueError("train.batch_size and train.accum_steps must be at least 1")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"train.warmup_fraction must be in [0, 1), got {self.warmup_fraction}")


@dataclass
class Config:
    arm: str
    model_name: str = "Roboflow/rf-detr-base"
    lora: LoRAConfig | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str | None = None  # set by for_run

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if (self.arm in LORA_ARMS) != (self.lora is not None):
            raise ValueError(f"arm {self.arm}: a `lora` section is required for arms {LORA_ARMS} and not allowed for others")


def config_from_dict(d: dict) -> Config:
    d = dict(d)
    lora = d.pop("lora", None)
    return Config(
        arm=d.pop("arm"),
        lora=LoRAConfig(**lora) if lora else None,
        train=TrainConfig(**d.pop("train", {})),
        **d,
    )


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return config_from_dict(yaml.safe_load(f))


def format_lr(lr: float) -> str:
    """3e-4 -> '3e-4', 0.001 -> '1e-3', 2.5e-5 -> '2.5e-5': short and safe in a folder name."""
    mantissa, exponent = f"{lr:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def make_run_name(arm: str, lr: float, seed: int, body_lr: float | None = None) -> str:
    """`b_lr3e-4_s0`; arm F names its body rate first, then the head rate: `f_lr3e-5_h3e-4_s0`."""
    if arm == "F":
        if body_lr is None:
            raise ValueError("arm F needs a body learning rate")
        return f"f_lr{format_lr(body_lr)}_h{format_lr(lr)}_s{seed}"
    return f"{arm.lower()}_lr{format_lr(lr)}_s{seed}"


def for_run(cfg: Config, lr: float, seed: int, body_lr: float | None = None) -> Config:
    """Copy of the arm's config for one run. `lr` is the rate of all trainable parameters (H, D, B) or of the
    heads (F); `body_lr` is F's rate for all other parameters."""
    if (cfg.arm == "F") != (body_lr is not None):
        raise ValueError("body_lr is required for arm F and not allowed for the other arms")
    train = replace(cfg.train, lr=lr, body_lr=body_lr, seed=seed)
    return replace(cfg, train=train, run_name=make_run_name(cfg.arm, lr, seed, body_lr))


_RUN_NAME = re.compile(r"^(?P<arm>[hdbf])_lr(?P<lr>[0-9.e+-]+?)(?:_h(?P<head>[0-9.e+-]+?))?_s(?P<seed>\d+)$")


def parse_run_name(name: str) -> dict | None:
    """`b_lr3e-4_s0` -> {"arm": "B", "lr": 3e-4, "body_lr": None, "seed": 0}; None if the name is not a run name.
    For arm F, "lr" is the head rate and "body_lr" the body rate, as in `for_run`."""
    m = _RUN_NAME.match(name)
    if not m:
        return None
    arm, first, head = m["arm"].upper(), float(m["lr"]), m["head"]
    if (arm == "F") != (head is not None):
        return None
    if arm == "F":
        return {"arm": arm, "lr": float(head), "body_lr": first, "seed": int(m["seed"])}
    return {"arm": arm, "lr": first, "body_lr": None, "seed": int(m["seed"])}
