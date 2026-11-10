"""Checkpoints hold only the trainable tensors plus optimizer/scheduler state: the heads and LoRA
for arms H, D and B (a few MB), every parameter for arm F.

Writes are atomic (temp file + rename), so a Colab disconnect mid-save cannot corrupt a file.
`meta` (class names, full config) lets evaluation rebuild the model from the checkpoint alone.

- `last.pt`: written after every epoch, with the optimizer state. A re-run resumes from it.
- `final.pt`: written when the run ends, without the optimizer state. It is the result of the run;
  `last.pt` is then removed.
"""

import os
from pathlib import Path

import torch

from .model.arms import load_trainable_state_dict, trainable_state_dict


def _write(state: dict, path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch: int, meta: dict, skipped: int = 0) -> None:
    _write({
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "skipped": skipped,
        "meta": meta,
    }, path)


def save_final(path, model, epoch: int, meta: dict) -> None:
    _write({"model": trainable_state_dict(model), "epoch": epoch, "meta": meta}, path)


def _load(path) -> dict:
    # weights_only=False: our own file; the optimizer state and meta are not only tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


def read_meta(path) -> dict:
    return _load(path)["meta"]


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> dict:
    state = _load(path)
    load_trainable_state_dict(model, state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return {"epoch": state["epoch"], "meta": state["meta"], "skipped": state.get("skipped", 0)}
