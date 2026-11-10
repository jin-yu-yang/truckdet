"""Full training run: data -> start model -> arm -> a fixed number of epochs -> the final checkpoint.

There is no early stopping and no model selection: the checkpoint after the last epoch is the result
(design §3, decision 9). Re-running with the same output directory resumes from `last.pt`. That is
how the Colab notebook survives disconnects. A finished run trains nothing.

What the seed sets: the adapters' start (A matrices), the order of the images, the flips and the
dropout masks. Epoch e of seed s uses `epoch_seed(s, e)` for all of them, so a resumed run has the
epochs that an uninterrupted run would have had.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from .checkpoint import load_checkpoint, save_checkpoint, save_final
from .config import Config, config_to_dict
from .data.dataset import EpochSampler, MioTcdDetection, collate
from .data.miotcd import CLASSES, load_annotations, split_image_ids
from .engine import NanGuard, build_optimizer, build_scheduler, optimizer_steps, train_one_epoch
from .model.arms import apply_arm, build_start_model, format_param_table, load_pretrained, param_table
from .reporting import append_jsonl, plot_curves, read_jsonl, write_jsonl

ANNOTATIONS = "annotations.json"
IMAGE_DIR = "train"  # MIO-TCD's folder of labelled images; it holds all of our splits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epoch_seed(seed: int, epoch: int) -> int:
    """Mixing (seed, epoch) through SeedSequence keeps seeds apart; `seed + epoch` would give seed 0 at
    epoch 1 the randomness of seed 1 at epoch 0."""
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1, dtype=np.uint64)[0] % 2**63)


def load_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def resolve_loaders(model_loader=None, processor_loader=None):
    """The loaders to use: the given ones, or the library's. The names are looked up at call time, so a test
    can replace `training.load_pretrained` and `training.load_processor` for every caller."""
    return model_loader or load_pretrained, processor_loader or load_processor


def build_model(cfg: Config, model_loader=None):
    """The start model with the arm applied. The adapters' start is a function of `cfg.train.seed`."""
    model = build_start_model(cfg.model_name, resolve_loaders(model_loader)[0])
    torch.manual_seed(cfg.train.seed)
    apply_arm(model, cfg.arm, cfg.lora)
    return model


def run_training(
    cfg: Config,
    data_root: Path,
    splits_dir: Path,
    output_root: Path,
    device: str | torch.device = "cuda",
    epochs: int | None = None,
    train_ids: list[int] | None = None,
    model_loader=None,
    processor_loader=None,
    log_every: int = 50,
    stop_after: int | None = None,
) -> Path:
    """Train `cfg` (made by `config.for_run`) and return the run folder, which then holds `final.pt`.

    `train_ids` (COCO ids) replaces the train subset of `splits_dir/train.txt`; the checks of stage 4 use it
    to train on 16 images. `stop_after` ends the call after that many epochs of the run, as a disconnect
    would: `last.pt` stays, there is no `final.pt`, and the next call resumes."""
    if cfg.run_name is None or cfg.train.lr is None:
        raise ValueError("the config has no run name or learning rate; make it with config.for_run")
    device = torch.device(device)
    model_loader, processor_loader = resolve_loaders(model_loader, processor_loader)
    data_root, output_root = Path(data_root), Path(output_root)
    epochs = epochs if epochs is not None else cfg.train.epochs
    train = cfg.train
    run_dir = output_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path, last_path, metrics_path = run_dir / "final.pt", run_dir / "last.pt", run_dir / "metrics.jsonl"
    if final_path.exists():
        print(f"[train] {cfg.run_name}: already finished ({final_path})")
        return run_dir

    seed_everything(train.seed)
    coco = load_annotations(data_root / ANNOTATIONS)
    if train_ids is None:
        train_ids = split_image_ids(coco, Path(splits_dir) / "train.txt")
    dataset = MioTcdDetection(coco, data_root / IMAGE_DIR, train_ids, processor_loader(cfg.model_name))
    sampler = EpochSampler(len(dataset), train.seed, flip=train.flip)
    loader = DataLoader(
        dataset, batch_size=train.batch_size, sampler=sampler, collate_fn=collate, num_workers=train.num_workers,
        pin_memory=device.type == "cuda", drop_last=len(dataset) >= train.batch_size,
    )

    model = build_model(cfg, model_loader).to(device)
    table = param_table(model)
    print(f"[train] {cfg.run_name}: arm {cfg.arm}, {len(dataset)} train images, {epochs} epochs, "
          f"batch {train.batch_size} x {train.accum_steps}, lr {train.lr}"
          + (f", body lr {train.body_lr}" if train.body_lr is not None else ""))
    print(format_param_table(table))
    (run_dir / "params.json").write_text(json.dumps(
        {"arm": cfg.arm, "table": table, "class_names": list(CLASSES), "train_images": len(dataset)}, indent=2))

    optimizer = build_optimizer(model, train.lr, train.weight_decay, train.body_lr)
    steps_per_epoch = optimizer_steps(len(loader), train.accum_steps)
    scheduler = build_scheduler(optimizer, epochs * steps_per_epoch, train.warmup_fraction)
    scaler = torch.amp.GradScaler(device.type, enabled=train.amp and device.type == "cuda")
    meta = {"class_names": list(CLASSES), "config": config_to_dict(cfg), "epochs": epochs}

    start_epoch = 0
    guard = NanGuard()
    if last_path.exists():
        state = load_checkpoint(last_path, model, optimizer, scheduler, scaler)
        if state["meta"]["epochs"] != epochs:
            raise ValueError(f"{last_path} is from a run of {state['meta']['epochs']} epochs, this run has {epochs}: "
                             "the learning-rate schedule would not continue. Use a new output folder.")
        start_epoch = state["epoch"] + 1
        guard.skipped = state["skipped"]
        print(f"[train] resuming from epoch {start_epoch}")
    # Keep only rows for epochs whose checkpoint was saved; a crash between the row append and the
    # last.pt save leaves a row that the re-run will write again.
    write_jsonl(metrics_path, [r for r in read_jsonl(metrics_path) if r["epoch"] < start_epoch])

    epoch_times: list[float] = []
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        torch.manual_seed(epoch_seed(train.seed, epoch))  # dropout masks of this epoch
        stats = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, device, train.clip_max_norm, train.amp, guard,
            accum_steps=train.accum_steps, log_every=log_every, epoch=epoch,
        )
        append_jsonl(metrics_path, {"epoch": epoch, **stats, "lr": scheduler.get_last_lr()[0]})
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, meta, skipped=guard.skipped)
        print(f"[train] epoch {epoch} done in {stats['time_s']:.0f}s ({stats['images_per_s']:.1f} images/s), "
              f"loss {stats['loss']:.4f}", flush=True)
        epoch_times.append(stats["time_s"])
        remaining = epochs - epoch - 1
        mean_time = sum(epoch_times) / len(epoch_times)
        print(f"[train] ETA {remaining * mean_time / 60:.0f} min ({remaining} epochs x {mean_time:.0f}s mean)", flush=True)
        if stop_after is not None and epoch + 1 >= stop_after and remaining > 0:
            print(f"[train] stopped after epoch {epoch} (stop_after={stop_after}); run again to resume")
            return run_dir

    save_final(final_path, model, epochs - 1, {**meta, "skipped": guard.skipped})
    last_path.unlink(missing_ok=True)
    plot_curves(metrics_path, run_dir / "curves.png")
    if guard.skipped:
        print(f"[train] WARNING: {guard.skipped} batches were skipped for a non-finite loss")
    return run_dir
