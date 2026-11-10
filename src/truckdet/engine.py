"""One epoch of training: forward (fp16 autocast on CUDA) -> the library's set loss (fp32) ->
scaled backward -> every `accum_steps` batches: gradient clipping -> AdamW step -> LR schedule step.

The loss is the library's (`LwDetrForObjectDetectionLoss`: Hungarian matching for each query group,
an IoU-aware sigmoid class loss, L1 and GIoU box losses, the same for the auxiliary decoder layers
and the encoder proposals). The library calculates it inside `forward` when the call has `labels`,
in the dtype of the forward pass. Its class loss takes log(p) and log(1 - p) of sigmoid outputs; in
fp16 p rounds to 0 below a logit of about -17 and to 1 above about 8, and the loss becomes NaN or
inf. So the forward pass runs without `labels`, and `training_loss` gives the same tensors to the
same loss function in fp32.
"""

import math
import time
import warnings
from collections import defaultdict

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .data.dataset import labels_on
from .model.arms import is_head

LOGGED = ("loss", "loss_ce", "loss_bbox", "loss_giou")


def step_schedule(scheduler) -> None:
    """`scheduler.step()`, without one warning of PyTorch.

    In fp16 the gradient scaler can leave out the first optimizer step of a run (seen on the T4 in stage 0).
    PyTorch then warns that the scheduler stepped before the optimizer. The callers do step the optimizer first;
    the step that the scaler left out is counted (`scaler_skips`) and printed, so the warning adds nothing."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
        scheduler.step()


def training_loss(model, pixel_values: torch.Tensor, labels: list[dict], amp: bool = False) -> tuple[torch.Tensor, dict]:
    """(total loss, the library's loss dict) of one batch, for a model in training mode.

    Equal to `model(pixel_values, labels=labels).loss` when `amp` is False. The model must be in
    training mode: the library's loss always matches 13 query groups, and in evaluation mode the
    model gives one."""
    if not model.training:
        raise RuntimeError("the library's loss is only correct for a model in training mode")
    device_type = pixel_values.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp):
        out = model(pixel_values=pixel_values)
        aux_class, aux_boxes = None, None
        if model.config.auxiliary_loss:
            aux_class, aux_boxes = model.predict_class_and_boxes(
                out.intermediate_hidden_states, out.intermediate_reference_points
            )
    with torch.autocast(device_type=device_type, enabled=False):
        loss, loss_dict, _ = model.loss_function(
            out.logits.float(),
            labels,
            pixel_values.device,
            out.pred_boxes.float(),
            model.config,
            None if aux_class is None else aux_class.float(),
            None if aux_boxes is None else aux_boxes.float(),
            out.enc_outputs_class.float(),
            out.enc_outputs_coord_logits.float(),
        )
    return loss, loss_dict


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, body_lr: float | None = None) -> torch.optim.AdamW:
    """AdamW over the trainable parameters. With `body_lr` (arm F), the heads use `lr` and all other
    parameters use `body_lr`; without it, one rate for everything trainable."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if body_lr is None:
        groups = [{"params": [p for _, p in named], "lr": lr, "name": "all"}]
    else:
        groups = [
            {"params": [p for n, p in named if is_head(n)], "lr": lr, "name": "heads"},
            {"params": [p for n, p in named if not is_head(n)], "lr": body_lr, "name": "body"},
        ]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps: int, warmup_fraction: float = 0.05) -> LambdaLR:
    """Linear warm-up for the first `warmup_fraction` of the steps, then a cosine decrease to zero."""
    warmup = max(1, round(warmup_fraction * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def optimizer_steps(batches: int, accum_steps: int) -> int:
    """Optimizer steps in an epoch of `batches` batches: one for every `accum_steps`, and one for a last partial group."""
    return math.ceil(batches / accum_steps)


class NanGuard:
    """Skip non-finite batches (rare fp16 blow-ups); abort if they keep happening."""

    def __init__(self, max_consecutive: int = 10):
        self.max_consecutive = max_consecutive
        self.consecutive = 0
        self.skipped = 0

    def check(self, loss: torch.Tensor) -> bool:
        if torch.isfinite(loss):
            self.consecutive = 0
            return True
        self.consecutive += 1
        self.skipped += 1
        print(f"[train] non-finite loss ({loss.item()}); skipping batch ({self.consecutive} in a row)")
        if self.consecutive >= self.max_consecutive:
            raise FloatingPointError(f"{self.consecutive} consecutive non-finite losses; aborting")
        return False


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    clip_max_norm: float,
    amp: bool,
    nan_guard: NanGuard,
    accum_steps: int = 1,
    log_every: int = 50,
    epoch: int = 0,
) -> dict:
    """Returns the mean of each logged loss over the batches, and:
    - `steps`: optimizer steps taken;
    - `skipped`: batches left out for a non-finite loss (design, stage 6: must be 0);
    - `scaler_skips`: optimizer steps that the fp16 gradient scaler did not apply because a gradient was
      not finite. The scaler lowers its scale and goes on; a few at the start of a run are normal."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    use_amp = amp and device.type == "cuda"
    sums, batches, steps, scaler_skips = defaultdict(float), 0, 0, 0
    skipped_before, start = nan_guard.skipped, time.time()
    images = 0
    optimizer.zero_grad(set_to_none=True)
    pending = 0  # batches whose gradients wait for the next optimizer step
    n_batches = len(loader)
    for i, (pixel_values, targets) in enumerate(loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        loss, parts = training_loss(model, pixel_values, labels_on(targets, device), use_amp)
        images += len(targets)
        if nan_guard.check(loss.detach()):
            scaler.scale(loss / accum_steps).backward()
            pending += 1
            batches += 1
            sums["loss"] += loss.item()
            for key in LOGGED[1:]:
                sums[key] += parts[key].item()
        if (i + 1) % accum_steps == 0 or i + 1 == n_batches:
            if pending:
                if clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, clip_max_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scaler_skips += int(scaler.get_scale() < scale_before)
                steps += 1
            optimizer.zero_grad(set_to_none=True)
            step_schedule(scheduler)  # the schedule follows the planned steps, also when a step had nothing to apply
            pending = 0
        if log_every and i % log_every == 0:
            print(
                f"[train] epoch {epoch} batch {i}/{n_batches} loss {loss.item():.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
    elapsed = time.time() - start
    stats = {key: sums[key] / max(batches, 1) for key in LOGGED}
    stats.update(steps=steps, skipped=nan_guard.skipped - skipped_before, scaler_skips=scaler_skips,
                 time_s=elapsed, images_per_s=images / max(elapsed, 1e-9))
    return stats
