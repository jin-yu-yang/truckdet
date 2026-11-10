import copy

import pytest
import torch

from tests.conftest import tiny_config
from tests.tiny import TINY_SIZE, tiny_loader
from truckdet.engine import (
    LOGGED, NanGuard, build_optimizer, build_scheduler, optimizer_steps, train_one_epoch, training_loss,
)
from truckdet.model import arms

CPU = torch.device("cpu")


def batch(n: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randn(n, 3, TINY_SIZE, TINY_SIZE, generator=g)
    targets = [{"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]])},
               {"class_labels": torch.tensor([1, 4]), "boxes": torch.tensor([[0.7, 0.6, 0.3, 0.2], [0.2, 0.8, 0.1, 0.1]])}]
    return pixels, (targets * n)[:n]


def model_for(arm: str):
    model = arms.build_start_model("tiny", tiny_loader)
    torch.manual_seed(0)
    arms.apply_arm(model, arm, tiny_config(arm).lora)
    return model


@pytest.mark.parametrize("arm", ["H", "D", "B", "F"])
def test_training_loss_is_the_librarys_loss(arm):
    model = model_for(arm).train()
    pixels, targets = batch()
    loss, parts = training_loss(model, pixels, targets)
    library = model(pixel_values=pixels, labels=targets)
    assert torch.equal(loss, library.loss)
    assert set(LOGGED[1:]) <= set(parts) and {"loss_ce_0", "loss_ce_enc"} <= set(parts)  # aux layer and proposals
    assert all(torch.equal(parts[k], library.loss_dict[k]) for k in parts)
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and grads <= {n for n, p in model.named_parameters() if p.requires_grad}


def test_training_loss_refuses_evaluation_mode():
    """The library's loss always matches 13 query groups; in evaluation mode the model gives one."""
    pixels, targets = batch()
    with pytest.raises(RuntimeError, match="training mode"):
        training_loss(model_for("H").eval(), pixels, targets)


def test_scheduler_warmup_then_cosine():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    sched = build_scheduler(opt, total_steps=200, warmup_fraction=0.05)  # 10 warm-up steps
    lrs = []
    for _ in range(200):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10])) and all(a > b for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[105] == pytest.approx(0.5, abs=0.01) and lrs[-1] < 1e-3


def test_optimizer_groups():
    for arm in "HDB":
        opt = build_optimizer(model_for(arm), lr=3e-4, weight_decay=1e-4)
        assert [g["lr"] for g in opt.param_groups] == [3e-4] and opt.param_groups[0]["weight_decay"] == 1e-4
    model = model_for("F")
    opt = build_optimizer(model, lr=3e-4, weight_decay=1e-4, body_lr=1e-5)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["heads"]["lr"] == 3e-4 and by_name["body"]["lr"] == 1e-5
    heads = sum(p.numel() for n, p in model.named_parameters() if arms.is_head(n))
    assert sum(p.numel() for p in by_name["heads"]["params"]) == heads
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == sum(p.numel() for p in model.parameters())


def test_nan_guard():
    guard = NanGuard(max_consecutive=3)
    assert guard.check(torch.tensor(1.0))
    assert not guard.check(torch.tensor(float("nan")))
    assert not guard.check(torch.tensor(float("inf")))
    assert guard.check(torch.tensor(2.0))  # resets the streak
    assert not guard.check(torch.tensor(float("nan")))
    assert not guard.check(torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError):
        guard.check(torch.tensor(float("nan")))
    assert guard.skipped == 5


def run_epoch(model, n_batches: int, accum_steps: int, lr: float = 1e-3, total_steps: int = 10_000):
    optimizer = build_optimizer(model, lr=lr, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, total_steps, warmup_fraction=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stats = train_one_epoch(model, [batch()] * n_batches, optimizer, scheduler, scaler, CPU, clip_max_norm=1.0,
                            amp=False, nan_guard=NanGuard(), accum_steps=accum_steps, log_every=0)
    return stats, scheduler


def test_accumulation_steps():
    assert [optimizer_steps(b, 4) for b in (8, 9, 1, 2000)] == [2, 3, 1, 500]
    stats, scheduler = run_epoch(model_for("H"), n_batches=9, accum_steps=4)
    assert stats["steps"] == 3 and scheduler.last_epoch == 3  # 4 + 4 + a last group of 1
    assert stats["skipped"] == 0 and stats["scaler_skips"] == 0 and stats["images_per_s"] > 0


def test_accumulating_equal_batches_is_one_step_on_their_mean_gradient():
    """4 x the same batch with accumulation 4 is the step of 1 batch with accumulation 1 (SGD-like check on AdamW's
    first step, whose size does not depend on the gradient's scale but whose direction does)."""
    a, b = model_for("H"), model_for("H")
    run_epoch(a, n_batches=4, accum_steps=4)
    run_epoch(b, n_batches=1, accum_steps=1)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


@pytest.mark.parametrize("arm", ["H", "D", "B", "F"])
def test_overfits_one_batch(arm):
    """End-to-end plumbing check: model + library loss + optimizer can drive the loss down, in every arm."""
    model = model_for(arm)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    pixels, targets = batch()
    start = training_loss(model.train(), pixels, targets)[0].item()
    stats, _ = run_epoch(model, n_batches=60, accum_steps=1, lr=3e-3)
    end = training_loss(model.train(), pixels, targets)[0].item()
    assert stats["steps"] == 60 and stats["skipped"] == 0
    assert end < 0.8 * start
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)  # frozen stays frozen


def test_nan_batches_are_skipped_without_updating(monkeypatch):
    model = model_for("D")
    before = copy.deepcopy(model.state_dict())

    def nan_loss(model, pixel_values, labels, amp=False):
        return model.class_embed.weight.sum() * float("nan"), {}

    monkeypatch.setattr("truckdet.engine.training_loss", nan_loss)
    optimizer = build_optimizer(model, lr=1e-2, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, 100)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    stats = train_one_epoch(model, [batch()] * 4, optimizer, scheduler, scaler, CPU, 0.1, False, NanGuard(), accum_steps=2, log_every=0)
    assert stats["steps"] == 0 and stats["skipped"] == 4
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert scheduler.last_epoch == 2  # the schedule follows the planned steps


def test_a_step_that_the_gradient_scaler_skips_is_counted(monkeypatch, recwarn):
    """What fp16 does on the T4: the loss is finite, a scaled gradient is not. The scaler leaves the step out and
    halves its scale. The step is counted, the schedule goes on, and the later steps change the weights."""
    model = model_for("H")
    start = copy.deepcopy(model.state_dict())
    after_first = {}

    def loss_fn(model, pixel_values, labels, amp=False):
        loss, parts = training_loss(model, pixel_values, labels, amp)
        if not after_first:  # the first batch: sqrt has no finite gradient at 0, and its value there is 0
            loss = loss + torch.sqrt(model.class_embed.weight.sum() * 0)
        else:
            after_first.setdefault("weights", copy.deepcopy(model.state_dict()))
        after_first.setdefault("called", True)
        return loss, parts

    monkeypatch.setattr("truckdet.engine.training_loss", loss_fn)
    optimizer = build_optimizer(model, lr=1e-2, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, 100)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    stats = train_one_epoch(model, [batch()] * 3, optimizer, scheduler, scaler, CPU, 0.1, False, NanGuard(),
                            accum_steps=1, log_every=0)
    assert stats["steps"] == 3 and stats["scaler_skips"] == 1 and stats["skipped"] == 0
    assert scaler.get_scale() == 65536 / 2
    assert all(torch.equal(v, start[k]) for k, v in after_first["weights"].items())  # the skipped step changed nothing
    assert any(not torch.equal(v, start[k]) for k, v in model.state_dict().items())  # the next steps did
    assert scheduler.last_epoch == 3
    assert not [w for w in recwarn if "lr_scheduler.step()" in str(w.message)]  # explained in engine.py; not shown
