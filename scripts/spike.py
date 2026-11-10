"""Stage 0: the facts about the model that the design depends on (§5.1 to §5.3), on the real checkpoint.

    python scripts/spike.py                         # the checks; a CPU is enough
    python scripts/spike.py --time --device cuda    # and images per second and peak memory of each arm
    python scripts/spike.py --image-dir <folder> --sheet /content/checks/coco_detections.jpg   # and 20 images

Checks: the parameter table and the trainable parameters of each arm; a change to q_proj of a backbone layer
that the detector reads changes the logits, and a change to a later layer does not; each arm gives the output
of the start model at step 0; an adapter changes the output when its B matrix is not zero; the merge
difference (in float64) is below 1e-4.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from PIL import Image

from truckdet.config import load_config
from truckdet.engine import NanGuard, build_optimizer, build_scheduler, training_loss
from truckdet.model.arms import (
    apply_arm, build_start_model, format_param_table, inference_head_params, param_table,
    used_backbone_layers,
)
from truckdet.evaluation import merge_difference
from truckdet.model.lora import LoRALinear
from truckdet.sheets import contact_sheet, draw_boxes
from truckdet.training import resolve_loaders

# Design §5.2 and §5.3. F is the count measured on the real checkpoint: its decoder samples 2 points per head
# (`decoder_n_points`), the library default is 4, so the decoder has 4,613,152 parameters and not 4,687,168.
EXPECTED_TRAINABLE = {"H": 1_896_146, "D": 2_048_210, "B": 2_048_210, "F": 31_886_690}
EXPECTED_ADAPTER = 152_064
EXPECTED_INFERENCE_HEADS = 270_878
IMAGES_PER_RUN = 96_000  # 8,000 images x 12 epochs


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAILED'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def synthetic_batch(batch_size: int, size: int, device, boxes: int = 3, classes: int = 11):
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(batch_size, 3, size, size, generator=generator).to(device)
    labels = []
    for _ in range(batch_size):
        centre = 0.2 + 0.6 * torch.rand(boxes, 2, generator=generator)
        extent = 0.05 + 0.2 * torch.rand(boxes, 2, generator=generator)
        labels.append({"class_labels": torch.randint(0, classes, (boxes,), generator=generator).to(device),
                       "boxes": torch.cat([centre, extent], dim=1).to(device)})
    return pixel_values, labels


def time_arm(model, device, batch_size: int, size: int, steps: int, amp: bool) -> tuple[float, float]:
    """(images per second, peak memory in GB) of training steps on a synthetic batch."""
    model.to(device).train()
    optimizer = build_optimizer(model, 1e-5, 1e-4)
    scheduler = build_scheduler(optimizer, steps + 3)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    pixel_values, labels = synthetic_batch(batch_size, size, device)
    guard = NanGuard()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps + 3):
        if step == 3:  # the first steps warm up
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
        loss, _ = training_loss(model, pixel_values, labels, amp and device.type == "cuda")
        guard.check(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return steps * batch_size / elapsed, peak


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Roboflow/rf-detr-base")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time", action="store_true", help="measure images per second and peak memory of each arm")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--image-dir", type=Path, help="draw the COCO checkpoint's detections on 20 images of this folder")
    parser.add_argument("--sheet", type=Path, default=Path("coco_detections.jpg"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    ok = True

    load_model, load_processor = resolve_loaders()
    processor = load_processor(args.model_name)
    size = processor.size.height
    raw = load_model(args.model_name).eval()
    config, backbone = raw.config, raw.config.backbone_config
    print(f"checkpoint {args.model_name}: {config.num_queries} queries x {config.group_detr} groups in training, "
          f"{config.decoder_layers} decoder layers of width {config.d_model}, {config.decoder_n_points} sampling points; "
          f"backbone {backbone.num_hidden_layers} layers of width {backbone.hidden_size}, {backbone.num_attention_heads} "
          f"heads, patch {backbone.patch_size}, {backbone.num_windows} x {backbone.num_windows} windows, features after "
          f"stages {backbone.out_indices}; {len(config.id2label)} labels; processor {size} x {processor.size.width} px")
    multiple = backbone.patch_size * backbone.num_windows
    ok &= check(f"the input side is a multiple of {multiple}", size % multiple == 0, f"{size} px")

    start_model = build_start_model(args.model_name, load_model).eval()
    print(format_param_table(param_table(start_model)))
    ok &= check("head parameters that inference uses", inference_head_params(start_model) == EXPECTED_INFERENCE_HEADS,
                f"{inference_head_params(start_model):,}")

    # A change to q_proj of a backbone layer: layers that the detector reads change the logits, later ones do not.
    pixel_values = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(0))
    used = used_backbone_layers(start_model)
    with torch.no_grad():
        reference = start_model(pixel_values=pixel_values)
        changes = []
        for i in range(backbone.num_hidden_layers):
            weight = start_model.model.backbone.backbone.encoder.layer[i].attention.q_proj.weight
            saved = weight.clone()
            weight.add_(0.05 * torch.randn(weight.shape, generator=torch.Generator().manual_seed(i)))
            changes.append((start_model(pixel_values=pixel_values).logits - reference.logits).abs().max().item())
            weight.copy_(saved)
    print("largest change of a logit after a change to q_proj of each backbone layer: "
          + ", ".join(f"{i}: {c:.2g}" for i, c in enumerate(changes)))
    ok &= check(f"backbone layers 0 to {used - 1} change the logits", all(c > 0 for c in changes[:used]))
    ok &= check(f"backbone layers {used} and later change no logit", all(c == 0 for c in changes[used:]),
                f"{backbone.num_hidden_layers - used} dead layer(s)")

    models = {}
    for arm in "HDBF":
        cfg = load_config(args.configs / f"arm_{arm.lower()}.yaml")
        model = copy.deepcopy(start_model)
        torch.manual_seed(0)
        wrapped = apply_arm(model, arm, cfg.lora)
        table = param_table(model)["total"]
        ok &= check(f"arm {arm} trainable parameters", table["trainable"] == EXPECTED_TRAINABLE[arm],
                    f"{table['trainable']:,} (expected {EXPECTED_TRAINABLE[arm]:,}); {len(wrapped)} adapted projections")
        if wrapped:
            ok &= check(f"arm {arm} adapter parameters", table["adapter"] == EXPECTED_ADAPTER, f"{table['adapter']:,}")
        with torch.no_grad():
            out = model.eval()(pixel_values=pixel_values)
            same = torch.equal(out.logits, reference.logits) and torch.equal(out.pred_boxes, reference.pred_boxes)
            ok &= check(f"arm {arm} gives the output of the start model at step 0", same,
                        f"largest difference {(out.logits - reference.logits).abs().max().item():.1e}")
            if wrapped:
                trained = copy.deepcopy(model)
                for module in trained.modules():
                    if isinstance(module, LoRALinear):
                        module.lora_B.weight.normal_(std=0.02, generator=torch.Generator().manual_seed(1))
                changed = trained(pixel_values=pixel_values)
                moved = (changed.logits - reference.logits).abs().max().item()
                ok &= check(f"arm {arm} adapters change the output when B is not zero", moved > 0, f"{moved:.2g}")
                diff = merge_difference(trained, pixel_values)
                ok &= check(f"arm {arm} merge difference is below 1e-4 (float64)", diff < 1e-4, f"{diff:.1e}")
        models[arm] = model

    if args.time:
        print(f"\ntraining speed on {device} (batch {args.batch_size}, {args.steps} steps, "
              f"{'fp32' if args.no_amp or device.type != 'cuda' else 'fp16 autocast'}, {size} px, 3 boxes per image):")
        print("| Arm | Images per second | Peak memory (GB) | Hours for one run (96,000 image passes) |\n|---|---|---|---|")
        for arm, model in models.items():
            try:
                speed, peak = time_arm(model, device, args.batch_size, size, args.steps, not args.no_amp)
                print(f"| {arm} | {speed:.1f} | {peak:.2f} | {IMAGES_PER_RUN / speed / 3600:.2f} |", flush=True)
            except torch.OutOfMemoryError:
                print(f"| {arm} | out of memory at batch {args.batch_size}: use a smaller batch and more accumulation steps | | |")
            model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.image_dir:
        raw.to(device)
        paths = sorted(args.image_dir.glob("*.jpg"))[:20]
        tiles, captions = [], []
        for path in paths:
            with Image.open(path) as im:
                image = im.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = raw(**inputs)
            det = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[image.size[::-1]])[0]
            labels = [f"{config.id2label[int(l)]} {s:.2f}" for l, s in zip(det["labels"], det["scores"])]
            tiles.append(draw_boxes(image, det["boxes"].tolist(), labels, colour=(42, 120, 214)))
            captions.append(f"{path.name}: {len(labels)} detections")
        contact_sheet(tiles, captions, columns=4, tile_size=(480, 360), out_path=args.sheet)
        print(f"\nCOCO detections of the checkpoint on {len(paths)} images: {args.sheet}")

    print("\nstage 0: " + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
