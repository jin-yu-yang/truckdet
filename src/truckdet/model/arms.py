"""The start model and the four arms.

Pipeline: pretrained RF-DETR -> new 11-class heads, started from COCO rows (the *start model*, the
same for every arm and seed) -> freeze by name -> add the adapters of the arm.

    H  heads only
    D  heads, and LoRA in the decoder
    B  heads, and LoRA in the backbone
    F  all parameters

In arms H, D and B the projector, the queries and the proposal layers stay frozen, so the adapter
position is the only difference between D and B.
"""

import math

import torch
from torch import nn
from transformers import RfDetrForObjectDetection

from ..config import LORA_ARMS, LoRAConfig
from ..data.miotcd import CLASSES
from .lora import inject_lora

# Parameter-name prefixes of each part of the model (design §5.1). Every parameter is in exactly one part.
PARTS: dict[str, tuple[str, ...]] = {
    "backbone": ("model.backbone.backbone.",),
    "projector": ("model.backbone.projector.",),
    "decoder": ("model.decoder.",),
    "queries and proposals": (
        "model.query_feat.", "model.reference_point_embed.", "model.enc_output.", "model.enc_output_norm.",
    ),
    "heads": ("class_embed.", "bbox_embed.", "model.enc_out_class_embed.", "model.enc_out_bbox_embed."),
}

# MIO-TCD class -> the COCO class whose row starts it (design §5.4). None: no COCO class is near it.
COCO_SOURCE: dict[str, str | None] = {
    "articulated_truck": "truck",
    "pickup_truck": "truck",
    "single_unit_truck": "truck",
    "car": "car",
    "work_van": "car",
    "motorized_vehicle": "car",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "pedestrian": "person",
    "non-motorized_vehicle": None,
}

# The library starts `class_embed` with a bias that gives every class a probability of 0.01.
PRIOR_PROBABILITY = 0.01
# Seed of the rows that have no COCO source. It is fixed, so the start model does not depend on the run's seed.
HEAD_SEED = 0


def load_pretrained(model_name: str) -> RfDetrForObjectDetection:
    return RfDetrForObjectDetection.from_pretrained(model_name)


def part_of(name: str) -> str:
    for part, prefixes in PARTS.items():
        if name.startswith(prefixes):
            return part
    raise KeyError(f"parameter {name!r} is in no known part of the model; update PARTS")


def is_head(name: str) -> bool:
    return name.startswith(PARTS["heads"])


def is_adapter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def class_head_layers(model: RfDetrForObjectDetection) -> list[tuple[nn.Module, str]]:
    """(parent module, attribute) of every class layer: `class_embed` and each proposal copy."""
    proposals = model.model.enc_out_class_embed
    return [(model, "class_embed")] + [(proposals, str(i)) for i in range(len(proposals))]


def replace_class_heads(
    model: RfDetrForObjectDetection,
    class_names: tuple[str, ...] = CLASSES,
    source: dict[str, str | None] = COCO_SOURCE,
) -> None:
    """Swap every COCO class layer for one with a row per class in `class_names`.

    A row with a COCO source is a copy of that COCO row of the *same* layer (weights and bias), found
    by class name through `config.id2label`. A row without one gets the library's start for a class
    row: weights from N(0, init_std) and the prior-probability bias. The library gives that bias to
    `class_embed` only and leaves the proposal copies at 0; a bias of 0 is a score of 0.5 at every
    position, which would decide the top-300 proposals by itself, so all layers get the prior bias.
    The box heads keep their weights."""
    coco_id = {name: i for i, name in model.config.id2label.items()}
    missing = sorted({s for s in source.values() if s is not None and s not in coco_id})
    if missing:
        raise KeyError(f"the checkpoint has no class named {missing}; its labels are {sorted(coco_id)[:10]}...")
    unknown = sorted(set(class_names) - set(source))
    if unknown:
        raise KeyError(f"no COCO source is defined for {unknown}")
    generator = torch.Generator().manual_seed(HEAD_SEED)
    prior_bias = -math.log((1 - PRIOR_PROBABILITY) / PRIOR_PROBABILITY)
    for parent, attribute in class_head_layers(model):
        old = getattr(parent, attribute)
        new = nn.Linear(old.in_features, len(class_names), device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            start = torch.empty(new.weight.shape).normal_(0.0, model.config.init_std, generator=generator)
            new.weight.copy_(start)
            new.bias.fill_(prior_bias)
            for row, name in enumerate(class_names):
                if source[name] is not None:
                    new.weight[row] = old.weight[coco_id[source[name]]]
                    new.bias[row] = old.bias[coco_id[source[name]]]
        setattr(parent, attribute, new)
    model.config.id2label = dict(enumerate(class_names))
    model.config.label2id = {name: i for i, name in enumerate(class_names)}
    if model.config.num_labels != len(class_names):
        raise RuntimeError(f"config.num_labels is {model.config.num_labels}, expected {len(class_names)}")


def build_start_model(model_name: str, loader=load_pretrained) -> RfDetrForObjectDetection:
    """The model that every arm starts from: the checkpoint with the 11-class heads of design §5.4."""
    model = loader(model_name)
    replace_class_heads(model)
    return model


def used_backbone_layers(model: RfDetrForObjectDetection) -> int:
    """How many backbone layers the detector reads. `out_indices` count the patch embedding as stage 0, so
    the highest index n is the output of layer n - 1: layers 0 .. n - 1 are used, and a later layer is
    calculated and thrown away (layer 11 of the base model)."""
    return max(model.config.backbone_config.out_indices)


def _backbone_filter(model: RfDetrForObjectDetection):
    prefixes = tuple(
        f"model.backbone.backbone.encoder.layer.{i}.attention." for i in range(used_backbone_layers(model))
    )
    return lambda name: name.startswith(prefixes)


def lora_targets(model: RfDetrForObjectDetection, arm: str) -> dict:
    """Which projections arm `arm` adapts, as keyword arguments of `inject_lora` (design §5.3).

    B: query and value projections of the backbone layers that the detector reads.
    D: query and value projections of decoder self-attention, and `value_proj` of the deformable
       cross-attention. Deformable attention has no query-key product; `value_proj` is its value
       projection, so it takes the place of `v_proj`."""
    if arm == "B":
        return {"target_names": ("q_proj", "v_proj"), "parent_names": ("attention",),
                "layer_filter": _backbone_filter(model)}
    if arm == "D":
        return {"target_names": ("q_proj", "v_proj", "value_proj"), "parent_names": ("self_attn", "cross_attn"),
                "layer_filter": lambda name: name.startswith("model.decoder.layers.")}
    raise ValueError(f"arm {arm!r} has no adapters; LoRA arms are {LORA_ARMS}")


def apply_arm(model: RfDetrForObjectDetection, arm: str, lora: LoRAConfig | None = None) -> list[str]:
    """Freeze by name and add the arm's adapters, in place. Returns the names of the adapted layers.

    The adapters' A matrices are drawn from torch's global generator: seed it first.
    (`model.freeze_backbone()` raises AttributeError in transformers 5.20.1, so nothing here uses it.)"""
    if arm in LORA_ARMS and lora is None:
        raise ValueError(f"arm {arm} needs a LoRA configuration")
    for name, p in model.named_parameters():
        p.requires_grad_(arm == "F" or is_head(name))
    if arm not in LORA_ARMS:
        return []
    wrapped = inject_lora(model, r=lora.r, alpha=lora.alpha, dropout=lora.dropout, **lora_targets(model, arm))
    if not wrapped:  # a silent no-op would train a heads-only model under a LoRA label
        raise ValueError(f"arm {arm}: no projection matched the LoRA targets")
    return wrapped


def param_table(model: nn.Module) -> dict[str, dict[str, int]]:
    """Per part and in total: `total` (without adapters), `trainable` (with adapters) and `adapter` parameters."""
    table = {part: {"total": 0, "trainable": 0, "adapter": 0} for part in PARTS}
    for name, p in model.named_parameters():
        row = table[part_of(name)]
        if is_adapter(name):
            row["adapter"] += p.numel()
        else:
            row["total"] += p.numel()
        if p.requires_grad:
            row["trainable"] += p.numel()
    table["total"] = {key: sum(row[key] for row in table.values()) for key in ("total", "trainable", "adapter")}
    return table


def format_param_table(table: dict[str, dict[str, int]]) -> str:
    lines = ["| Part | Parameters | Trainable | Adapter |", "|---|---|---|---|"]
    for part, row in table.items():
        name = "**Total**" if part == "total" else part.capitalize()
        lines.append(f"| {name} | {row['total']:,} | {row['trainable']:,} | {row['adapter']:,} |")
    return "\n".join(lines)


def inference_head_params(model: RfDetrForObjectDetection) -> int:
    """Head parameters that inference uses: `class_embed`, `bbox_embed` and copy 0 of the proposal heads."""
    modules = [model.class_embed, model.bbox_embed, model.model.enc_out_class_embed[0], model.model.enc_out_bbox_embed[0]]
    return sum(p.numel() for m in modules for p in m.parameters())


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only what training changed: the heads and LoRA A/B for H, D and B; everything for F."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise KeyError(f"checkpoint mismatch; missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
