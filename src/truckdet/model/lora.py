"""Low-Rank Adaptation (LoRA), written from scratch. From the truck project (`detr-lora`), with one
change: `inject_lora` takes a layer filter.

Fine-tuning a frozen linear layer W (out x in) means learning an update dW. LoRA's bet is
that the useful update has low rank, so it parameterizes dW = B @ A with A: (r x in) and
B: (out x r), r << in. For a 384x384 projection with r=9 that is 6,912 trainable numbers
instead of 147,456.

    y = W x + b + (alpha / r) * B(A(x))

- B starts at zero, so at step 0 the layer is exactly the pretrained layer. Training
  starts from the pretrained model's behavior rather than from a random perturbation.
- alpha / r scales the update so that changing r does not change the effective learning
  rate much.
- After training, W' = W + (alpha / r) * B @ A folds the adapter into a plain nn.Linear
  ("merging"). The merged model has the original architecture and costs nothing extra
  at inference.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, r, bias=False, **factory)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return a plain nn.Linear with W' = W + (alpha/r) * B @ A (frozen)."""
        base = self.base
        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        merged.weight.copy_(base.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight))
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        for p in merged.parameters():
            p.requires_grad_(False)
        return merged


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float = 0.0,
    target_names: tuple[str, ...] = ("q_proj", "v_proj"),
    parent_names: tuple[str, ...] = ("self_attn", "encoder_attn"),
    layer_filter: Callable[[str], bool] | None = None,
) -> list[str]:
    """Wrap every nn.Linear named in `target_names` whose parent module is named in
    `parent_names` (e.g. 'model.decoder.layers.2.self_attn.q_proj').

    `layer_filter` gets the full module name and returns False for a layer to leave alone. The
    selection is by name only, so without a filter it would also take a layer whose output the
    model never uses (RF-DETR's last backbone layer), where an adapter is dead weight."""
    wrapped = []
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if (
            isinstance(module, nn.Linear)
            and len(parts) >= 2
            and parts[-1] in target_names
            and parts[-2] in parent_names
            and (layer_filter is None or layer_filter(name))
        ):
            _set_submodule(model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            wrapped.append(name)
    return wrapped


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear with its merged nn.Linear, in place."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        _set_submodule(model, name, model.get_submodule(name).merged_linear())
    return len(names)


def count_params(model: nn.Module) -> dict[str, int]:
    params = list(model.named_parameters())
    return {
        "total": sum(p.numel() for _, p in params),
        "trainable": sum(p.numel() for _, p in params if p.requires_grad),
        "lora": sum(p.numel() for n, p in params if "lora_" in n),
    }
