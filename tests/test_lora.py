import copy

import pytest
import torch
from torch import nn

from truckdet.model.lora import LoRALinear, count_params, inject_lora, merge_lora


def randomize_lora(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.weight.normal_(std=0.1)


def test_zero_init_is_identity():
    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    lora = LoRALinear(copy.deepcopy(base), r=4, alpha=8)
    x = torch.randn(5, 16)
    assert torch.equal(lora(x), base(x))


def test_only_adapter_is_trainable():
    lora = LoRALinear(nn.Linear(16, 12), r=4, alpha=8)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A.weight", "lora_B.weight"}
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4 * (16 + 12)


def test_scaling_and_rank_validation():
    assert LoRALinear(nn.Linear(4, 4), r=8, alpha=16).scaling == 2.0
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)


def test_merged_linear_matches_wrapper():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 12), r=4, alpha=8, dropout=0.1).eval()
    randomize_lora(lora)
    x = torch.randn(5, 16)
    merged = lora.merged_linear()
    assert isinstance(merged, nn.Linear) and not isinstance(merged, LoRALinear)
    assert torch.allclose(merged(x), lora(x), atol=1e-5)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.encoder_attn, self.other = Attn(), Attn(), Attn()

    def forward(self, x):
        return self.other(self.encoder_attn(self.self_attn(x)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_inject_targets_only_matching_modules():
    model = Toy()
    names = inject_lora(model, r=2, alpha=4)
    assert len(names) == 2 * 2 * 2  # layers x (self_attn, encoder_attn) x (q, v)
    assert all(n.endswith(("q_proj", "v_proj")) for n in names)
    assert not any(".other." in n for n in names)
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[0].self_attn.k_proj, nn.Linear)
    assert isinstance(model.layers[0].other.q_proj, nn.Linear)


def test_layer_filter_skips_a_layer():
    """The one change to the truck project's code: a filter on the full module name."""
    model = Toy()
    names = inject_lora(model, r=2, alpha=4, layer_filter=lambda name: not name.startswith("layers.1."))
    assert len(names) == 4 and all(n.startswith("layers.0.") for n in names)
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[1].self_attn.q_proj, nn.Linear) and not isinstance(model.layers[1].self_attn.q_proj, LoRALinear)


def test_merge_removes_wrappers_and_preserves_outputs():
    torch.manual_seed(0)
    model = Toy()
    inject_lora(model, r=2, alpha=4)
    randomize_lora(model)
    model.eval()
    x = torch.randn(3, 8)
    before = model(x)
    assert merge_lora(model) == 8
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(model(x), before, atol=1e-5)


def test_count_params():
    model = Toy()
    total_before = count_params(model)["total"]
    for p in model.parameters():
        p.requires_grad_(False)
    inject_lora(model, r=2, alpha=4)
    counts = count_params(model)
    assert counts["lora"] == 8 * 2 * (8 + 8)
    assert counts["trainable"] == counts["lora"]
    assert counts["total"] == total_before + counts["lora"]
