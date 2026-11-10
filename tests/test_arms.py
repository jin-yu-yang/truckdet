import copy

import pytest
import torch
from torch import nn

from tests.conftest import tiny_config
from tests.tiny import TINY_LABELS, TINY_SIZE, make_tiny_rfdetr, tiny_loader
from truckdet.data.miotcd import CLASSES
from truckdet.evaluation import merge_difference
from truckdet.model import arms
from truckdet.model.lora import LoRALinear


@pytest.fixture
def start():
    return arms.build_start_model("tiny", tiny_loader).eval()


@pytest.fixture
def pixels():
    return torch.randn(2, 3, TINY_SIZE, TINY_SIZE, generator=torch.Generator().manual_seed(0))


def make(start, arm):
    model = copy.deepcopy(start)
    torch.manual_seed(0)
    names = arms.apply_arm(model, arm, tiny_config(arm).lora)
    return model, names


def test_class_rows_come_from_the_coco_row_of_the_same_layer():
    old = make_tiny_rfdetr()
    new = copy.deepcopy(old)
    arms.replace_class_heads(new)
    coco = {name: i for i, name in TINY_LABELS.items()}
    pairs = list(zip(arms.class_head_layers(old), arms.class_head_layers(new)))
    assert len(pairs) == 1 + old.config.group_detr  # class_embed and one proposal copy for each query group
    for (old_parent, attribute), (new_parent, _) in pairs:
        before, after = getattr(old_parent, attribute), getattr(new_parent, attribute)
        assert after.weight.shape == (len(CLASSES), before.in_features)
        for row, name in enumerate(CLASSES):
            source = arms.COCO_SOURCE[name]
            if source is None:
                continue
            assert torch.equal(after.weight[row], before.weight[coco[source]]), (attribute, name)
            assert torch.equal(after.bias[row], before.bias[coco[source]])
    assert new.config.num_labels == 11 and new.config.id2label == dict(enumerate(CLASSES))
    for a, b in zip(old.bbox_embed.parameters(), new.bbox_embed.parameters()):  # the box heads keep their weights
        assert torch.equal(a, b)


def test_sources_follow_the_design():
    assert {c for c, s in arms.COCO_SOURCE.items() if s == "truck"} == {"articulated_truck", "pickup_truck", "single_unit_truck"}
    assert {c for c, s in arms.COCO_SOURCE.items() if s == "car"} == {"car", "work_van", "motorized_vehicle"}
    assert arms.COCO_SOURCE["pedestrian"] == "person" and arms.COCO_SOURCE["non-motorized_vehicle"] is None
    assert all(arms.COCO_SOURCE[c] == c for c in ("bus", "bicycle", "motorcycle"))
    assert set(arms.COCO_SOURCE) == set(CLASSES)


def test_row_without_a_source_gets_the_prior_bias_in_every_layer(start):
    row = CLASSES.index("non-motorized_vehicle")
    for parent, attribute in arms.class_head_layers(start):
        layer = getattr(parent, attribute)
        assert layer.bias[row].item() == pytest.approx(-4.59512, abs=1e-4)  # sigmoid = 0.01
        assert 0 < layer.weight[row].abs().max() < 0.2  # N(0, 0.02), not the COCO rows (std 0.5 in the tiny model)


def test_start_model_does_not_depend_on_the_global_seed():
    torch.manual_seed(1)
    a = arms.build_start_model("tiny", tiny_loader)
    torch.manual_seed(2)
    b = arms.build_start_model("tiny", tiny_loader)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_a_missing_coco_class_is_an_error():
    model = make_tiny_rfdetr()
    model.config.id2label = {i: f"LABEL_{i}" for i in range(10)}
    with pytest.raises(KeyError, match="truck"):
        arms.replace_class_heads(model)


def test_every_parameter_is_in_one_part(start):
    table = arms.param_table(start)
    assert table["total"]["total"] == sum(p.numel() for p in start.parameters())
    assert sum(table[part]["total"] for part in arms.PARTS) == table["total"]["total"]
    with pytest.raises(KeyError):
        arms.part_of("model.something_new.weight")


def test_trainable_parameters_of_each_arm(start):
    total = sum(p.numel() for p in start.parameters())
    heads = arms.param_table(start)["heads"]["total"]
    counts = {}
    for arm in "HDBF":
        model, names = make(start, arm)
        table = arms.param_table(model)
        counts[arm] = table["total"]
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        if arm == "F":
            assert table["total"]["trainable"] == total and not names
        else:
            # the projector, the queries and the proposal layers stay frozen: only heads and adapters train
            assert all(arms.is_head(n) or arms.is_adapter(n) for n in trainable)
            assert {n for n in trainable if arms.is_head(n)} == {n for n, _ in model.named_parameters() if arms.is_head(n)}
            assert table["total"]["trainable"] == heads + table["total"]["adapter"]
        assert table["total"]["total"] == total  # adapters are counted apart from the model's own parameters
    assert counts["H"]["adapter"] == 0 and counts["F"]["adapter"] == 0
    assert counts["B"]["adapter"] == counts["D"]["adapter"] == 1152  # the equal budget of the tiny configs
    assert counts["B"]["trainable"] == counts["D"]["trainable"] == heads + 1152


def test_adapter_positions(start):
    _, backbone = make(start, "B")
    _, decoder = make(start, "D")
    used = arms.used_backbone_layers(start)
    assert used == 3 and start.config.backbone_config.num_hidden_layers == 4
    assert backbone == [f"model.backbone.backbone.encoder.layer.{i}.attention.{p}" for i in range(used) for p in ("q_proj", "v_proj")]
    assert decoder == [f"model.decoder.layers.{i}.{p}" for i in range(2)
                       for p in ("self_attn.q_proj", "self_attn.v_proj", "cross_attn.value_proj")]


def test_the_last_backbone_layer_is_dead_and_gets_no_adapter(start, pixels):
    """Design §5.1: the detector does not read the last backbone layer, so a change there changes no logit."""
    with torch.no_grad():
        reference = start(pixel_values=pixels).logits
        changes = []
        for layer in start.model.backbone.backbone.encoder.layer:
            weight = layer.attention.q_proj.weight
            saved = weight.clone()
            weight.add_(torch.randn_like(weight))
            changes.append((start(pixel_values=pixels).logits - reference).abs().max().item())
            weight.copy_(saved)
    assert all(c > 0 for c in changes[:3]) and changes[3] == 0
    model, _ = make(start, "B")
    assert not isinstance(model.model.backbone.backbone.encoder.layer[3].attention.q_proj, LoRALinear)


def test_step_zero_output_is_the_start_model(start, pixels):
    with torch.no_grad():
        reference = start(pixel_values=pixels)
        outputs = {arm: make(start, arm)[0].eval()(pixel_values=pixels) for arm in "HDBF"}
    # Adapters add nothing at step 0: D and B are exactly H, and F is exactly the start model.
    for arm, other in (("D", "H"), ("B", "H")):
        assert torch.equal(outputs[arm].logits, outputs[other].logits)
        assert torch.equal(outputs[arm].pred_boxes, outputs[other].pred_boxes)
    assert torch.equal(outputs["F"].logits, reference.logits)
    # Freezing changes which float kernel torch uses for a linear layer, so H is the start model to rounding.
    assert torch.allclose(outputs["H"].logits, reference.logits, atol=1e-5)
    assert torch.allclose(outputs["H"].pred_boxes, reference.pred_boxes, atol=1e-6)


@pytest.mark.parametrize("arm", ["D", "B"])
def test_adapters_change_the_output_and_merge_exactly(start, pixels, arm):
    model, names = make(start, arm)
    model.eval()
    with torch.no_grad():
        before = model(pixel_values=pixels).logits
        for name in names:
            model.get_submodule(name).lora_B.weight.normal_(std=0.05)
        after = model(pixel_values=pixels).logits
    assert (after - before).abs().max() > 1e-3
    assert merge_difference(model, pixels) < 1e-9  # float64: the merge is exact
    assert any(isinstance(m, LoRALinear) for m in model.modules())  # merge_difference leaves the model as it was
    assert merge_difference(make(start, "H")[0], pixels) == 0.0


def test_features_need_no_gradient_in_arms_h_and_d(start, pixels):
    """Design §6, "Speed": no trainable parameter comes before the decoder in H and D, so autograd keeps no
    graph for the backbone and the projector. That is `torch.no_grad()` for that part, without the call."""
    for arm, expected in (("H", False), ("D", False), ("B", True), ("F", True)):
        model, _ = make(start, arm)
        out = model.train()(pixel_values=pixels)
        assert out.backbone_features.requires_grad is expected, arm


def test_trainable_state_dict_round_trip(start):
    model, _ = make(start, "B")
    state = arms.trainable_state_dict(model)
    assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(".base." in n or "backbone.backbone.embeddings" in n for n in state)
    other, _ = make(start, "B")
    with torch.no_grad():
        for p in other.parameters():
            if p.requires_grad:
                p.add_(1.0)
    arms.load_trainable_state_dict(other, state)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), other.parameters()))
    with pytest.raises(KeyError):
        arms.load_trainable_state_dict(make(start, "D")[0], state)
    assert isinstance(arms.format_param_table(arms.param_table(model)), str)
    assert isinstance(model.class_embed, nn.Linear)
