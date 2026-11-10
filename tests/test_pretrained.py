"""The facts of design §5.1 to §5.3 on the real checkpoint. Needs the network (about 130 MB the first time):

    .venv/bin/python -m pytest -m slow
"""

import copy

import pytest
import torch
from transformers import AutoImageProcessor

from truckdet.config import load_config
from truckdet.evaluation import merge_difference
from truckdet.model import arms
from truckdet.model.lora import LoRALinear

pytestmark = pytest.mark.slow
MODEL = "Roboflow/rf-detr-base"
CONFIGS = __import__("pathlib").Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def start():
    return arms.build_start_model(MODEL, arms.load_pretrained).eval()


@pytest.fixture(scope="module")
def pixels():
    return torch.randn(1, 3, 560, 560, generator=torch.Generator().manual_seed(0))


def arm_model(start, arm):
    model = copy.deepcopy(start)
    torch.manual_seed(0)
    names = arms.apply_arm(model, arm, load_config(CONFIGS / f"arm_{arm.lower()}.yaml").lora)
    return model, names


def test_parameter_table(start):
    table = arms.param_table(start)
    assert {part: row["total"] for part, row in table.items()} == {
        "backbone": 22_056_576, "projector": 1_444_864,
        "decoder": 4_613_152,  # the design measured 4,687,168 on the library default (4 sampling points; the checkpoint has 2)
        "queries and proposals": 1_875_952, "heads": 1_896_146, "total": 31_886_690,
    }
    assert arms.inference_head_params(start) == 270_878
    assert start.config.num_queries == 300 and start.config.group_detr == 13 and start.config.decoder_n_points == 2


def test_arms_have_the_trainable_parameters_of_the_design(start):
    expected = {"H": (1_896_146, 0, 0), "D": (2_048_210, 152_064, 9), "B": (2_048_210, 152_064, 22), "F": (31_886_690, 0, 0)}
    for arm, (trainable, adapter, layers) in expected.items():
        model, names = arm_model(start, arm)
        total = arms.param_table(model)["total"]
        assert (total["trainable"], total["adapter"], len(names)) == (trainable, adapter, layers), arm
    _, backbone = arm_model(start, "B")
    assert {int(n.split(".")[5]) for n in backbone} == set(range(11))  # layers 0 to 10: layer 11 is skipped


def test_backbone_layer_11_changes_no_logit(start, pixels):
    assert arms.used_backbone_layers(start) == 11
    with torch.no_grad():
        reference = start(pixel_values=pixels).logits
        for i, live in ((0, True), (10, True), (11, False)):
            weight = start.model.backbone.backbone.encoder.layer[i].attention.q_proj.weight
            saved = weight.clone()
            weight.add_(0.05 * torch.randn(weight.shape, generator=torch.Generator().manual_seed(i)))
            change = (start(pixel_values=pixels).logits - reference).abs().max().item()
            weight.copy_(saved)
            assert (change > 0) is live, f"layer {i}: {change}"


def test_step_zero_and_merge(start, pixels):
    with torch.no_grad():
        reference = start(pixel_values=pixels)
    for arm in "DB":
        model, names = arm_model(start, arm)
        with torch.no_grad():
            out = model.eval()(pixel_values=pixels)
            assert torch.allclose(out.logits, reference.logits, atol=1e-5) and torch.allclose(out.pred_boxes, reference.pred_boxes, atol=1e-6)
            for name in names:
                module = model.get_submodule(name)
                assert isinstance(module, LoRALinear)
                module.lora_B.weight.normal_(std=0.02)
            assert (model(pixel_values=pixels).logits - reference.logits).abs().max() > 1e-3
        assert merge_difference(model, pixels) < 1e-4


def test_processor_and_class_rows(start):
    processor = AutoImageProcessor.from_pretrained(MODEL)
    assert (processor.size.height, processor.size.width) == (560, 560) and 560 % 56 == 0
    raw = arms.load_pretrained(MODEL)
    coco = {name: i for i, name in raw.config.id2label.items()}
    assert all(name in coco for name in ("truck", "car", "bus", "bicycle", "motorcycle", "person"))
    row = lambda name: start.config.label2id[name]  # noqa: E731
    assert torch.equal(start.class_embed.weight[row("pickup_truck")], raw.class_embed.weight[coco["truck"]])
    assert torch.equal(start.class_embed.weight[row("work_van")], raw.class_embed.weight[coco["car"]])
    assert torch.equal(start.model.enc_out_class_embed[7].bias[row("pedestrian")], raw.model.enc_out_class_embed[7].bias[coco["person"]])
