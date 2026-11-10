"""CPU-fast stand-ins: a randomly initialized RF-DETR with the module structure of the base model, an
image processor with a small size, and a synthetic MIO-TCD folder."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import RfDetrConfig, RfDetrDinov2Config, RfDetrForObjectDetection, RfDetrImageProcessor

from truckdet.data.miotcd import CLASSES

# Enough of COCO's label map for the class-head start of design §5.4, with gaps as in the checkpoint.
TINY_LABELS = {0: "N/A", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
               7: "train", 8: "truck", 9: "boat"}
TINY_SIZE = 56  # patch 14 x 2 windows: the input side must be a multiple of 28


def make_tiny_rfdetr(dropout: float = 0.0) -> RfDetrForObjectDetection:
    """4 backbone layers, of which the detector reads the first 3 (the last one is dead, like layer 11 of the
    base model); 2 decoder layers; 3 query groups of 12 queries; COCO-style labels."""
    torch.manual_seed(0)
    backbone = RfDetrDinov2Config(
        hidden_size=24, num_hidden_layers=4, num_attention_heads=2, mlp_ratio=2, patch_size=14,
        image_size=TINY_SIZE, num_windows=2, window_block_indexes=[0, 2],
        out_features=["stage1", "stage2", "stage3"], num_register_tokens=0,
    )
    config = RfDetrConfig(
        backbone_config=backbone, d_model=32, decoder_layers=2, decoder_self_attention_heads=2,
        decoder_cross_attention_heads=4, decoder_n_points=2, decoder_ffn_dim=48, num_queries=12, group_detr=3,
        dropout=dropout, c2f_num_blocks=1, id2label=dict(TINY_LABELS),
        label2id={name: i for i, name in TINY_LABELS.items()},
    )
    model = RfDetrForObjectDetection(config).eval()
    with torch.no_grad():  # distinct rows, so a test can tell which COCO row a class row came from
        for layer in [model.class_embed, *model.model.enc_out_class_embed]:
            layer.weight.normal_(std=0.5)
            layer.bias.normal_(mean=-2.0, std=0.5)
    return model


def tiny_loader(model_name: str) -> RfDetrForObjectDetection:
    """Stand-in for `arms.load_pretrained`: ignores the name."""
    return make_tiny_rfdetr()


def tiny_processor(model_name: str = "") -> RfDetrImageProcessor:
    return RfDetrImageProcessor(size={"height": TINY_SIZE, "width": TINY_SIZE})


COLOURS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
           (255, 128, 0), (128, 0, 255), (0, 128, 128), (128, 128, 0), (200, 200, 200)]


def write_synthetic_miotcd(root: Path, n_images: int = 24, scenes: int = 4, seed: int = 0) -> Path:
    """`root/gt_train.csv` and `root/train/*.jpg`: coloured rectangles (one colour per class) on a flat
    background whose colour is the image's "camera". Image k has class k mod 11 and, every third image, a
    `motorized_vehicle` box too. IDs have leading zeros."""
    rng = np.random.default_rng(seed)
    (root / "train").mkdir(parents=True, exist_ok=True)
    backgrounds = rng.integers(40, 215, (scenes, 3))
    rows = []
    for k in range(n_images):
        width, height = (96, 72) if k % 2 else (80, 80)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[:] = backgrounds[k % scenes]
        pixels = np.clip(pixels + rng.integers(-6, 7, pixels.shape), 0, 255).astype(np.uint8)
        image_id = f"{k:08d}"
        boxes = [(k % len(CLASSES), 8 + k % 5, 10, 30, 24)]
        if k % 3 == 0:
            boxes.append((CLASSES.index("motorized_vehicle"), 50, 44, 14, 12))
        for label, x, y, w, h in boxes:
            pixels[y : y + h, x : x + w] = COLOURS[label]
            rows.append([image_id, CLASSES[label], x, y, x + w, y + h])
        Image.fromarray(pixels).save(root / "train" / f"{image_id}.jpg", quality=95)
    with open(root / "gt_train.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return root
