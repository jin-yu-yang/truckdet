import numpy as np
import torch

from tests.tiny import TINY_SIZE, tiny_processor
from truckdet.data.dataset import EpochSampler, MioTcdDetection, collate, labels_on
from truckdet.data.miotcd import load_annotations


def dataset(data_root):
    coco = load_annotations(data_root / "annotations.json")
    ids = [im["id"] for im in coco["images"]]
    return MioTcdDetection(coco, data_root / "train", ids, tiny_processor()), coco


def test_item_has_the_format_of_the_librarys_loss(data_root):
    ds, _ = dataset(data_root)
    pixels, target = ds[0]  # image 0: 80 x 80, class 0 at (8, 10, 30, 24) and motorized_vehicle at (50, 44, 14, 12)
    assert pixels.shape == (3, TINY_SIZE, TINY_SIZE) and pixels.dtype == torch.float32
    assert target["class_labels"].tolist() == [0, 5] and target["class_labels"].dtype == torch.int64
    expected = torch.tensor([[(8 + 15) / 80, (10 + 12) / 80, 30 / 80, 24 / 80], [(50 + 7) / 80, (44 + 6) / 80, 14 / 80, 12 / 80]])
    assert torch.allclose(target["boxes"], expected, atol=1e-6)
    assert target["image_id"] == 0 and target["orig_size"].tolist() == [80, 80]


def test_flip_mirrors_pixels_and_boxes(data_root):
    ds, _ = dataset(data_root)
    plain, target = ds[(1, False)]  # image 1 is 96 x 72: normalised boxes do not depend on the aspect ratio
    flipped, mirrored = ds[(1, True)]
    assert torch.allclose(flipped, plain.flip(-1), atol=1e-4)
    assert torch.allclose(mirrored["boxes"][:, 0], 1 - target["boxes"][:, 0], atol=1e-6)
    assert torch.allclose(mirrored["boxes"][:, 1:], target["boxes"][:, 1:], atol=1e-6)
    assert mirrored["class_labels"].tolist() == target["class_labels"].tolist()


def test_image_without_boxes(data_root):
    coco = load_annotations(data_root / "annotations.json")
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != 2]
    ds = MioTcdDetection(coco, data_root / "train", [2], tiny_processor())
    _, target = ds[0]
    assert target["class_labels"].shape == (0,) and target["boxes"].shape == (0, 4)


def test_sampler_is_a_function_of_seed_and_epoch():
    a, b = EpochSampler(50, seed=0), EpochSampler(50, seed=0)
    assert list(a) == list(b)
    assert sorted(i for i, _ in a) == list(range(50))  # every image one time
    first = list(a)
    a.set_epoch(1)
    assert list(a) != first
    b.set_epoch(1)
    assert list(a) == list(b)  # epoch 1 does not depend on having run epoch 0: a resumed run gets the same epoch
    assert list(EpochSampler(50, seed=1)) != first
    flips = [f for _, f in EpochSampler(2000, seed=0)]
    assert 0.4 < np.mean(flips) < 0.6
    assert not any(f for _, f in EpochSampler(50, seed=0, flip=False))


def test_loader_with_the_sampler(data_root):
    ds, _ = dataset(data_root)
    sampler = EpochSampler(len(ds), seed=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate)
    pixels, targets = next(iter(loader))
    assert pixels.shape == (4, 3, TINY_SIZE, TINY_SIZE) and len(targets) == 4
    order, _ = sampler.plan(0)
    assert [t["image_id"] for t in targets] == order[:4].tolist()
    labels = labels_on(targets, "cpu")
    assert set(labels[0]) == {"class_labels", "boxes"}
