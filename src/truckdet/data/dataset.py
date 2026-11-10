"""Dataset and collate function, with the library's image processor.

The processor of the checkpoint resizes every image to one square size (560 x 560 for the base
model) without keeping the aspect ratio, and normalises it. So a box in normalised coordinates is
the same box in the resized image and in the original image, and a prediction goes back to pixels
with one multiplication by the original width and height.

The one augmentation is a horizontal flip. The order of an epoch and its flips are a function of
(seed, epoch) only (`EpochSampler`): they do not depend on the number of workers, and a resumed
run continues with the epochs that an uninterrupted run would have had.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class EpochSampler(Sampler):
    """Yields (index, flip) for every image, in the random order of epoch `epoch` of seed `seed`."""

    def __init__(self, size: int, seed: int, flip: bool = True):
        self.size, self.seed, self.flip = size, seed, flip
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def plan(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """(order, flips) of one epoch."""
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(self.size)
        flips = rng.random(self.size) < 0.5 if self.flip else np.zeros(self.size, dtype=bool)
        return order, flips

    def __iter__(self):
        order, flips = self.plan(self.epoch)
        return iter(zip(order.tolist(), flips.tolist()))

    def __len__(self) -> int:
        return self.size


class MioTcdDetection(Dataset):
    """Images of a COCO dict, in the order of `image_ids` (COCO ids).

    An item is (pixel_values [3, S, S], target). The target has `class_labels` (0-based) and `boxes`
    (normalised cx, cy, w, h), as the library's loss expects, and `image_id` and `orig_size` (h, w)."""

    def __init__(self, coco: dict, image_dir: str | Path, image_ids, processor):
        self.image_dir = Path(image_dir)
        self.processor = processor
        images = {im["id"]: im for im in coco["images"]}
        missing = [i for i in image_ids if i not in images]
        if missing:
            raise KeyError(f"{len(missing)} image ids are not in the annotations, e.g. {missing[:3]}")
        anns: dict[int, list[dict]] = {int(i): [] for i in image_ids}
        for a in coco["annotations"]:
            if a["image_id"] in anns:
                anns[a["image_id"]].append(a)
        self.items = [(images[int(i)], anns[int(i)]) for i in image_ids]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key):
        index, flip = key if isinstance(key, tuple) else (key, False)
        info, anns = self.items[index]
        with Image.open(self.image_dir / info["file_name"]) as im:
            image = im.convert("RGB")
        width, height = image.size
        boxes = [list(a["bbox"]) for a in anns]
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[width - x - w, y, w, h] for x, y, w, h in boxes]
        annotations = [{"bbox": box, "category_id": a["category_id"] - 1, "area": box[2] * box[3], "iscrowd": 0}
                       for box, a in zip(boxes, anns)]
        encoded = self.processor(images=image, annotations={"image_id": info["id"], "annotations": annotations},
                                 return_tensors="pt")
        labels = encoded["labels"][0]
        target = {
            "class_labels": labels["class_labels"].to(torch.int64).reshape(-1),
            "boxes": labels["boxes"].to(torch.float32).reshape(-1, 4),
            "image_id": int(info["id"]),
            "orig_size": torch.tensor([height, width]),
        }
        return encoded["pixel_values"][0], target


def collate(batch):
    """Every image has the processor's size, so the batch is a plain stack."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def labels_on(targets: list[dict], device) -> list[dict]:
    """The part of the targets that the library's loss reads, on `device`."""
    return [{"class_labels": t["class_labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]
