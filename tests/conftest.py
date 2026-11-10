import json

import pytest

from tests.tiny import tiny_loader, tiny_processor, write_synthetic_miotcd
from truckdet.config import Config, LoRAConfig, TrainConfig
from truckdet.data.miotcd import convert, write_ids

N_IMAGES = 24


@pytest.fixture(autouse=True)
def tiny_loaders(monkeypatch):
    """Every test gets the tiny model and processor in place of the checkpoint: no network."""
    monkeypatch.setattr("truckdet.training.load_pretrained", tiny_loader)
    monkeypatch.setattr("truckdet.training.load_processor", tiny_processor)


@pytest.fixture
def data_root(tmp_path):
    """A synthetic MIO-TCD folder with its converted annotations."""
    root = write_synthetic_miotcd(tmp_path / "data", n_images=N_IMAGES)
    coco, _ = convert(root / "gt_train.csv", root / "train")
    (root / "annotations.json").write_text(json.dumps(coco))
    return root


@pytest.fixture
def splits_dir(tmp_path):
    ids = [f"{k:08d}" for k in range(N_IMAGES)]
    folder = tmp_path / "splits"
    folder.mkdir()
    for name, part in (("train", ids[:12]), ("train_pool", ids[:12]), ("val", ids[12:18]), ("test", ids[18:])):
        write_ids(folder / f"{name}.txt", part)
    return folder


def tiny_config(arm: str, **train) -> Config:
    """An arm's config for the tiny model. Ranks 4 (backbone, 6 x 24-wide projections) and 3 (decoder, 6 x 32-wide)
    give both LoRA arms 1,152 adapter parameters, as the real ranks 9 and 33 give both 152,064."""
    lora = {"B": LoRAConfig(r=4, alpha=8.0), "D": LoRAConfig(r=3, alpha=6.0)}.get(arm)
    defaults = dict(epochs=2, batch_size=2, accum_steps=2, num_workers=0, amp=False, eval_batch_size=4)
    return Config(arm=arm, lora=lora, train=TrainConfig(**{**defaults, **train}))
