"""Run artifacts: the metrics log (jsonl), JSON files, and the loss curves."""

import json
from pathlib import Path

from .plotting import loss_curves


def append_jsonl(path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_jsonl(path, records: list[dict]) -> None:
    """Atomically replace a jsonl file with `records`."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(path)


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value) -> None:
    """Atomically write `value` as JSON with sorted keys, so a second run gives the same bytes."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def plot_curves(metrics_path, out_png) -> None:
    loss_curves(read_jsonl(metrics_path), out_png)
