"""Run folders: evaluate a finished run, find runs by name, choose learning rates, the gate.

A run folder `<runs>/<run name>/` holds `final.pt`, `metrics.jsonl`, `params.json`, and after
evaluation `pred_<split>.npz` (the saved predictions) and `eval_<split>.json`. The run name gives
the arm, the learning rate(s) and the seed (`config.make_run_name`).

`<runs>/selection.json` records the learning rate chosen for each arm on the val split with seed 0.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import stats
from .checkpoint import _load, load_checkpoint
from .config import ARMS, Config, config_from_dict, make_run_name, parse_run_name
from .data.dataset import MioTcdDetection, collate
from .data.miotcd import SPECIFIC_CLASSES, TRUCK_FAMILY, load_annotations, split_image_ids, subset
from .errors import box_table, error_split
from .evaluation import (
    load_predictions, merge_check, predict, primary_matches, metric_vector, save_predictions, score_predictions,
)
from .model.arms import build_start_model
from .reporting import write_json
from .training import ANNOTATIONS, IMAGE_DIR, build_model, resolve_loaders

SPLITS = ("val", "test")
SEEDS = (0, 1, 2)
SELECTION = "selection.json"
GATE = "gate.json"


def split_ground_truth(data_root, splits_dir, split: str, coco: dict | None = None) -> dict:
    """The plain COCO ground truth of a split (no ignore regions)."""
    coco = coco if coco is not None else load_annotations(Path(data_root) / ANNOTATIONS)
    return subset(coco, split_image_ids(coco, Path(splits_dir) / f"{split}.txt"))


def load_run_model(run_dir, device, model_loader=None):
    """(model with the run's final weights, the checkpoint's meta). Adapters are not merged."""
    final = Path(run_dir) / "final.pt"
    if not final.exists():
        raise FileNotFoundError(f"{final} does not exist: the run is not finished")
    cfg = config_from_dict(_load(final)["meta"]["config"])
    model = build_model(cfg, model_loader)
    state = load_checkpoint(final, model)
    return model.to(device).eval(), state["meta"]


def _error_numbers(gt: dict, predictions: dict) -> dict:
    table = box_table(gt, predictions)
    out = {}
    for label, names in (("specific classes", SPECIFIC_CLASSES), ("truck family", TRUCK_FAMILY)):
        share, accuracy = error_split(table, names)
        out[label] = {"localized share": share, "label accuracy": accuracy}
    return out


def _predict_split(model, gt: dict, data_root, processor, device, batch_size: int, num_workers: int):
    """(predictions, merge difference). The adapters are merged into the model first, as at deployment."""
    ids = sorted(im["id"] for im in gt["images"])
    dataset = MioTcdDetection(gt, Path(data_root) / IMAGE_DIR, ids, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    first = collate([dataset[i] for i in range(min(2, len(dataset)))])[0].to(device)
    merge_diff = merge_check(model, first)
    return predict(model, loader, device), merge_diff


def evaluate_run(
    run_dir,
    data_root,
    splits_dir,
    split: str = "val",
    device: str | torch.device = "cuda",
    force: bool = False,
    model_loader=None,
    processor_loader=None,
    num_workers: int | None = None,
) -> dict:
    """Predict a split with the run's final checkpoint, save the predictions, score them, write
    `eval_<split>.json`, and return its content.

    Existing predictions are scored again and not predicted again. The test split is scored one time for
    each final checkpoint (design §3, decision 8): a second call for `test` is refused unless `force`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    run_dir, device = Path(run_dir), torch.device(device)
    model_loader, processor_loader = resolve_loaders(model_loader, processor_loader)
    pred_path, eval_path = run_dir / f"pred_{split}.npz", run_dir / f"eval_{split}.json"
    if split == "test" and eval_path.exists() and not force:
        raise RuntimeError(f"{eval_path} exists: the test split is scored one time for each final checkpoint. "
                           "Analysis reads the saved predictions; pass force=True only after a code error.")
    gt = split_ground_truth(data_root, splits_dir, split)
    if pred_path.exists() and not force:
        predictions, meta = load_predictions(pred_path)
    else:
        model, ckpt_meta = load_run_model(run_dir, device, model_loader)
        cfg = config_from_dict(ckpt_meta["config"])
        workers = cfg.train.num_workers if num_workers is None else num_workers
        predictions, merge_diff = _predict_split(model, gt, data_root, processor_loader(cfg.model_name), device,
                                                 cfg.train.eval_batch_size, workers)
        meta = {"run": run_dir.name, "arm": cfg.arm, "seed": cfg.train.seed, "split": split,
                "merge_max_abs_diff": merge_diff, "skipped_batches": ckpt_meta.get("skipped", 0)}
        save_predictions(pred_path, predictions, meta)
    result = {**score_predictions(gt, predictions), "errors": _error_numbers(gt, predictions), **meta}
    write_json(eval_path, result)
    return result


def evaluate_start_model(
    model_name: str,
    data_root,
    splits_dir,
    out_dir,
    split: str = "val",
    device: str | torch.device = "cuda",
    model_loader=None,
    processor_loader=None,
    max_images: int | None = None,
    num_workers: int = 2,
) -> dict:
    """Score the start model (the COCO checkpoint with the class rows of design §5.4, no training) on a split.
    The check of stage 2: AP50 of `car` and `bus` must be low and above zero; an AP near zero shows an error
    in the box format or the resize."""
    device = torch.device(device)
    model_loader, processor_loader = resolve_loaders(model_loader, processor_loader)
    gt = split_ground_truth(data_root, splits_dir, split)
    if max_images:
        gt = subset(gt, sorted(im["id"] for im in gt["images"])[:max_images])
    model = build_start_model(model_name, model_loader).to(device).eval()
    predictions, _ = _predict_split(model, gt, data_root, processor_loader(model_name), device, 16, num_workers)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(out_dir / f"pred_{split}.npz", predictions, {"run": "start_model", "split": split})
    result = score_predictions(gt, predictions)
    result["AP50 per class"] = ap50_per_class(gt, predictions)
    write_json(out_dir / f"eval_{split}.json", result)
    return result


def ap50_per_class(gt: dict, predictions: dict) -> dict[str, float]:
    from .evaluation import ap_table  # local: only this check needs the table itself

    matches = primary_matches(gt, predictions)
    return {c: float(v) for c, v in zip(matches.classes, ap_table(matches)[:, 0])}


# --- Finding runs --------------------------------------------------------------------------


def list_runs(runs_root) -> list[dict]:
    """Every run folder under `runs_root` whose name is a run name: {"name", "dir", "arm", "lr", "body_lr", "seed"}."""
    out = []
    for d in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        parsed = parse_run_name(d.name) if d.is_dir() else None
        if parsed:
            out.append({"name": d.name, "dir": d, **parsed})
    return out


def read_eval(run_dir, split: str) -> dict | None:
    path = Path(run_dir) / f"eval_{split}.json"
    return json.loads(path.read_text()) if path.exists() else None


def swept_lr(run: dict) -> float:
    """The rate that the learning-rate runs of the arm vary: the body rate for F, the one rate for the others."""
    return run["body_lr"] if run["arm"] == "F" else run["lr"]


def val_ap_by_lr(runs_root, arm: str, head_lr: float | None = None) -> dict[float, float]:
    """Swept rate -> val AP of the final checkpoint, for the evaluated seed-0 runs of `arm`. For arm F only the
    runs whose heads use `head_lr` count."""
    out = {}
    for run in list_runs(runs_root):
        if run["arm"] != arm or run["seed"] != 0:
            continue
        if arm == "F" and (head_lr is None or not np.isclose(run["lr"], head_lr, rtol=1e-6)):
            continue
        ev = read_eval(run["dir"], "val")
        if ev is not None:
            out[swept_lr(run)] = ev["AP"]
    return out


def read_selection(runs_root) -> dict:
    path = Path(runs_root) / SELECTION
    return json.loads(path.read_text()) if path.exists() else {}


def select_lr(runs_root, cfg: Config) -> dict:
    """Apply the rule of design §6 to the seed-0 runs of the arm of `cfg`. Returns a status:

    - {"status": "run", "lr": [...]}: these rates have no evaluated run yet (rates of the arm's list, or the one
      more rate when the best rate is at an end of the list);
    - {"status": "chosen", "lr": ..., ...}: the choice, also written to `selection.json`.
    For arm F the rates are body rates, and the head rate is the choice of arm H (which must exist)."""
    arm, grid = cfg.arm, list(cfg.train.lr_grid)
    selection = read_selection(runs_root)
    head_lr = None
    if arm == "F":
        if "H" not in selection:
            return {"status": "blocked", "reason": "arm F uses the rate that arm H chose; choose H first"}
        head_lr = selection["H"]["lr"]
    val_ap = val_ap_by_lr(runs_root, arm, head_lr)
    missing = [lr for lr in grid if not any(np.isclose(lr, done, rtol=1e-6) for done in val_ap)]
    if missing:
        return {"status": "run", "lr": missing, "val_ap": val_ap, "head_lr": head_lr}
    extra = stats.lr_to_add(grid, val_ap)
    if extra is not None:
        return {"status": "run", "lr": [extra], "val_ap": val_ap, "head_lr": head_lr,
                "reason": "the best rate is at an end of its list: one more step in that direction, one time"}
    chosen = stats.choose_lr(val_ap)
    entry = {"lr": head_lr if arm == "F" else chosen, "body_lr": chosen if arm == "F" else None,
             "val_ap": {f"{lr:g}": ap for lr, ap in sorted(val_ap.items())},
             "run": make_run_name(arm, head_lr if arm == "F" else chosen, 0, chosen if arm == "F" else None)}
    selection[arm] = entry
    write_json(Path(runs_root) / SELECTION, selection)
    return {"status": "chosen", **entry}


def selected_run_name(selection: dict, arm: str, seed: int) -> str:
    if arm not in selection:
        raise KeyError(f"no learning rate is chosen for arm {arm} yet (see scripts/select_lr.py)")
    entry = selection[arm]
    return make_run_name(arm, entry["lr"], seed, entry["body_lr"])


def final_runs(runs_root, seeds=SEEDS, arms=ARMS) -> dict[str, list[Path]]:
    """Arm -> the run folders of its chosen learning rate, one for each seed."""
    selection = read_selection(runs_root)
    return {arm: [Path(runs_root) / selected_run_name(selection, arm, seed) for seed in seeds] for arm in arms}


# --- The gate ------------------------------------------------------------------------------


def run_gate(runs_root, data_root, splits_dir, n_boot: int = stats.N_BOOT) -> dict:
    """The gate of design §2: AP_F - AP_H on the val split with seed 0, from the saved predictions of the
    chosen H and F runs. Writes `gate.json`."""
    selection = read_selection(runs_root)
    gt = split_ground_truth(data_root, splits_dir, "val")
    weights = stats.resample_weights(len(gt["images"]), n_boot)
    point, samples = {}, {}
    for arm in ("H", "F"):
        pred_path = Path(runs_root) / selected_run_name(selection, arm, 0) / "pred_val.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"{pred_path} does not exist: evaluate the run on the val split first")
        matches = primary_matches(gt, load_predictions(pred_path)[0])
        point[arm] = float(metric_vector(matches)[0])
        samples[arm] = stats.bootstrap_metrics(matches, weights)[:, 0]
    result = stats.gate(point["H"], point["F"], samples["H"], samples["F"])
    result.update(runs={arm: selected_run_name(selection, arm, 0) for arm in ("H", "F")}, n_boot=n_boot,
                  val_images=len(gt["images"]))
    write_json(Path(runs_root) / GATE, result)
    return result


def gate_text(result: dict) -> str:
    verdict = ("The gate passes: continue with the main runs." if result["passed"] else
               "The gate fails: stop. The data cannot separate the arms.")
    return (f"AP_F − AP_H on the val split (seed 0) = {result['difference']:+.3f} "
            f"(AP_H {result['AP_H']:.3f}, AP_F {result['AP_F']:.3f}); bootstrap standard error "
            f"{result['bootstrap_se']:.4f}; the difference is {result['ratio']:.1f} times its standard error. "
            f"Needed: at least {result['min_difference']:.2f} AP and at least {result['min_ratio']:.0f} times the "
            f"standard error. {verdict}")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
