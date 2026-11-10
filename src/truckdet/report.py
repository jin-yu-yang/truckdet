"""The analysis of the main runs: tables, intervals and the three readings of design §2, as markdown.

Everything is calculated on a CPU from the saved predictions (`pred_<split>.npz` of each run) and
has no random part that is not seeded, so a second run gives the same text.

An arm's number is the mean of its seeds. An interval is the 95% range of 400 paired bootstrap
resamples of the scene groups of the split (design §7.3): in each resample all arms and seeds are
scored on the same images, and the seeds of an arm are averaged. The interval of AP_B - AP_D from
resamples of single images is shown next to it, one time.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import stats
from .config import ARMS
from .data.miotcd import CLASSES, IGNORE_UNDER, MOTORIZED_VEHICLE, SPECIFIC_CLASSES, TRUCK_FAMILY
from .errors import NOT_LOCALIZED_NAME, bootstrap_error_split, box_table, confusion_counts, error_split
from .evaluation import (
    check_same_images, load_predictions, metric_names, metric_vector, primary_matches, score_predictions,
)
from .plotting import recovered_share_figure
from .reporting import write_json
from .runs import GATE, SEEDS, file_sha256, final_runs, gate_text, read_selection, split_groups, split_ground_truth

# Bump when the analysis code changes what it calculates: the cache key covers files, not code.
ANALYSIS_VERSION = "2"
ERROR_SETS = {"specific classes": SPECIFIC_CLASSES, "truck family": TRUCK_FAMILY}
ARM_LABELS = {"H": "H: heads only", "D": "D: LoRA in the decoder", "B": "B: LoRA in the backbone",
              "F": "F: full fine-tuning"}

QUESTION = ("With the same number of adapter parameters, does LoRA recover more of the full fine-tuning gain "
            "in the backbone or in the decoder?")
PREDICTIONS = {
    "P1": "LoRA in the backbone is better than LoRA in the decoder.",
    "P2": "The advantage of the backbone is in the labels, not in the boxes.",
    "P3": "LoRA in the backbone recovers most of the full fine-tuning gain.",
}


@dataclass
class RunAnalysis:
    name: str
    metrics: dict  # evaluation.score_predictions
    vector: np.ndarray  # [M] evaluation.metric_vector
    samples: np.ndarray  # [n_boot, M] the same on each resample
    errors: dict[str, np.ndarray]  # error set -> [2] (localized share, label accuracy)
    error_samples: dict[str, np.ndarray]  # error set -> [n_boot, 2]
    confusion: dict[str, dict[str, int]]  # truck-family class -> label at the representative query -> boxes
    meta: dict
    image_ap_samples: np.ndarray | None = None  # [n_boot] AP on resamples of single images (arms B and D)


def analyse_run(pred_path, gt: dict, weights: np.ndarray, cache: bool = True,
                image_weights: np.ndarray | None = None) -> RunAnalysis:
    """Everything the report needs from one run's saved predictions. `weights`: the resamples of the scene
    groups. `image_weights`: resamples of single images; with them the AP is also calculated on those. With
    `cache`, the result is kept next to the predictions and reused while the prediction file, the images and
    the resamples are the same."""
    pred_path = Path(pred_path)
    cache_path = pred_path.with_name(pred_path.stem.replace("pred_", "analysis_") + ".npz")
    image_ids = sorted(im["id"] for im in gt["images"])
    key = "|".join([file_sha256(pred_path), ANALYSIS_VERSION, _digest(weights), _digest(np.array(image_ids)),
                    _digest(image_weights) if image_weights is not None else "no image resamples"])
    if cache and cache_path.exists():
        with np.load(cache_path) as z:
            if str(z["key"]) == key:
                extra = json.loads(str(z["extra"]))
                return RunAnalysis(
                    pred_path.parent.name, extra["metrics"], z["vector"], z["samples"],
                    {s: z[f"errors_{i}"] for i, s in enumerate(ERROR_SETS)},
                    {s: z[f"error_samples_{i}"] for i, s in enumerate(ERROR_SETS)}, extra["confusion"], extra["meta"],
                    z["image_ap_samples"] if "image_ap_samples" in z else None,
                )
    predictions, meta = load_predictions(pred_path)
    check_same_images(gt, predictions)
    primary = primary_matches(gt, predictions)
    table = box_table(gt, predictions)
    confusion = {c: dict(sorted(counts.items())) for c, counts in confusion_counts(table, TRUCK_FAMILY).items()}
    out = RunAnalysis(
        pred_path.parent.name, score_predictions(gt, predictions, primary), metric_vector(primary),
        stats.bootstrap_metrics(primary, weights),
        {s: np.array(error_split(table, names)) for s, names in ERROR_SETS.items()},
        {s: bootstrap_error_split(table, names, weights) for s, names in ERROR_SETS.items()}, confusion, meta,
        stats.bootstrap_metrics(primary, image_weights)[:, 0] if image_weights is not None else None,
    )
    if cache:
        arrays = {"key": np.array(key), "vector": out.vector, "samples": out.samples,
                  "extra": np.array(json.dumps({"metrics": out.metrics, "confusion": confusion, "meta": meta}))}
        for i, s in enumerate(ERROR_SETS):
            arrays[f"errors_{i}"], arrays[f"error_samples_{i}"] = out.errors[s], out.error_samples[s]
        if out.image_ap_samples is not None:
            arrays["image_ap_samples"] = out.image_ap_samples
        tmp = cache_path.with_suffix(".part")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        tmp.replace(cache_path)
    return out


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# --- Numbers -------------------------------------------------------------------------------


def _estimate(point: float, samples: np.ndarray) -> dict:
    low, high = stats.interval(samples) if np.isfinite(samples).any() else (float("nan"), float("nan"))
    return {"value": float(point), "low": low, "high": high}


def build_numbers(analyses: dict[str, list[RunAnalysis]]) -> dict:
    """All numbers of the report from the analyses of each arm's seeds (the arms H, D, B, F)."""
    names = metric_names()
    points = {arm: np.stack([r.vector for r in runs]) for arm, runs in analyses.items()}  # [S, M]
    samples = {arm: np.mean([r.samples for r in runs], axis=0) for arm, runs in analyses.items()}  # [n_boot, M]
    mean = {arm: p.mean(axis=0) for arm, p in points.items()}

    def difference(a: str, b: str, j: int = 0) -> dict:
        agree, total = stats.pairs_in_agreement(points[a][:, j], points[b][:, j])
        return {**_estimate(mean[a][j] - mean[b][j], samples[a][:, j] - samples[b][:, j]),
                "seed_pairs_in_agreement": agree, "seed_pairs": total}

    arms = {}
    for arm, runs in analyses.items():
        arms[arm] = {
            "runs": [r.name for r in runs],
            "metrics": {n: {**_estimate(mean[arm][j], samples[arm][:, j]), "by_seed": points[arm][:, j].tolist()}
                        for j, n in enumerate(names)},
            "other": {k: float(np.mean([r.metrics[k] for r in runs]))
                      for k in ("AP small", "AP 11 classes, no ignore regions", f"AP {MOTORIZED_VEHICLE}",
                                "box AP", "box AP50")},
            "merge_max_abs_diff": [r.meta.get("merge_max_abs_diff") for r in runs],
            "skipped_batches": [r.meta.get("skipped_batches") for r in runs],
            "scaler_skips": [r.meta.get("scaler_skips") for r in runs],
        }

    recovered = {}
    for arm in ("D", "B"):
        point = float(stats.recovered_share(mean[arm][0], mean["H"][0], mean["F"][0]))
        recovered[arm] = _estimate(point, stats.recovered_share(samples[arm][:, 0], samples["H"][:, 0], samples["F"][:, 0]))

    errors, error_samples = {}, {}
    for label in ERROR_SETS:
        errors[label], error_samples[label] = {}, {}
        for arm, runs in analyses.items():
            point = np.mean([r.errors[label] for r in runs], axis=0)
            boot = np.mean([r.error_samples[label] for r in runs], axis=0)
            error_samples[label][arm] = boot
            errors[label][arm] = {"localized share": _estimate(point[0], boot[:, 0]),
                                  "label accuracy": _estimate(point[1], boot[:, 1])}
        b, d = errors[label]["B"], errors[label]["D"]
        boot = error_samples[label]["B"] - error_samples[label]["D"]
        errors[label]["B - D"] = {
            "localized share": _estimate(b["localized share"]["value"] - d["localized share"]["value"], boot[:, 0]),
            "label accuracy": _estimate(b["label accuracy"]["value"] - d["label accuracy"]["value"], boot[:, 1]),
        }

    confusion = {}
    for arm, runs in analyses.items():
        confusion[arm] = {c: {} for c in TRUCK_FAMILY}
        for r in runs:
            for c, counts in r.confusion.items():
                for predicted, n in counts.items():
                    confusion[arm][c][predicted] = confusion[arm][c].get(predicted, 0) + n

    differences = {f"{a} - {b}": difference(a, b) for a, b in (("B", "D"), ("D", "H"), ("B", "H"), ("F", "H"), ("F", "B"))}
    family = errors["truck family"]["B - D"]
    p1 = differences["B - D"]
    readings = {
        "P1": stats.reading_p1((p1["low"], p1["high"])),
        "P2": stats.reading_p2((family["label accuracy"]["low"], family["label accuracy"]["high"]),
                               (family["localized share"]["low"], family["localized share"]["high"])),
        "P3": stats.reading_p3((recovered["B"]["low"], recovered["B"]["high"])),
    }
    # The same difference with resamples of single images, for one comparison in the report (design §7.3).
    image_bootstrap = None
    if all(r.image_ap_samples is not None for arm in ("B", "D") for r in analyses[arm]):
        by_image = {arm: np.mean([r.image_ap_samples for r in analyses[arm]], axis=0) for arm in ("B", "D")}
        image_bootstrap = {"B - D": _estimate(mean["B"][0] - mean["D"][0], by_image["B"] - by_image["D"])}
    return {"arms": arms, "differences": differences, "recovered_share": recovered, "errors": errors,
            "confusion": confusion, "readings": readings, "image_bootstrap": image_bootstrap}


# --- Text ----------------------------------------------------------------------------------


def _f(x: float | None, signed: bool = False) -> str:
    if x is None or x != x:
        return "–"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _ci(e: dict, signed: bool = False) -> str:
    if e["value"] != e["value"]:
        return "–"
    return f"{_f(e['value'], signed)} [{_f(e['low'], signed)}, {_f(e['high'], signed)}]"


def _limit(x: float | None) -> str:
    return "the chosen limit" if x is None else f"{x:.4f}"


def _pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.1f}%"


def _pct_ci(e: dict, signed: bool = False) -> str:
    if e["value"] != e["value"]:
        return "–"
    sign = "+" if signed else ""
    return f"{100 * e['value']:{sign}.1f}% [{100 * e['low']:{sign}.1f}, {100 * e['high']:{sign}.1f}]"


def render(numbers: dict, context: dict) -> str:
    """The report as markdown. `context`: images, groups (scene groups of the split), seeds, n_boot, split,
    trainable (arm -> parameters), selection, gate, figure (file name or None), samples (file names), and
    split_meta (`split_meta.json`: the limit and the caps of the scene groups)."""
    arms, readings = numbers["arms"], numbers["readings"]
    b_d = numbers["differences"]["B - D"]
    family = numbers["errors"]["truck family"]["B - D"]
    r_b, r_d = numbers["recovered_share"]["B"], numbers["recovered_share"]["D"]
    n_seeds = len(context["seeds"])
    split_meta = context.get("split_meta") or {}
    of_groups = f"the {context['groups']:,} scene groups" if context.get("groups") else "the scene groups"
    lines = [
        "# LoRA position in RF-DETR: backbone against decoder, on MIO-TCD — results", "",
        f"**Question.** {QUESTION}", "",
        "**Predictions and outcomes.** The predictions were written before any training run. An outcome is "
        "\"confirmed\", \"refuted\" or \"not shown\".", "",
        "| | Prediction | Outcome | Evidence (test split, 95% interval) |", "|---|---|---|---|",
        f"| P1 | {PREDICTIONS['P1']} | **{readings['P1']}** | AP_B − AP_D = {_ci(b_d, signed=True)} |",
        f"| P2 | {PREDICTIONS['P2']} | **{readings['P2']}** | Truck family, B − D: label accuracy "
        f"{_pct_ci(family['label accuracy'], signed=True)} points; localized share "
        f"{_pct_ci(family['localized share'], signed=True)} points |",
        f"| P3 | {PREDICTIONS['P3']} | **{readings['P3']}** | Recovered share of B = {_ci(r_b)}; the line is "
        f"{stats.P3_THRESHOLD:.3f} |", "",
        f"Recovered share of an arm X: (AP_X − AP_H) / (AP_F − AP_H). Decoder: {_ci(r_d)}. Backbone: {_ci(r_b)}.", "",
        f"*How to read the numbers.* {context['split'].capitalize()} split, {context['images']:,} images. A number of "
        f"an arm is the mean of {n_seeds} seeds ({', '.join(map(str, context['seeds']))}). An interval is the 2.5% to "
        f"97.5% range of {context['n_boot']} paired bootstrap resamples of {of_groups} of the split: frames of one "
        "camera are not independent, so a resample draws whole groups. All arms and seeds are scored "
        "on the same resample, and the seeds of an arm are averaged. AP is COCO AP (IoU 0.50 to 0.95) as the mean of "
        "the 10 specific classes, from the 100 best (query, class) pairs of each image. A detection of "
        f"{', '.join(f'`{c}`' for c in IGNORE_UNDER)} on a `{MOTORIZED_VEHICLE}` box is not a true positive and not a "
        "false positive.", "",
        "## Results", "",
        "| Arm | Trainable parameters | AP [95% interval] | AP by seed | Recovered share [95% interval] |",
        "|---|---|---|---|---|",
    ]
    fixed_share = {"H": "0 (by definition)", "F": "1 (by definition)"}
    for arm in ARMS:
        ap = arms[arm]["metrics"]["AP"]
        share = fixed_share.get(arm) or _ci(numbers["recovered_share"][arm])
        trainable = context["trainable"].get(arm)
        lines.append(f"| {ARM_LABELS[arm]} | {trainable:,} | {_ci(ap)} | "
                     f"{' / '.join(_f(v) for v in ap['by_seed'])} | {share} |" if trainable is not None else
                     f"| {ARM_LABELS[arm]} | – | {_ci(ap)} | {' / '.join(_f(v) for v in ap['by_seed'])} | {share} |")
    if context.get("figure"):
        lines += ["", f"![Recovered share by adapter position]({context['figure']})"]
    lines += ["", "### Differences in AP", "",
              "| Difference | Value [95% interval] | Seed pairs that agree in sign with it |", "|---|---|---|"]
    for name, d in numbers["differences"].items():
        lines.append(f"| {name.replace(' - ', ' − ')} | {_ci(d, signed=True)} | "
                     f"{d['seed_pairs_in_agreement']} of {d['seed_pairs']} |")

    if numbers.get("image_bootstrap"):
        lines += ["", f"*The unit of the bootstrap.* AP_B − AP_D with resamples of scene groups: {_ci(b_d, signed=True)}. "
                  f"With resamples of single images: {_ci(numbers['image_bootstrap']['B - D'], signed=True)}. Frames of "
                  "one camera are not independent, so the image interval is too narrow. All other intervals and all "
                  "readings of this report use the scene groups."]

    lines += ["", "## Error split: boxes and labels", "",
              "Candidates: the 50 queries of an image with the highest class score. A ground-truth box is localized "
              "if a candidate has IoU ≥ 0.5 with it; the candidate with the highest IoU is its representative query. "
              "Label accuracy: of the localized boxes, the share whose representative query has the correct top "
              "class. Box AP: AP with all classes merged into one, one detection per query.", ""]
    for label in ERROR_SETS:
        e = numbers["errors"][label]
        lines += [f"### {label.capitalize()}", "", "| Arm | Localized share [95% interval] | Label accuracy [95% interval] |"
                  + (" Box AP |" if label == "specific classes" else ""),
                  "|---|---|---|" + ("---|" if label == "specific classes" else "")]
        for arm in ARMS:
            row = f"| {ARM_LABELS[arm]} | {_pct_ci(e[arm]['localized share'])} | {_pct_ci(e[arm]['label accuracy'])} |"
            if label == "specific classes":
                row += f" {_f(arms[arm]['other']['box AP'])} |"
            lines.append(row)
        diff = e["B - D"]
        lines += [f"| B − D (points) | {_pct_ci(diff['localized share'], signed=True)} | "
                  f"{_pct_ci(diff['label accuracy'], signed=True)} |" + (" |" if label == "specific classes" else ""), ""]
    lines += ["Box AP has all 11 classes as one class, `motorized_vehicle` boxes included.", ""]

    columns = [*CLASSES, NOT_LOCALIZED_NAME]
    for arm in ARMS:
        lines += [f"### Confusion of the truck family, {ARM_LABELS[arm]}", "",
                  f"Ground-truth box → top class of its representative query; share of the class's boxes, pooled over "
                  f"{n_seeds} seeds.", "",
                  "| Ground truth \\ label | " + " | ".join(columns) + " | Boxes per seed |",
                  "|---|" + "---|" * (len(columns) + 1)]
        for c in TRUCK_FAMILY:
            counts = numbers["confusion"][arm][c]
            total = sum(counts.values())
            cells = [f"{100 * counts.get(k, 0) / total:.0f}%" if total else "–" for k in columns]
            lines.append(f"| {c} | " + " | ".join(cells) + f" | {total // n_seeds if n_seeds else 0} |")
        lines.append("")

    lines += ["## Other metrics", "", "| Metric | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
              "|---|" + "---|" * len(ARMS)]
    for metric in ("AP50", "AP75"):
        lines.append(f"| {metric} | " + " | ".join(_ci(arms[a]["metrics"][metric]) for a in ARMS) + " |")
    for key in ("AP small", "AP truck family", f"AP {MOTORIZED_VEHICLE}", "AP 11 classes, no ignore regions",
                "box AP", "box AP50"):
        if key == "AP truck family":
            cells = [_ci(arms[a]["metrics"][key]) for a in ARMS]
        else:
            cells = [_f(arms[a]["other"][key]) for a in ARMS]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "AP of each specific class:", "", "| Class | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " |",
              "|---|" + "---|" * len(ARMS)]
    for c in SPECIFIC_CLASSES:
        lines.append(f"| {c} | " + " | ".join(_ci(arms[a]["metrics"][c]) for a in ARMS) + " |")

    lines += ["", "## Learning rates and the gate (val split, seed 0)", ""]
    selection = context.get("selection") or {}
    if selection:
        lines += ["| Arm | Chosen rate | Val AP of each rate tried |", "|---|---|---|"]
        for arm in ARMS:
            if arm in selection:
                s = selection[arm]
                chosen = (f"body {s['body_lr']:g}, heads {s['lr']:g}" if s.get("body_lr") is not None else f"{s['lr']:g}")
                tried = ", ".join(f"{lr}: {ap:.3f}" for lr, ap in s["val_ap"].items())
                lines.append(f"| {ARM_LABELS[arm]} | {chosen} | {tried} |")
        lines.append("")
    if context.get("gate"):
        lines += [gate_text(context["gate"]), ""]

    lines += ["## Checks", "",
              "| Arm | Runs | Merge difference (largest) | Skipped batches | Steps the fp16 scaler skipped |",
              "|---|---|---|---|---|"]
    for arm in ARMS:
        merges = [m for m in arms[arm]["merge_max_abs_diff"] if m is not None]
        skipped = [s for s in arms[arm]["skipped_batches"] if s is not None]
        scaler = [s for s in arms[arm]["scaler_skips"] if s is not None]
        lines.append(f"| {ARM_LABELS[arm]} | {', '.join(arms[arm]['runs'])} | "
                     f"{max(merges):.1e} | {sum(skipped)} | {sum(scaler)} |" if merges else
                     f"| {ARM_LABELS[arm]} | {', '.join(arms[arm]['runs'])} | – | – | – |")
    lines += ["", "A skipped batch had a loss that was not finite. A step that the fp16 scaler skipped had a scaled "
              "gradient that was not finite; the scaler halves its scale and the run goes on."]
    if context.get("samples"):
        lines += ["", "## Sample images", "",
                  "White: ground truth. Colour: detections with a score of 0.5 or more, by arm (seed 0).", ""]
        lines += [f"![{name}]({name})" for name in context["samples"]]
    caps = split_meta.get("caps") or {}
    lines += ["", "## Limits", "",
              "- **One model and one dataset.** The answer is for RF-DETR-base on traffic-camera images. It is not a "
              "general law.",
              "- **Full fine-tuning uses a simple recipe** (no weight averaging, no layer-wise learning-rate decay). If "
              "AP_F is low for that reason, the recovered shares are too high.",
              "- **Equal counts are not equal capacity.** Rank 33 is 13% of the full rank of a decoder matrix; rank 9 "
              "is 2% of a backbone matrix. This is to the advantage of the decoder.",
              "- **Scene groups are an estimate.** A scene group is a connected set of images that are linked at a "
              f"cosine similarity of {_limit(split_meta.get('limit'))} or more. A group can hold more than one "
              "camera: the lowest limit joins similar cameras. This is safe for the split, and it makes the groups "
              "fewer and larger.",
              "- **Two parts of one camera can stay in two groups** if no chain of near frames joins them (for "
              "example day and night). The scene check finds this only when the two parts are nearer than all other "
              "cameras.",
              f"- **The caps change the weight of the cameras.** A group gives at most {caps.get('test', 300)} images "
              f"to the test split and at most {caps.get('val', 100)} to the val split. A camera with many frames has "
              "no more weight in the test split than a camera at the cap. The train subset keeps the natural weights.",
              "- **The 20% rule is a choice.** The search for the limit stops when the largest group has more than "
              "20% of the images. No camera is expected to have 20% of the images.",
              "- **The intervals show the dependence on the test scenes.** They do not show the variance between "
              f"seeds; the tables give the {n_seeds} seed values and the seed pairs for that.",
              "- **Not a leaderboard number.** The public MIO-TCD leaderboard uses another split, all the training "
              "data and another protocol. Its numbers and these numbers are not a comparison.",
              "- **The library port.** Numbers can be different from those of the original `rfdetr` package."]
    return "\n".join(lines) + "\n"


def analyse(runs_root, data_root, splits_dir, out_dir, split: str = "test", n_boot: int = stats.N_BOOT,
            seeds=SEEDS, cache: bool = True) -> str:
    """Write `report.md`, `numbers.json` and `recovered_share.png` to `out_dir` and return the report."""
    runs_root, out_dir = Path(runs_root), Path(out_dir)
    runs = final_runs(runs_root, seeds)
    missing = [str(d / f"pred_{split}.npz") for dirs in runs.values() for d in dirs if not (d / f"pred_{split}.npz").exists()]
    if missing:
        raise FileNotFoundError("these predictions do not exist yet:\n  " + "\n  ".join(missing))
    gt = split_ground_truth(data_root, splits_dir, split)
    groups = split_groups(gt, splits_dir)
    weights = stats.group_resample_weights(groups, n_boot)
    image_weights = stats.resample_weights(len(gt["images"]), n_boot)  # for one comparison: AP_B - AP_D
    analyses = {}
    for arm, dirs in runs.items():
        analyses[arm] = []
        for d in dirs:
            print(f"[analyse] {d.name}", flush=True)
            analyses[arm].append(analyse_run(d / f"pred_{split}.npz", gt, weights, cache,
                                             image_weights if arm in ("B", "D") else None))
    numbers = build_numbers(analyses)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable = {}
    for arm, dirs in runs.items():
        params = dirs[0] / "params.json"
        trainable[arm] = json.loads(params.read_text())["table"]["total"]["trainable"] if params.exists() else None
    rows = [{"label": ARM_LABELS[arm], **numbers["recovered_share"][arm]} for arm in ("D", "B")]
    figure = "recovered_share.png"
    recovered_share_figure(
        rows, out_dir / figure, stats.P3_THRESHOLD,
        f"(AP of the arm − AP of H) / (AP of F − AP of H); {split} split, mean of {len(seeds)} seeds, 95% interval",
    )
    gate_path = runs_root / GATE
    samples_dir = out_dir / "samples"
    meta_path = Path(splits_dir) / "split_meta.json"
    context = {
        "images": len(gt["images"]), "groups": int(len(np.unique(groups))), "seeds": list(seeds), "n_boot": n_boot,
        "split": split, "trainable": trainable,
        "split_meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        "selection": read_selection(runs_root), "gate": json.loads(gate_path.read_text()) if gate_path.exists() else None,
        "figure": figure,
        "samples": sorted(f"samples/{p.name}" for p in samples_dir.glob("*.png")) if samples_dir.exists() else [],
    }
    text = render(numbers, context)
    write_json(out_dir / "numbers.json", numbers)
    (out_dir / "report.md").write_text(text)
    return text
