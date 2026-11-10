"""Uncertainty, and the readings of design §2 as text (from `diagnostics.py` of the truck project).

- Paired bootstrap of scene groups (design §7.3): frames of one camera are not independent, so a
  resample draws whole scene groups of a split with replacement, and not single images. Every run
  is scored on the *same* resample. A resample is a weight for each image (how many times its
  group was drawn); `evaluation.average_precision` and `errors.bootstrap_error_split` take the
  weights, so nothing is matched again. Resample i depends only on (seed, i, the groups): runs
  that are scored at different times are still scored on the same resamples.
- The bootstrap of single images (`resample_weights`) stays for one comparison in the report: the
  image interval of AP_B - AP_D next to the group interval.
- A number of an arm is the mean of its seeds. In each resample the seeds are scored on the same
  images and then averaged, so an interval is for the seed mean. It shows the dependence on which
  scenes are in the split. It does not show the variance between seeds: for that the report gives
  the value of each seed and `seed_pairings`.
- The recovered share R_X = (AP_X - AP_H) / (AP_F - AP_H) is calculated in each resample from the
  seed means; its interval comes from the same resamples.
"""

import numpy as np

from .config import LR_LADDER
from .evaluation import Matches, metric_vector

N_BOOT = 400
BOOTSTRAP_SEED = 0
GATE_MIN_DIFFERENCE = 0.05
GATE_MIN_RATIO = 5.0
P3_THRESHOLD = 2.0 / 3.0

CONFIRMED, REFUTED, NOT_SHOWN = "confirmed", "refuted", "not shown"


def resample_seeds(seed: int, n_boot: int) -> list[int]:
    """One seed per bootstrap resample."""
    return np.random.default_rng(seed).integers(0, 2**32 - 1, size=n_boot).tolist()


def resample_weights(n_images: int, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """[n_boot, n_images]: how many times each image is drawn in each resample of `n_images` draws with
    replacement. Column k is for the k-th image in sorted id order."""
    weights = np.empty((n_boot, n_images), dtype=np.int32)
    for row, s in enumerate(resample_seeds(seed, n_boot)):
        picks = np.random.default_rng(s).integers(0, n_images, size=n_images)
        weights[row] = np.bincount(picks, minlength=n_images)
    return weights


def group_resample_weights(groups, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """[n_boot, n_images]: the weights of a bootstrap that resamples scene groups. `groups` has the scene group of
    each image, with the images in sorted id order. A resample draws G of the G groups with replacement; the
    weight of an image is the number of times its group was drawn. With one image in each group these are the
    weights of `resample_weights`."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(inverse.max()) + 1 if len(inverse) else 0
    weights = np.empty((n_boot, len(inverse)), dtype=np.int32)
    for row, s in enumerate(resample_seeds(seed, n_boot)):
        picks = np.random.default_rng(s).integers(0, n_groups, size=n_groups)
        weights[row] = np.bincount(picks, minlength=n_groups)[inverse]
    return weights


def bootstrap_metrics(matches: Matches, weights: np.ndarray) -> np.ndarray:
    """`evaluation.metric_vector` for each resample: [n_boot, M]."""
    return np.stack([metric_vector(matches, w) for w in weights])


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, ignoring NaN."""
    tail = (1 - level) / 2 * 100
    return float(np.nanpercentile(samples, tail)), float(np.nanpercentile(samples, 100 - tail))


def bootstrap_se(samples: np.ndarray) -> float:
    """Standard error of a number: the sd (ddof=1) of its bootstrap resamples, ignoring NaN."""
    finite = np.asarray(samples, dtype=float)
    finite = finite[~np.isnan(finite)]
    return float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")


# Differences this small are float noise (e.g. mean([0.4] * 3) - 0.4 == 1e-16) and count as zero.
TOL = 1e-9


def seed_pairings(a, b) -> tuple[int, int, int]:
    """Every seed of A against every seed of B: (pairs with a > b, pairs with b > a, all pairs).
    Ties (within TOL) count for neither side."""
    diff = np.asarray(a, dtype=float)[:, None] - np.asarray(b, dtype=float)[None, :]
    return int((diff > TOL).sum()), int((diff < -TOL).sum()), int(diff.size)


def pairs_in_agreement(a, b) -> tuple[int, int]:
    """(seed pairs that agree in sign with the difference of the means, all pairs): design §7.3."""
    a_wins, b_wins, total = seed_pairings(a, b)
    difference = float(np.mean(a) - np.mean(b))
    return (a_wins if difference > TOL else b_wins if difference < -TOL else 0), total


def recovered_share(ap_x, ap_h, ap_f):
    """R_X = (AP_X - AP_H) / (AP_F - AP_H); works on numbers and on arrays of resamples. NaN where F equals H."""
    gain = np.asarray(ap_f, dtype=float) - np.asarray(ap_h, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(gain) > TOL, (np.asarray(ap_x, dtype=float) - ap_h) / gain, np.nan)


# --- The readings of design §2 -------------------------------------------------------------


def reading(low: float, high: float, threshold: float = 0.0) -> str:
    """A prediction "the number is above `threshold`": confirmed if the interval is above it, refuted if the
    interval is below it, not shown in all other cases."""
    if np.isnan(low) or np.isnan(high):
        return NOT_SHOWN
    if low > threshold:
        return CONFIRMED
    if high < threshold:
        return REFUTED
    return NOT_SHOWN


def reading_p1(ap_b_minus_d: tuple[float, float]) -> str:
    """P1: LoRA in the backbone is better than LoRA in the decoder."""
    return reading(*ap_b_minus_d)


def reading_p2(label_b_minus_d: tuple[float, float], localized_b_minus_d: tuple[float, float]) -> str:
    """P2: the advantage of the backbone is in the labels, not in the boxes (truck family). Confirmed if the
    interval of the label accuracy difference is above zero and the interval of the localized share
    difference includes zero. Refuted if the interval of the label accuracy difference is below zero."""
    labels = reading(*label_b_minus_d)
    if labels == REFUTED:
        return REFUTED
    if labels == CONFIRMED and reading(*localized_b_minus_d) == NOT_SHOWN and not np.isnan(localized_b_minus_d[0]):
        return CONFIRMED
    return NOT_SHOWN


def reading_p3(recovered_b: tuple[float, float]) -> str:
    """P3: LoRA in the backbone recovers most of the full fine-tuning gain (the interval of R_B is above 2/3)."""
    return reading(*recovered_b, threshold=P3_THRESHOLD)


def gate(ap_h: float, ap_f: float, samples_h: np.ndarray, samples_f: np.ndarray) -> dict:
    """The gate of design §2, on the val split with seed 0: continue only if AP_F - AP_H is at least 0.05 AP
    and at least 5 times its bootstrap standard error. `samples_*`: the AP of each run on the same resamples
    (of the scene groups of the val split)."""
    difference = float(ap_f - ap_h)
    se = bootstrap_se(np.asarray(samples_f) - np.asarray(samples_h))
    ratio = difference / se if se > 0 else float("inf") if difference > 0 else 0.0
    return {
        "AP_H": float(ap_h), "AP_F": float(ap_f), "difference": difference, "bootstrap_se": se, "ratio": ratio,
        "min_difference": GATE_MIN_DIFFERENCE, "min_ratio": GATE_MIN_RATIO,
        "passed": bool(difference >= GATE_MIN_DIFFERENCE and ratio >= GATE_MIN_RATIO),
    }


# --- Learning-rate choice (design §6) ------------------------------------------------------


def choose_lr(val_ap: dict[float, float]) -> float:
    """The rate with the highest val AP of the final checkpoint. If two are equal to 3 decimal places, the lower rate."""
    if not val_ap:
        raise ValueError("no learning-rate run to choose from")
    best = max(round(ap, 3) for ap in val_ap.values())
    return min(lr for lr, ap in val_ap.items() if round(ap, 3) == best)


def _ladder_index(lr: float) -> int:
    for i, step in enumerate(LR_LADDER):
        if np.isclose(lr, step, rtol=1e-6):
            return i
    raise ValueError(f"learning rate {lr} is not on the ladder {LR_LADDER}")


def lr_to_add(grid: list[float], val_ap: dict[float, float]) -> float | None:
    """The one more rate to run, or None. "If the best rate is at an end of its list, add one step in that
    direction, one time": the answer is a rate only while exactly the rates of `grid` have a result."""
    done = sorted(val_ap)
    if len(done) != len(grid) or not np.allclose(done, sorted(grid), rtol=1e-6):
        return None
    best = choose_lr(val_ap)
    i = _ladder_index(best)
    if np.isclose(best, min(grid), rtol=1e-6) and i > 0:
        return LR_LADDER[i - 1]
    if np.isclose(best, max(grid), rtol=1e-6) and i + 1 < len(LR_LADDER):
        return LR_LADDER[i + 1]
    return None
