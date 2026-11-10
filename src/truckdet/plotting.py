"""Figures: the loss curves of a run, and the one figure of the report (recovered share by arm).

One style for both: the data is the only loud thing. Marks use the series colours; all text uses
ink colours; the grid and the axes are hairlines one step off the surface.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical colours in fixed order (blue, orange, aqua). Checked as a set for colour-blind separation.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")


def loss_curves(records: list[dict], out_png) -> None:
    """Training loss by epoch: the weighted total, and the three terms of the final decoder layer."""
    if not records:
        return
    epochs = [r["epoch"] + 1 for r in records]
    fig, (total, terms) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    total.plot(epochs, [r["loss"] for r in records], color=SERIES[0], linewidth=2, marker="o", markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    _style(total)
    _title(total, "Training loss", "All terms and layers, weighted; mean over the epoch")
    labels = {"loss_ce": "class", "loss_bbox": "box L1", "loss_giou": "box GIoU"}
    for colour, (key, label) in zip(SERIES, labels.items()):
        terms.plot(epochs, [r[key] for r in records], color=colour, linewidth=2, marker="o", markersize=5,
                   markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    _style(terms)
    _title(terms, "Final decoder layer, by term", "Not weighted")
    legend = terms.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    for ax in (total, terms):
        ax.set_xlabel("epoch", color=INK_SECONDARY, fontsize=9)
        ax.set_xticks(epochs if len(epochs) <= 12 else epochs[:: max(1, len(epochs) // 12)])
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def recovered_share_figure(rows: list[dict], out_png, threshold: float, subtitle: str) -> None:
    """One row per LoRA arm: the recovered share (a dot) with its interval (a line).

    `rows`: [{"label": ..., "value": ..., "low": ..., "high": ...}], drawn from the top down. Reference lines
    mark 0 (heads only), 1 (full fine-tuning) and `threshold` (the line of prediction P3)."""
    fig, ax = plt.subplots(figsize=(8.5, 1.9 + 0.75 * len(rows)), facecolor=SURFACE)
    _style(ax)
    ax.grid(False)
    positions = list(range(len(rows)))[::-1]
    finite = [v for r in rows for v in (r["low"], r["high"], r["value"]) if v == v]
    low, high = min([0.0, *finite]), max([1.0, *finite])
    span = high - low
    ax.set_xlim(low - 0.06 * span, high + 0.30 * span)  # room on the right for the label of the last reference line
    ax.set_ylim(-0.65, len(rows) - 0.1)
    references = [(0.0, "0: heads only (H)"), (threshold, "2/3: line of P3"), (1.0, "1: full fine-tuning (F)")]
    for x, label in references:
        ax.axvline(x, color=AXIS, linewidth=1, zorder=1)
        ax.annotate(label, (x, len(rows) - 0.32), xytext=(4, 0), textcoords="offset points", color=INK_MUTED,
                    fontsize=8, ha="left", va="center")
    for y, r in zip(positions, rows):
        if r["value"] != r["value"]:
            continue
        ax.plot([r["low"], r["high"]], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([r["value"]], [y], marker="o", markersize=9, color=SERIES[0], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.text(r["value"], y - 0.28, f"{r['value']:.2f}  [{r['low']:.2f}, {r['high']:.2f}]", color=INK,
                fontsize=9, ha="center", va="top")
    ax.set_yticks(positions)
    ax.set_yticklabels([r["label"] for r in rows], color=INK, fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("recovered share", color=INK_SECONDARY, fontsize=9)
    _title(ax, "Recovered share of the full fine-tuning gain, by adapter position", subtitle)
    fig.tight_layout()
    fig.savefig(Path(out_png), dpi=140, facecolor=SURFACE)
    plt.close(fig)
