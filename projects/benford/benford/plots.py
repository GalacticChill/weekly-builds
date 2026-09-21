"""Charts: one dataset against the Benford curve, and a grid of many."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .law import DIGITS, analyze

_CONFORM_COLOR = "#2e86ab"
_FAIL_COLOR = "#d1495b"


def _bars(ax, result, title):
    """Observed bars + the Benford reference curve on one axis."""
    conforms = result.mad < 0.015
    color = _CONFORM_COLOR if conforms else _FAIL_COLOR
    ax.bar(DIGITS, result.observed, color=color, alpha=0.85, label="observed")
    ax.plot(DIGITS, result.expected, "o-", color="black", linewidth=1.5,
            markersize=4, label="Benford")
    ax.set_xticks(DIGITS)
    ax.set_title(f"{title}\n(n={result.n:,}, MAD={result.mad:.3f}, {result.conforms})",
                 fontsize=9)
    ax.set_ylim(0, max(result.observed.max(), result.expected.max()) * 1.15)


def plot_single(values, out_path: str | Path, title: str = "Leading digits") -> Path:
    """A single dataset against the Benford curve."""
    out_path = Path(out_path)
    result = analyze(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    _bars(ax, result, title)
    ax.set_xlabel("leading digit")
    ax.set_ylabel("proportion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_grid(datasets: dict, out_path: str | Path) -> Path:
    """A small-multiple grid: every dataset against the Benford curve.

    Blue bars conform, red bars don't — so the real-world and mathematical sets
    should read blue and the controls red, at a glance.
    """
    out_path = Path(out_path)
    names = list(datasets)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        values, _kind = datasets[name]
        _bars(ax, analyze(values), name)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle("Benford's Law across datasets — blue conforms, red does not",
                 fontsize=13)
    fig.supxlabel("leading digit")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
