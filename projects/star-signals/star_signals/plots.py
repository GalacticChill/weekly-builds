"""Charts: feature importances, the signal-vs-substance split, and model fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from . import features as ft
from .model import ModelResult

_COLORS = {"signal": "#d1495b", "substance": "#2e86ab"}


def plot_importances(res: ModelResult, out_path: str | Path) -> Path:
    """Horizontal bar of random-forest importances, coloured by group."""
    out_path = Path(out_path)
    imp = res.importances[::-1]  # smallest at bottom for a clean top-down read
    colors = [_COLORS[ft.group_of(f)] for f in imp.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title("What predicts a repo's stars? (random-forest importance)")
    ax.set_xlabel("Importance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _COLORS.values()]
    ax.legend(handles, [k.capitalize() for k in _COLORS], loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_group_split(res: ModelResult, out_path: str | Path) -> Path:
    """Side-by-side bars: share of predictive power from signal vs substance,
    by both the forest and the linear model."""
    out_path = Path(out_path)
    groups = ["signal", "substance"]
    forest = [res.group_importance.get(g, 0.0) for g in groups]
    linear = [res.group_coef_share.get(g, 0.0) for g in groups]

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x - width / 2, forest, width, label="Random forest",
           color=[_COLORS[g] for g in groups])
    ax.bar(x + width / 2, linear, width, label="Linear |coef|",
           color=[_COLORS[g] for g in groups], alpha=0.55)
    ax.set_xticks(x, [g.capitalize() for g in groups])
    ax.set_ylabel("Share of predictive power")
    ax.set_title("Signal vs substance: what earns the stars?")
    ax.legend()
    for i, v in enumerate(forest):
        ax.text(i - width / 2, v + 0.01, f"{v:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_fit(res: ModelResult, out_path: str | Path) -> Path:
    """Predicted vs actual log-stars on the held-out test set."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(res.y_test, res.y_pred, s=12, alpha=0.4, color="#2e86ab")
    lims = [
        min(res.y_test.min(), res.y_pred.min()),
        max(res.y_test.max(), res.y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual log(stars)")
    ax.set_ylabel("Predicted log(stars)")
    ax.set_title(f"Out-of-sample fit (R² = {res.forest_r2:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
