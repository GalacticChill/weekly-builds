"""Charts: forecast-vs-actual for each series, and an error comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = {
    "naive-last": "#e0a800",
    "seasonal-naive": "#16a085",
    "trend+seasonal": "#d1495b",
}


def plot_forecast(series: pd.Series, card, out_path: str | Path,
                  context: int = 120) -> Path:
    """History, then the held-out future with every method's forecast overlaid."""
    out_path = Path(out_path)
    train_tail = series.iloc[-(card.horizon + context):-card.horizon]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(train_tail.index, train_tail.values, color="#2c3e50", linewidth=1.2,
            label="history (train)")
    ax.plot(card.actual.index, card.actual.values, color="black", linewidth=2.2,
            label="actual (held-out future)")
    for label, fc in card.forecasts.items():
        ax.plot(card.actual.index, fc, "--", color=_COLORS[label], linewidth=1.6,
                label=f"{label} (MAE {card.mae[label]:.2f})")
    ax.axvline(card.train_end, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Forecasting {card.name}: {card.horizon} months ahead, "
                 f"trained only on the past\nbest = {card.best}")
    ax.set_ylabel(series.name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_error_comparison(cards: dict, out_path: str | Path) -> Path:
    """Grouped bars of forecast MAE by method, per series (log scale)."""
    out_path = Path(out_path)
    names = list(cards)
    methods = list(next(iter(cards.values())).mae)
    x = np.arange(len(names))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, method in enumerate(methods):
        vals = [cards[n].mae[method] for n in names]
        ax.bar(x + i * width, vals, width, label=method, color=_COLORS[method])
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("forecast error (MAE, log scale)")
    ax.set_title("Who wins depends on whether the series has real structure")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
