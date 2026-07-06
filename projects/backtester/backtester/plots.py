"""Charts: equity curves and the underwater (drawdown) plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt

from . import metrics
from .engine import BacktestResult


def plot_equity(
    results: dict[str, BacktestResult], out_path: str | Path
) -> Path:
    """Overlay the equity curves of several backtests (e.g. rebalanced vs hold)."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=label)
    ax.set_title("Equity curve")
    ax.set_ylabel("Portfolio value (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_drawdown(
    results: dict[str, BacktestResult], out_path: str | Path
) -> Path:
    """Underwater plot: how far below the running peak each strategy sits."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, res in results.items():
        dd = metrics.drawdown_series(res.equity)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3)
        ax.plot(dd.index, dd.values, label=label, linewidth=1)
    ax.set_title("Drawdown (underwater plot)")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
