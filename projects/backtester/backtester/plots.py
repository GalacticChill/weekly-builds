"""Charts: equity curves and the underwater (drawdown) plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt

import pandas as pd

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


def plot_curves(
    curves: dict[str, pd.Series], out_path: str | Path, title: str = "Equity curve"
) -> Path:
    """Overlay several bare equity-curve Series (e.g. the walk-forward comparison)."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, eq in curves.items():
        ax.plot(eq.index, eq.values, label=label)
    ax.set_title(title)
    ax.set_ylabel("Out-of-sample value (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_param_choices(
    choices: pd.DataFrame, param_name: str, out_path: str | Path
) -> Path:
    """Step plot of which parameter walk-forward picked in each fold.

    An unstable line is itself a finding: if the 'best' lookback lurches around
    from fold to fold, that parameter was never a stable signal to begin with.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = pd.to_datetime(choices["test_start"])
    ax.step(x, choices[param_name], where="post", marker="o")
    ax.set_title(f"Walk-forward: {param_name} chosen per fold (from past data only)")
    ax.set_ylabel(param_name)
    ax.set_xlabel("Out-of-sample window start")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
