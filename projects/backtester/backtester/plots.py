"""Charts: equity curves and the underwater (drawdown) plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt

import numpy as np
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


def plot_bootstrap(
    samples,
    observed: float,
    ci: tuple[float, float],
    out_path: str | Path,
    title: str = "Bootstrap distribution of the Sharpe ratio",
    xlabel: str = "Sharpe ratio",
) -> Path:
    """Histogram of a bootstrap distribution with the observed value, CI, and zero.

    Reading it: if the shaded confidence band sits entirely to the right of the
    dashed zero line, the edge is unlikely to be luck; if the band straddles zero,
    it might be.
    """
    out_path = Path(out_path)
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(samples, bins=40, color="#4c72b0", alpha=0.75, edgecolor="white")
    ax.axvspan(ci[0], ci[1], color="#4c72b0", alpha=0.15, label="95% CI")
    ax.axvline(observed, color="#c44e52", linewidth=2, label=f"observed = {observed:.2f}")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="no edge (0)")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bootstrap resamples")
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
