"""Charts: portfolio growth, correlation heatmap, and the efficient frontier."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import optimize as opt
from . import portfolio as pf


def plot_growth(
    returns: pd.DataFrame, weights: np.ndarray, out_path: str | Path
) -> Path:
    """Plot growth of $1 for the portfolio vs. each individual asset."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    asset_growth = (1.0 + returns).cumprod()
    for col in asset_growth.columns:
        ax.plot(asset_growth.index, asset_growth[col], alpha=0.5, label=col)

    port = pf.portfolio_growth(returns, weights)
    ax.plot(port.index, port.values, color="black", linewidth=2.2, label="Portfolio")

    ax.set_title("Growth of $1")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_correlation(returns: pd.DataFrame, out_path: str | Path) -> Path:
    """Heatmap of the return correlation matrix."""
    out_path = Path(out_path)
    corr = pf.correlation_matrix(returns)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black"
            )

    ax.set_title("Return correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_efficient_frontier(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    out_path: str | Path,
    risk_free: float = 0.0,
    n: int = 5000,
    seed: int | None = 42,
) -> Path:
    """Scatter of random portfolios coloured by Sharpe, overlaid with the true
    long-only efficient-frontier curve and the min-variance / max-Sharpe optima."""
    out_path = Path(out_path)
    cloud = pf.random_portfolios(mean_returns, cov, n=n, risk_free=risk_free, seed=seed)

    # The true frontier curve and the constrained (long-only) optima.
    frontier = opt.efficient_frontier(mean_returns, cov, n_points=60)
    mv_w = opt.min_variance_long_only(cov)
    ms_w = opt.max_sharpe_long_only(mean_returns, cov, risk_free)
    mv = pf.portfolio_performance(mv_w, mean_returns, cov, risk_free)
    ms = pf.portfolio_performance(ms_w, mean_returns, cov, risk_free)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        cloud["volatility"], cloud["return"], c=cloud["sharpe"], cmap="viridis", s=8
    )
    ax.plot(
        frontier["volatility"], frontier["return"],
        color="black", linewidth=2, label="Efficient frontier",
    )
    # Capital market line from the risk-free rate through the tangency portfolio.
    ax.plot(
        [0, ms[1]], [risk_free, ms[0]],
        color="gray", linestyle="--", linewidth=1, label="Capital market line",
    )
    ax.scatter(mv[1], mv[0], marker="*", s=320, color="red", label="Min variance")
    ax.scatter(ms[1], ms[0], marker="*", s=320, color="gold",
               edgecolor="black", label="Max Sharpe (tangency)")

    ax.set_title("Efficient frontier (long-only)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
