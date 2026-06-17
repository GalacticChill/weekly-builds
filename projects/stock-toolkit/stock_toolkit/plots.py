"""Chart helpers. Saves PNGs suitable for embedding in a README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import moving_average  # noqa: E402


def plot_price_with_mas(
    prices: pd.Series, ticker: str, windows=(20, 50), out: str | Path = "assets/price.png"
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices.index, prices.values, label="Close", linewidth=1.2)
    for w in windows:
        ax.plot(prices.index, moving_average(prices, w).values, label=f"{w}-day MA")
    ax.set_title(f"{ticker} — Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_cumulative_return(
    prices: pd.Series, ticker: str, out: str | Path = "assets/cumulative.png"
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cum = prices / prices.iloc[0] - 1.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum.index, cum.values * 100, color="green", linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{ticker} — Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
