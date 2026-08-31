"""Three charts: the Preston curve, what predicts longevity, and who beats the odds."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .collect import MONEY, TARGET

# Colour income groups consistently across charts.
_INCOME_COLORS = {
    "Low income": "#d1495b",
    "Lower middle income": "#e0a800",
    "Upper middle income": "#4c9f70",
    "High income": "#2e86ab",
}


def _income_color(group: str) -> str:
    return _INCOME_COLORS.get(group, "#888888")


def plot_preston(df: pd.DataFrame, fit, out_path: str | Path, annotate=None) -> Path:
    """Life expectancy vs. income on a log-income axis, with the fitted curve."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    groups = df["income_group"] if "income_group" in df else pd.Series("", index=df.index)
    for grp in _INCOME_COLORS:
        m = groups == grp
        if m.any():
            ax.scatter(df.loc[m, MONEY], df.loc[m, TARGET], s=28, alpha=0.8,
                       color=_income_color(grp), label=grp, edgecolor="white", linewidth=0.4)
    other = ~groups.isin(_INCOME_COLORS)
    if other.any():
        ax.scatter(df.loc[other, MONEY], df.loc[other, TARGET], s=28, alpha=0.8,
                   color="#888888", label="Other / unclassified", edgecolor="white", linewidth=0.4)

    grid = np.logspace(np.log10(df[MONEY].min()), np.log10(df[MONEY].max()), 200)
    ax.plot(grid, fit.predict(grid), color="black", linewidth=2,
            label=f"log-income fit (R² = {fit.r2_log:.2f})")

    if annotate is not None:
        for iso3, row in annotate.iterrows():
            ax.annotate(row["country"], (df.loc[iso3, MONEY], df.loc[iso3, TARGET]),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("GDP per capita, PPP (international $, log scale)")
    ax.set_ylabel("Life expectancy at birth (years)")
    ax.set_title("Does money buy a longer life? The Preston curve")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_importances(result, out_path: str | Path) -> Path:
    """Random-forest feature importances for predicting life expectancy."""
    out_path = Path(out_path)
    imps = result.importances.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(imps.index, imps.values, color="#2e86ab")
    ax.set_xlabel("Random-forest importance (share of predictive power)")
    ax.set_title("What predicts a country's life expectancy?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_residuals(resid: pd.DataFrame, out_path: str | Path, n: int = 12) -> Path:
    """Countries that most beat, or fall short of, what their income predicts."""
    out_path = Path(out_path)
    top = resid.head(n)          # biggest positive residuals ("beat the odds")
    bottom = resid.tail(n)       # biggest negative residuals ("fall short")
    both = pd.concat([bottom, top])
    colors = ["#d1495b" if v < 0 else "#4c9f70" for v in both["residual"]]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.barh(both["country"], both["residual"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Extra years of life vs. what income alone predicts")
    ax.set_title("Who beats the odds? Life expectancy above/below the income curve")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
