"""Charts: the shrinkage pull, accuracy by sample size, and example intervals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_shrinkage(df: pd.DataFrame, prior_mean: float, out_path: str | Path) -> Path:
    """Raw rate vs. shrunken rate; small-sample players are pulled hardest."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sizes = 6 + 60 * (df["trials"] / df["trials"].max())
    sc = ax.scatter(df["raw_rate"], df["shrunk_rate"], s=sizes, alpha=0.5,
                    c=np.log10(df["trials"]), cmap="viridis", edgecolor="white",
                    linewidth=0.2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="no shrinkage (y = x)")
    ax.axhline(prior_mean, color="#d1495b", linewidth=1.2, linestyle=":",
               label=f"prior mean ({prior_mean:.2f})")
    ax.set_xlabel("raw rate (successes / trials)")
    ax.set_ylabel("shrunken rate (empirical-Bayes posterior mean)")
    ax.set_title("Shrinkage pulls noisy small-sample rates toward the crowd")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log10(trials)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_rmse_by_size(table: pd.DataFrame, out_path: str | Path) -> Path:
    """Grouped bars: raw vs shrunken error within each sample-size bucket."""
    out_path = Path(out_path)
    x = np.arange(len(table))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - w / 2, table["raw_rmse"], w, label="raw rate", color="#e0a800")
    ax.bar(x + w / 2, table["shrunk_rmse"], w, label="shrunken (empirical Bayes)",
           color="#2e86ab")
    ax.set_xticks(x)
    ax.set_xticklabels(table["bucket"], rotation=15)
    ax.set_ylabel("RMSE vs. true rate (lower is better)")
    ax.set_title("Shrinkage helps most exactly where data is thinnest")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_examples(df: pd.DataFrame, prior_mean: float, out_path: str | Path,
                  n: int = 12) -> Path:
    """Credible intervals for a few tiny-sample vs well-measured players."""
    out_path = Path(out_path)
    small = df[df["trials"] <= 5].nlargest(n // 2, "raw_rate")
    big = df[df["trials"] >= 100].nlargest(n // 2, "raw_rate")
    pick = pd.concat([small, big]).sort_values("trials")

    y = np.arange(len(pick))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.hlines(y, pick["ci_low"], pick["ci_high"], color="#2e86ab", linewidth=2,
              alpha=0.7, label="90% credible interval")
    ax.scatter(pick["raw_rate"], y, color="#e0a800", zorder=3, label="raw rate")
    ax.scatter(pick["shrunk_rate"], y, color="#2e86ab", zorder=3, label="shrunken")
    ax.scatter(pick["true_rate"], y, marker="x", color="black", zorder=4, label="true rate")
    ax.axvline(prior_mean, color="#d1495b", linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.successes}/{r.trials}" for r in pick.itertuples()])
    ax.set_xlabel("success rate")
    ax.set_title("A 5-for-5 isn't a 100% player: raw rate, shrunken estimate, and truth")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
