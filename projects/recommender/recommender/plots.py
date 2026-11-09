"""Charts: the RMSE ladder, the cold-start curve, and ranking vs popularity."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rmse_ladder(rmses: dict, out_path: str | Path) -> Path:
    """Bar chart of test RMSE per model — note how little MF adds over the biases."""
    out_path = Path(out_path)
    names = list(rmses)
    vals = [rmses[n] for n in names]
    colors = ["#c0c0c0", "#e0a800", "#4c9f70", "#2e86ab"][: len(names)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("test RMSE (lower is better)")
    ax.set_title("Rating-prediction error: the simple biases do most of the work")
    ax.set_ylim(0.8, max(vals) * 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cold_start(table: pd.DataFrame, out_path: str | Path) -> Path:
    """MF RMSE by how many ratings a user has — thin history, worse predictions."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(table["bucket"], table["rmse"], color="#2e86ab")
    for i, r in table.iterrows():
        ax.text(i, r["rmse"] + 0.004, f"{r['rmse']:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("test RMSE")
    ax.set_xlabel("ratings the user has in training")
    ax.set_title("Cold start: the less a user has rated, the worse the guess")
    ax.set_ylim(0.8, table["rmse"].max() * 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_ranking(recall: dict, out_path: str | Path) -> Path:
    """Top-K recall of held-out liked movies: MF vs the popularity baseline."""
    out_path = Path(out_path)
    k = recall["k"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    names = ["matrix\nfactorization", "most\npopular"]
    vals = [recall["mf"], recall["popularity"]]
    bars = ax.bar(names, vals, color=["#2e86ab", "#d1495b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel(f"recall@{k} of held-out liked movies")
    ax.set_title("The twist: better ratings, worse recommendations\n"
                 "(RMSE-trained MF loses top-N recall to plain popularity)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
