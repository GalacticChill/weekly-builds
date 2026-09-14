"""Three charts: the confusion matrix, author signatures, and the style map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import features as ft

_PALETTE = ["#2e86ab", "#d1495b", "#16a085", "#e0a800", "#8e44ad", "#e67e22"]


def _colors(labels):
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(sorted(set(labels)))}


def plot_confusion(result, out_path: str | Path) -> Path:
    """Confusion matrix of who-gets-mistaken-for-whom (row-normalized)."""
    out_path = Path(out_path)
    cm = result.confusion.astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = result.labels

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted author")
    ax.set_ylabel("True author")
    ax.set_title(f"Authorship attribution ({result.representation}-word view)\n"
                 f"leave-books-out accuracy = {result.accuracy:.0%}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of passages")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_signatures(sig: pd.DataFrame, out_path: str | Path) -> Path:
    """Small multiples: each author's most over-used function words."""
    out_path = Path(out_path)
    authors = sorted(sig["author"].unique())
    colors = _colors(authors)
    n = len(authors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, author in zip(axes, authors):
        s = sig[sig["author"] == author].sort_values("distinctiveness")
        ax.barh(s["word"], s["distinctiveness"], color=colors[author])
        ax.set_title(author, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.axvline(0, color="black", linewidth=0.6)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Each author's fingerprint: most distinctive function words", fontsize=13)
    fig.supxlabel("standard deviations above the corpus average")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_style_map(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Every passage projected to 2-D from its function-word profile (PCA).

    If style is real, passages cluster by author even though the axes know nothing
    about who wrote what.
    """
    out_path = Path(out_path)
    x, _ = ft.function_word_matrix(df["text"].tolist())
    coords = PCA(n_components=2, random_state=0).fit_transform(x)
    authors = df["author"].to_numpy()
    colors = _colors(authors)

    fig, ax = plt.subplots(figsize=(10, 7))
    for author in sorted(set(authors)):
        m = authors == author
        ax.scatter(coords[m, 0], coords[m, 1], s=16, alpha=0.7,
                   color=colors[author], label=author, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("style axis 1")
    ax.set_ylabel("style axis 2")
    ax.set_title("The shape of style: passages by function-word profile (PCA)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
