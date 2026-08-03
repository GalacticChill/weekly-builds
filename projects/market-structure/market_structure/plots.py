"""Two charts: the clustering dendrogram and the reordered correlation heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

# A stable, colorblind-friendly-ish palette for sector coloring.
_PALETTE = [
    "#2e86ab", "#d1495b", "#16a085", "#e0a800",
    "#8e44ad", "#e67e22", "#2c3e50", "#c0392b",
]


def _sector_colors(sectors) -> dict[str, str]:
    """Map each distinct sector to a fixed color."""
    unique = sorted(set(sectors))
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(unique)}


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    sectors: list[str],
    out_path: str | Path,
    color_threshold: float | None = None,
) -> Path:
    """Draw the clustering tree, coloring each leaf label by its *true* sector.

    If the discovered tree matches reality, same-colored labels sit together in
    contiguous blocks — you can read the agreement straight off the axis.
    """
    out_path = Path(out_path)
    sector_of = dict(zip(labels, sectors))
    colors = _sector_colors(sectors)

    fig, ax = plt.subplots(figsize=(13, 6))
    # Draw all links in one neutral color so the ONLY semantic color on the chart
    # is the sector coloring of the leaf labels (matching the legend). Passing a
    # threshold of 0 sends every link through `above_threshold_color`.
    dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        color_threshold=0.0 if color_threshold is None else color_threshold,
        above_threshold_color="#9aa0a6",
        leaf_rotation=90,
        leaf_font_size=9,
    )
    ax.set_title("Stocks clustered by return correlation (labels colored by true sector)")
    ax.set_ylabel("distance = sqrt(2(1 - correlation))")

    for tick in ax.get_xticklabels():
        tick.set_color(colors[sector_of[tick.get_text()]])

    handles = [Patch(color=c, label=s) for s, c in colors.items()]
    ax.legend(handles=handles, title="True sector", loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(
    corr: pd.DataFrame, order: list[str], out_path: str | Path
) -> Path:
    """Correlation heatmap with rows/cols reordered by the clustering.

    Reordering is the whole trick: raw, the matrix looks like noise; sorted by the
    dendrogram, correlated groups line up into bright blocks along the diagonal.
    """
    out_path = Path(out_path)
    c = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(c.to_numpy(), vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title("Return correlation, reordered by clustering\n(diagonal blocks = discovered groups)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
