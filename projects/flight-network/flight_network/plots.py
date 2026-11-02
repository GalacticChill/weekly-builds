"""Charts: the degree tail, the robust-yet-fragile curves, and a hub map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graph import degree_distribution


def plot_degree_distribution(adj, meta: pd.DataFrame, out_path: str | Path) -> Path:
    """Log-log degree distribution — a heavy tail of a few giant hubs."""
    out_path = Path(out_path)
    vals, counts = degree_distribution(adj)
    pos = vals > 0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(vals[pos], counts[pos], s=18, color="#2e86ab", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("direct destinations (degree)")
    ax.set_ylabel("number of airports")
    ax.set_title("Most airports are small; a few are giant hubs (log-log)")
    top = meta.sort_values("degree", ascending=False).head(5)
    for code, r in top.iterrows():
        ax.annotate(code, (r["degree"], 1), fontsize=8, rotation=90,
                    va="bottom", ha="center", color="#d1495b")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_robustness(random_curve, targeted_curve, out_path: str | Path) -> Path:
    """Giant-component share as nodes are removed: random vs. hubs-first."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(random_curve.fraction_removed, random_curve.giant_fraction, "o-",
            color="#2e86ab", label=random_curve.label)
    ax.plot(targeted_curve.fraction_removed, targeted_curve.giant_fraction, "s-",
            color="#d1495b", label=targeted_curve.label)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="half the network")
    ax.set_xlabel("fraction of airports removed")
    ax.set_ylabel("largest connected component (share of all airports)")
    ax.set_title("Robust yet fragile: random failures barely hurt, hub attacks shatter")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_hub_map(meta: pd.DataFrame, out_path: str | Path, top: int = 30) -> Path:
    """Airports on a lon/lat scatter, sized by degree, with the big hubs named."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sizes = 2 + 90 * (meta["degree"] / meta["degree"].max())
    ax.scatter(meta["lon"], meta["lat"], s=sizes, alpha=0.35, color="#2e86ab",
               edgecolor="none")
    hubs = meta.sort_values("degree", ascending=False).head(top)
    ax.scatter(hubs["lon"], hubs["lat"], s=2 + 90 * (hubs["degree"] / meta["degree"].max()),
               color="#d1495b", alpha=0.8, edgecolor="white", linewidth=0.3)
    for code, r in hubs.head(12).iterrows():
        ax.annotate(code, (r["lon"], r["lat"]), fontsize=7.5, color="black",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("The world's airports, sized by number of direct destinations")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
