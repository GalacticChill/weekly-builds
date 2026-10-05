"""Charts: before/after cities, the tipping-point curve, and convergence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .model import EMPTY, A, B, segregation

# empty = light grey, type A = blue, type B = red
_CMAP = ListedColormap(["#ecebe4", "#2e86ab", "#d1495b"])


def plot_before_after(initial, final, tolerance: float, out_path: str | Path) -> Path:
    """Two city maps side by side: the random start and the settled result."""
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, grid, label in ((axes[0], initial, "Start (random)"),
                            (axes[1], final, "Settled")):
        ax.imshow(grid, cmap=_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"{label}\nsegregation = {segregation(grid):.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"A mild preference, a divided city  (agents want ≥ {tolerance:.0%} "
        f"same-type neighbours)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_tipping_curve(sweep, out_path: str | Path) -> Path:
    """Final segregation vs. tolerance, with the random-start baseline."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(sweep.tolerances, sweep.final_segregation, "o-", color="#2e86ab",
            linewidth=2, label="settled segregation")
    ax.fill_between(
        sweep.tolerances,
        sweep.final_segregation - sweep.final_segregation_std,
        sweep.final_segregation + sweep.final_segregation_std,
        color="#2e86ab", alpha=0.15,
    )
    ax.axhline(sweep.baseline, color="gray", linestyle="--", linewidth=1.2,
               label=f"random start (≈{sweep.baseline:.2f})")
    ax.axhline(0.7, color="#d1495b", linestyle=":", linewidth=1.2,
               label="\"divided city\" (0.70)")
    ax.set_xlabel("tolerance — minimum same-type neighbour share each agent wants")
    ax.set_ylabel("final segregation (mean same-type neighbour share)")
    ax.set_title("How little preference it takes: the segregation tipping curve")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_convergence(result, out_path: str | Path) -> Path:
    """Segregation and the content-share rising, round by round."""
    out_path = Path(out_path)
    rounds = np.arange(len(result.segregation_history))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(rounds, result.segregation_history, "o-", color="#2e86ab",
            label="segregation")
    ax.plot(rounds, result.happy_history, "s-", color="#16a085",
            label="content share")
    ax.set_xlabel("round")
    ax.set_ylabel("value")
    ax.set_title(f"Settling down (tolerance = {result.tolerance:.0%})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
