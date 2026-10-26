"""Charts: the bias summary, the collider illusion, and the three DAGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .estimate import effect


def plot_bias_summary(results: dict, out_path: str | Path) -> Path:
    """Per scenario: the true effect vs the naive and adjusted estimates."""
    out_path = Path(out_path)
    names = list(results)
    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    true = [results[n].true_effect for n in names]
    naive = [results[n].naive_mean for n in names]
    naive_e = [results[n].naive_std for n in names]
    adj = [results[n].adjusted_mean for n in names]
    adj_e = [results[n].adjusted_std for n in names]

    ax.bar(x - w, true, w, label="true effect", color="#2c3e50")
    ax.bar(x, naive, w, yerr=naive_e, capsize=3, label="naive (no adjustment)",
           color="#e0a800")
    ax.bar(x + w, adj, w, yerr=adj_e, capsize=3, label="adjusted for the 3rd variable",
           color="#2e86ab")

    # Mark which estimate is the correct one in each scenario.
    for i, n in enumerate(names):
        r = results[n]
        correct_x = (x[i] + w) if r.adjusting_is_right else x[i]
        ax.annotate("correct", (correct_x, r.correct_estimate),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, color="#16a085", weight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({results[n].extra})" for n in names])
    ax.set_ylabel("estimated effect of T on Y")
    ax.set_title("Adjusting for the third variable: sometimes right, sometimes ruinous")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_collider_illusion(out_path: str | Path, seed: int = 0) -> Path:
    """T and Y are independent — until you condition on their common effect Z."""
    from .dgp import collider
    out_path = Path(out_path)
    sc = collider(n=2000, effect=0.0, seed=seed)
    df = sc.data
    hi = df["Z"] > df["Z"].quantile(0.8)   # look only at high-Z cases

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    axes[0].scatter(df["T"], df["Y"], s=8, alpha=0.3, color="#2e86ab")
    axes[0].set_title(f"Everyone: T and Y independent\n(slope {effect(df):+.2f}, the truth)")
    sub = df[hi]
    axes[1].scatter(sub["T"], sub["Y"], s=10, alpha=0.4, color="#d1495b")
    # fit a line within the stratum to show the spurious slope
    b = np.polyfit(sub["T"], sub["Y"], 1)
    xs = np.linspace(df["T"].min(), df["T"].max(), 50)
    axes[1].plot(xs, np.polyval(b, xs), color="black", linewidth=1.5)
    axes[1].set_title(f"Only high-Z cases: a spurious link\n(slope {b[0]:+.2f} out of nothing)")
    for ax in axes:
        ax.set_xlabel("T"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Y")
    fig.suptitle("The collider illusion: conditioning on a common effect invents a correlation",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_dags(out_path: str | Path) -> Path:
    """Simple node-and-arrow diagrams of the three causal structures."""
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def node(ax, xy, label, color="#2c3e50"):
        ax.annotate(label, xy, ha="center", va="center", fontsize=13, color="white",
                    bbox=dict(boxstyle="circle,pad=0.5", fc=color, ec="none"))

    def arrow(ax, a, b):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555"))

    # Confounder: C up top pointing to T and Y; T -> Y
    ax = axes[0]
    arrow(ax, (0.5, 0.85), (0.2, 0.3)); arrow(ax, (0.5, 0.85), (0.8, 0.3))
    arrow(ax, (0.26, 0.28), (0.74, 0.28))
    node(ax, (0.5, 0.85), "C", "#d1495b"); node(ax, (0.2, 0.25), "T"); node(ax, (0.8, 0.25), "Y")
    ax.set_title("Confounder\nadjust for C  (right)", fontsize=10)

    # Collider: T and Y point down into Z
    ax = axes[1]
    arrow(ax, (0.2, 0.75), (0.5, 0.25)); arrow(ax, (0.8, 0.75), (0.5, 0.25))
    arrow(ax, (0.26, 0.78), (0.74, 0.78))
    node(ax, (0.2, 0.8), "T"); node(ax, (0.8, 0.8), "Y"); node(ax, (0.5, 0.2), "Z", "#d1495b")
    ax.set_title("Collider\ndo NOT adjust for Z  (wrong)", fontsize=10)

    # Mediator: T -> M -> Y
    ax = axes[2]
    arrow(ax, (0.16, 0.5), (0.42, 0.5)); arrow(ax, (0.58, 0.5), (0.84, 0.5))
    node(ax, (0.12, 0.5), "T"); node(ax, (0.5, 0.5), "M", "#d1495b"); node(ax, (0.88, 0.5), "Y")
    ax.set_title("Mediator\nadjusting changes the question", fontsize=10)

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.suptitle("Same data shape, opposite advice — you need the causal graph",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
