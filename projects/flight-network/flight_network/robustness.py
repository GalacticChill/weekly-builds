"""Robust yet fragile: remove airports at random vs. biggest-hub-first.

The signature property of hub-dominated networks. Knock out airports in random order
and the network barely notices -- most removals hit tiny airports. Knock them out in
order of connectedness and the whole thing disintegrates within a few percent, as the
hubs that were holding everything together vanish.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .graph import degrees


def _giant_fraction_over(adj: csr_matrix, keep: np.ndarray, n_total: int) -> float:
    """Giant-component share (of the ORIGINAL node count) among kept nodes."""
    if keep.sum() == 0:
        return 0.0
    sub = adj[keep][:, keep]
    _, labels = connected_components(sub, directed=False)
    largest = np.bincount(labels).max() if labels.size else 0
    return largest / n_total


@dataclass
class AttackCurve:
    fraction_removed: np.ndarray
    giant_fraction: np.ndarray
    label: str


def attack(adj: csr_matrix, order: np.ndarray, steps: int = 40,
           label: str = "") -> AttackCurve:
    """Remove nodes in `order`, recording giant-component share at each step."""
    n = adj.shape[0]
    fracs = np.linspace(0.0, 0.6, steps)
    giant = []
    removed_mask = np.zeros(n, dtype=bool)
    for f in fracs:
        k = int(round(f * n))
        removed_mask[:] = False
        removed_mask[order[:k]] = True
        giant.append(_giant_fraction_over(adj, ~removed_mask, n))
    return AttackCurve(fraction_removed=fracs, giant_fraction=np.array(giant),
                       label=label)


def random_attack(adj: csr_matrix, seed: int = 0, steps: int = 40) -> AttackCurve:
    order = np.random.default_rng(seed).permutation(adj.shape[0])
    return attack(adj, order, steps=steps, label="random failures")


def targeted_attack(adj: csr_matrix, steps: int = 40) -> AttackCurve:
    order = np.argsort(-degrees(adj))          # biggest hubs first
    return attack(adj, order, steps=steps, label="targeted (hubs first)")


def collapse_point(curve: AttackCurve, threshold: float = 0.5) -> float:
    """Fraction removed at which the giant component first drops below `threshold`."""
    below = curve.fraction_removed[curve.giant_fraction < threshold]
    return float(below.min()) if below.size else float("nan")
