"""Structure metrics: hubs, the giant component, and the small-world path length."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def degrees(adj: csr_matrix) -> np.ndarray:
    """Number of direct connections per node."""
    return np.asarray(adj.sum(axis=1)).ravel().astype(int)


def giant_component_fraction(adj: csr_matrix) -> float:
    """Share of nodes in the largest connected component."""
    n = adj.shape[0]
    if n == 0:
        return 0.0
    n_comp, labels = connected_components(adj, directed=False)
    largest = np.bincount(labels).max()
    return largest / n


def degree_distribution(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """(unique degree, count) — the raw material for the log-log tail plot."""
    deg = degrees(adj)
    vals, counts = np.unique(deg, return_counts=True)
    return vals, counts


def average_path_length(adj: csr_matrix, sample: int = 400, seed: int = 0) -> dict:
    """Mean and max shortest-path length (in hops) on the giant component.

    Computed from a random sample of source airports for speed; unweighted, so each
    hop is one flight. The 'six degrees' statistic.
    """
    n_comp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    giant = labels == sizes.argmax()
    giant_idx = np.where(giant)[0]

    rng = np.random.default_rng(seed)
    if len(giant_idx) > sample:
        sources = rng.choice(giant_idx, size=sample, replace=False)
    else:
        sources = giant_idx

    sub = adj[giant_idx][:, giant_idx]
    remap = {orig: k for k, orig in enumerate(giant_idx)}
    src_local = [remap[s] for s in sources]

    dist = shortest_path(sub, method="D", unweighted=True, indices=src_local)
    finite = dist[np.isfinite(dist)]
    finite = finite[finite > 0]
    return {
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "giant_size": int(giant.sum()),
    }
