"""Turn return correlations into a hierarchy of groups.

The idea in one line: stocks that move together are "close"; stocks that move
independently are "far". We turn the correlation matrix into a proper distance,
build a hierarchical clustering tree from those distances, and cut the tree into a
handful of groups. Nowhere does the algorithm see a sector label — the structure
comes entirely from how prices co-move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of the return columns."""
    return returns.corr()


def correlation_distance(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert correlations to distances via ``d = sqrt(2 (1 - corr))``.

    This is a genuine metric (it's the Euclidean distance between the assets'
    standardized return vectors): perfectly correlated names sit at distance 0,
    uncorrelated names at ~1.41, and perfectly anti-correlated names at 2.
    """
    corr = correlation_matrix(returns).to_numpy()
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    labels = returns.columns
    return pd.DataFrame(dist, index=labels, columns=labels)


def _condensed(dist: pd.DataFrame) -> np.ndarray:
    """Symmetrize a square distance frame and flatten to scipy's condensed form."""
    m = dist.to_numpy(dtype=float)
    m = 0.5 * (m + m.T)          # guard against tiny asymmetries
    np.fill_diagonal(m, 0.0)
    return squareform(m, checks=False)


def linkage_matrix(dist: pd.DataFrame, method: str = "average") -> np.ndarray:
    """Hierarchical-clustering linkage matrix from a distance frame."""
    return linkage(_condensed(dist), method=method)


def cluster_labels(
    dist: pd.DataFrame, k: int, method: str = "average"
) -> pd.Series:
    """Cut the tree into exactly ``k`` groups; return a label per ticker."""
    z = linkage_matrix(dist, method=method)
    labels = fcluster(z, t=k, criterion="maxclust")
    return pd.Series(labels, index=dist.index, name="cluster")


def leaf_order(dist: pd.DataFrame, method: str = "average") -> list[str]:
    """The dendrogram leaf order — the ordering that puts similar names adjacent.

    Reindexing the correlation matrix by this order is what makes the block
    structure pop out in the heatmap.
    """
    z = linkage_matrix(dist, method=method)
    order = leaves_list(z)
    return [dist.index[i] for i in order]
