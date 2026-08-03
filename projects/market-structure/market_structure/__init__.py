"""market-structure: discover sector groupings from return correlations alone."""

from .cluster import (
    cluster_labels,
    correlation_distance,
    correlation_matrix,
    leaf_order,
    linkage_matrix,
)
from .data import daily_returns, load_prices
from .evaluate import (
    adjusted_rand,
    cluster_composition,
    contingency,
    purity,
)
from . import universe

__all__ = [
    "load_prices",
    "daily_returns",
    "correlation_matrix",
    "correlation_distance",
    "linkage_matrix",
    "cluster_labels",
    "leaf_order",
    "adjusted_rand",
    "purity",
    "contingency",
    "cluster_composition",
    "universe",
]
