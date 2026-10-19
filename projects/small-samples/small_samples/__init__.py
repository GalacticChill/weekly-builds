"""small_samples: why a 100% success rate usually means nothing.

Empirical-Bayes shrinkage of success rates — fit a Beta prior from the whole
population, then pull each player's noisy raw rate toward it in proportion to how
little data backs it up. Scored against known true rates from a simulation.
"""

from .evaluate import (
    naive_leader,
    rank_correlation,
    ranking_overlap,
    rmse,
    rmse_by_sample_size,
)
from .shrink import Prior, empirical_bayes, fit_prior, shrink
from .simulate import prior_mean, simulate

__all__ = [
    "simulate",
    "prior_mean",
    "fit_prior",
    "shrink",
    "empirical_bayes",
    "Prior",
    "rmse",
    "rmse_by_sample_size",
    "ranking_overlap",
    "rank_correlation",
    "naive_leader",
]
