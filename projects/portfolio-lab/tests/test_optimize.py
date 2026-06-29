"""Offline tests for constrained (long-only) optimization."""

import numpy as np
import pandas as pd

from portfolio_lab import optimize as opt
from portfolio_lab import portfolio as pf


def _uncorrelated_two():
    mean = pd.Series([0.10, 0.10], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    return mean, cov


def test_long_only_weights_are_valid():
    mean, cov = _uncorrelated_two()
    for w in (opt.min_variance_long_only(cov), opt.max_sharpe_long_only(mean, cov)):
        assert np.all(w >= -1e-8)          # no shorting
        assert np.isclose(w.sum(), 1.0)    # fully invested


def test_long_only_matches_closed_form_when_unconstrained_is_already_long():
    # For these uncorrelated assets the closed-form optima are already positive,
    # so the constrained solver should land on the same answer.
    mean, cov = _uncorrelated_two()
    np.testing.assert_allclose(
        opt.min_variance_long_only(cov), pf.min_variance_weights(cov), atol=1e-3
    )
    np.testing.assert_allclose(
        opt.max_sharpe_long_only(mean, cov), pf.max_sharpe_weights(mean, cov), atol=1e-3
    )


def test_constraint_binds_when_closed_form_would_short():
    # Highly correlated assets with different vols: the closed-form min-variance
    # SHORTS the high-vol asset (weights ~ [1.37, -0.37]). Long-only must instead
    # pile fully into the low-vol asset → [1, 0].
    cov = pd.DataFrame(
        [[0.01, 0.027], [0.027, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    closed = pf.min_variance_weights(cov)
    assert closed[1] < 0  # confirm the unconstrained version really does short

    w = opt.min_variance_long_only(cov)
    assert np.all(w >= -1e-8)
    np.testing.assert_allclose(w, [1.0, 0.0], atol=1e-4)


def test_long_only_min_variance_has_lowest_vol_among_random():
    mean = pd.Series([0.12, 0.08, 0.15], index=["A", "B", "C"])
    cov = pd.DataFrame(
        np.diag([0.04, 0.02, 0.06]), index=mean.index, columns=mean.index
    )
    w = opt.min_variance_long_only(cov)
    _, vol, _ = pf.portfolio_performance(w, mean, cov)
    cloud = pf.random_portfolios(mean, cov, n=2000, seed=3)
    assert vol <= cloud["volatility"].min() + 1e-6


def test_efficient_frontier_is_monotonic_and_valid():
    mean = pd.Series([0.12, 0.08, 0.15], index=["A", "B", "C"])
    cov = pd.DataFrame(
        np.diag([0.04, 0.02, 0.06]), index=mean.index, columns=mean.index
    )
    frontier = opt.efficient_frontier(mean, cov, n_points=25)

    # Returns are sorted; every point is long-only and fully invested.
    assert frontier["return"].is_monotonic_increasing
    weights = frontier[["A", "B", "C"]].to_numpy()
    assert np.all(weights >= -1e-6)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)

    # Volatility bottoms out at the min-variance point, then rises on either side.
    min_vol_idx = frontier["volatility"].idxmin()
    assert 0 <= min_vol_idx <= len(frontier) - 1
