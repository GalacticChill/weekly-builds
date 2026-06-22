"""Offline tests: every case uses synthetic data with a hand-checkable answer."""

import numpy as np
import pandas as pd
import pytest

from portfolio_lab import portfolio as pf


def test_normalize_weights_sums_to_one():
    w = pf.normalize_weights(np.array([2.0, 2.0, 4.0]))
    assert np.isclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5])


def test_normalize_weights_rejects_zero_sum():
    with pytest.raises(ValueError):
        pf.normalize_weights(np.array([1.0, -1.0]))


def test_portfolio_performance_single_asset():
    # One asset: portfolio stats equal that asset's stats.
    mean = pd.Series([0.10], index=["A"])
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    ret, vol, sharpe = pf.portfolio_performance([1.0], mean, cov, risk_free=0.02)
    assert np.isclose(ret, 0.10)
    assert np.isclose(vol, 0.2)  # sqrt(0.04)
    assert np.isclose(sharpe, (0.10 - 0.02) / 0.2)


def test_min_variance_weights_known_answer():
    # Uncorrelated assets with variances 0.04 and 0.01.
    # w ∝ Σ⁻¹ 1 = [1/0.04, 1/0.01] = [25, 100] → normalized [0.2, 0.8].
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = pf.min_variance_weights(cov)
    np.testing.assert_allclose(w, [0.2, 0.8])
    assert np.isclose(w.sum(), 1.0)


def test_max_sharpe_weights_known_answer():
    # Equal excess returns, uncorrelated, variances 0.04 and 0.01.
    # w ∝ Σ⁻¹ μ = [0.1/0.04, 0.1/0.01] = [2.5, 10] → normalized [0.2, 0.8].
    mean = pd.Series([0.10, 0.10], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = pf.max_sharpe_weights(mean, cov, risk_free=0.0)
    np.testing.assert_allclose(w, [0.2, 0.8])


def test_max_sharpe_beats_random_portfolios():
    # The tangency portfolio's Sharpe should dominate a random long-only cloud.
    mean = pd.Series([0.12, 0.08, 0.15], index=["A", "B", "C"])
    rng = np.random.default_rng(0)
    cov = pd.DataFrame(
        np.diag([0.04, 0.02, 0.06]), index=mean.index, columns=mean.index
    )
    ms_w = pf.max_sharpe_weights(mean, cov)
    _, _, ms_sharpe = pf.portfolio_performance(ms_w, mean, cov)
    cloud = pf.random_portfolios(mean, cov, n=2000, seed=1)
    assert ms_sharpe >= cloud["sharpe"].max()


def test_random_portfolios_weights_sum_to_one():
    mean = pd.Series([0.1, 0.2], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    cloud = pf.random_portfolios(mean, cov, n=50, seed=7)
    weight_sums = cloud[["A", "B"]].sum(axis=1)
    np.testing.assert_allclose(weight_sums.to_numpy(), 1.0)
    assert len(cloud) == 50


def test_max_drawdown_known_series():
    # Peak 100 → trough 60 is a 40% drawdown, even with a later recovery.
    series = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert np.isclose(pf.max_drawdown(series), -0.5)  # 120 → 60 is the worst


def test_portfolio_growth_equal_weight():
    # Two assets each +10% then +10%; equal weight → same compounding.
    returns = pd.DataFrame({"A": [0.1, 0.1], "B": [0.1, 0.1]})
    growth = pf.portfolio_growth(returns, np.array([0.5, 0.5]))
    np.testing.assert_allclose(growth.to_numpy(), [1.1, 1.21])


def test_daily_returns_drops_first_row():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    r = pf.daily_returns(prices)
    np.testing.assert_allclose(r["A"].to_numpy(), [0.1, 0.1])
    assert len(r) == 2


def test_correlation_matrix_identical_series():
    returns = pd.DataFrame({"A": [0.1, -0.2, 0.3], "B": [0.1, -0.2, 0.3]})
    corr = pf.correlation_matrix(returns)
    assert np.isclose(corr.loc["A", "B"], 1.0)
