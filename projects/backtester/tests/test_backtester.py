"""Offline tests: synthetic returns with hand-checkable answers."""

import numpy as np
import pandas as pd
import pytest

from backtester import metrics
from backtester.engine import run_backtest


def _index(n: int) -> pd.DatetimeIndex:
    """A run of `n` business days starting 2020-01-01."""
    return pd.bdate_range("2020-01-01", periods=n)


def test_single_asset_equity_is_compounded_returns():
    # 100% in one asset: no trades ever, so equity is just compounded returns.
    r = pd.DataFrame({"A": np.full(60, 0.01)}, index=_index(60))
    res = run_backtest(r, [1.0], rebalance="M", cost=0.01)
    expected = np.cumprod(1.0 + r["A"].to_numpy())
    np.testing.assert_allclose(res.equity.to_numpy(), expected)
    assert res.total_cost == 0.0  # single asset never needs rebalancing trades


def test_zero_returns_stay_flat_with_no_cost():
    r = pd.DataFrame({"A": np.zeros(60), "B": np.zeros(60)}, index=_index(60))
    res = run_backtest(r, [0.5, 0.5], rebalance="M", cost=0.01)
    np.testing.assert_allclose(res.equity.to_numpy(), 1.0)
    assert res.total_cost == 0.0  # weights never drift, so no trades


def test_buy_and_hold_has_no_rebalances_or_cost():
    r = pd.DataFrame(
        {"A": np.full(80, 0.01), "B": np.full(80, -0.002)}, index=_index(80)
    )
    res = run_backtest(r, [0.5, 0.5], rebalance="none", cost=0.01)
    assert res.n_rebalances == 0
    assert res.total_cost == 0.0
    # Buy-and-hold equity is just the blended growth of the two sleeves.
    grow_a = np.cumprod(1.0 + r["A"].to_numpy())
    grow_b = np.cumprod(1.0 + r["B"].to_numpy())
    expected = 0.5 * grow_a + 0.5 * grow_b
    np.testing.assert_allclose(res.equity.to_numpy(), expected)


def test_transaction_costs_reduce_final_value():
    # Diverging assets force rebalancing trades; more cost => less money.
    r = pd.DataFrame(
        {"A": np.full(120, 0.01), "B": np.full(120, -0.005)}, index=_index(120)
    )
    free = run_backtest(r, [0.5, 0.5], rebalance="M", cost=0.0)
    pricey = run_backtest(r, [0.5, 0.5], rebalance="M", cost=0.01)
    assert pricey.total_cost > 0
    assert pricey.equity.iloc[-1] < free.equity.iloc[-1]
    assert free.total_cost == 0.0


def test_rebalance_count_matches_month_boundaries():
    # ~4 months of business days => 3 rebalances after the starting month.
    r = pd.DataFrame({"A": np.zeros(85), "B": np.zeros(85)}, index=_index(85))
    res = run_backtest(r, [0.6, 0.4], rebalance="M", cost=0.0)
    months = r.index.to_period("M").nunique()
    assert res.n_rebalances == months - 1


def test_weights_are_normalized():
    r = pd.DataFrame({"A": np.zeros(30), "B": np.zeros(30)}, index=_index(30))
    res = run_backtest(r, [2.0, 2.0], rebalance="none")
    np.testing.assert_allclose(res.weights.to_numpy(), [0.5, 0.5])


def test_weights_sum_zero_raises():
    r = pd.DataFrame({"A": np.zeros(10), "B": np.zeros(10)}, index=_index(10))
    with pytest.raises(ValueError):
        run_backtest(r, [1.0, -1.0])


# ---- metrics ----

def test_cagr_known_value():
    # A curve that doubles over exactly one trading year has CAGR = 100%.
    # 253 points => 252 steps => 1.0 years; endpoints 1.0 and 2.0.
    r = 2.0 ** (1.0 / 252) - 1.0
    equity = pd.Series((1.0 + r) ** np.arange(253))
    assert np.isclose(equity.iloc[0], 1.0) and np.isclose(equity.iloc[-1], 2.0)
    assert np.isclose(metrics.cagr(equity), 1.0, atol=1e-3)


def test_max_drawdown_known_series():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert np.isclose(metrics.max_drawdown(equity), -0.5)  # 120 -> 60


def test_summary_has_all_headline_metrics():
    equity = pd.Series(np.cumprod(np.full(50, 1.001)))
    s = metrics.summary(equity)
    assert set(s) == {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
    }
