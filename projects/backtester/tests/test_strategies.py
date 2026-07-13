"""Offline tests for signal-driven strategies and the point-in-time engine."""

import numpy as np
import pandas as pd

from backtester import strategies as st
from backtester.engine import run_strategy_backtest


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_equal_weight_is_uniform():
    hist = pd.DataFrame(np.zeros((10, 4)), columns=list("ABCD"), index=_index(10))
    np.testing.assert_allclose(st.equal_weight(hist), [0.25, 0.25, 0.25, 0.25])


def test_momentum_favours_the_winner_and_zeros_losers():
    # A rises steadily, B falls steadily: all weight should go to A.
    hist = pd.DataFrame(
        {"A": np.full(150, 0.01), "B": np.full(150, -0.01)}, index=_index(150)
    )
    w = st.momentum(hist, lookback=100)
    np.testing.assert_allclose(w, [1.0, 0.0], atol=1e-12)


def test_momentum_falls_back_to_equal_when_all_negative():
    hist = pd.DataFrame(
        {"A": np.full(150, -0.01), "B": np.full(150, -0.02)}, index=_index(150)
    )
    w = st.momentum(hist, lookback=100)
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_inverse_volatility_known_weights():
    # A has twice the volatility of B (alternating ±0.02 vs ±0.01).
    # Inverse-vol weights ∝ [1/0.02, 1/0.01] = [50, 100] → [1/3, 2/3].
    a = np.tile([0.02, -0.02], 40)
    b = np.tile([0.01, -0.01], 40)
    hist = pd.DataFrame({"A": a, "B": b}, index=_index(80))
    w = st.inverse_volatility(hist, lookback=80)
    np.testing.assert_allclose(w, [1 / 3, 2 / 3], atol=1e-6)


def test_strategy_weights_always_sum_to_one():
    rng = np.random.default_rng(0)
    hist = pd.DataFrame(
        rng.normal(0, 0.01, size=(200, 3)), columns=list("ABC"), index=_index(200)
    )
    for fn in (st.equal_weight, st.momentum, st.inverse_volatility):
        w = fn(hist)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= -1e-12)


def test_engine_never_looks_ahead():
    # A spy strategy records the last date it was shown on each call. Every one
    # must lie strictly before the rebalance date the engine used it for.
    rng = np.random.default_rng(1)
    idx = _index(400)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(400, 2)), columns=["A", "B"], index=idx
    )

    seen_last_dates: list[pd.Timestamp] = []

    def spy(history: pd.DataFrame) -> np.ndarray:
        if len(history) > 0:
            seen_last_dates.append(history.index[-1])
        return np.array([0.5, 0.5])

    res = run_strategy_backtest(returns, spy, rebalance="M", warmup=60)
    reb_dates = list(res.weights_history.index)

    # The first weight vector is set from pre-start history; subsequent rebalances
    # each see only data strictly before their date.
    for seen, reb_date in zip(seen_last_dates[1:], reb_dates[1:]):
        assert seen < reb_date


def test_strategy_backtest_runs_and_is_well_formed():
    rng = np.random.default_rng(2)
    idx = _index(400)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(400, 3)), columns=list("ABC"), index=idx
    )
    res = run_strategy_backtest(returns, st.momentum, rebalance="M", warmup=126)
    assert res.equity.index[0] >= idx[126]
    assert len(res.equity) == (len(idx) - returns.index.get_loc(res.equity.index[0]))
    # Every recorded weight vector is a valid long-only allocation.
    w = res.weights_history.to_numpy()
    assert np.all(w >= -1e-9)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
