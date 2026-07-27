"""Walk-forward validation: pick strategy parameters *honestly*.

A backtest that chooses its parameters by looking at the whole history is telling
you about the past, not the future. If you try six momentum lookbacks and report
the one with the best Sharpe, you've *searched* for a number — some of that Sharpe
is skill, and some is just the luckiest draw out of six. Walk-forward validation
removes the hindsight:

1. Slide a rolling **train** window through time.
2. In each train window, score every candidate parameter and pick the best one —
   using *only* data inside that window.
3. Apply that choice to the **test** window that immediately follows, which the
   selection never saw. Record those out-of-sample (OOS) returns.
4. Slide forward by one test window and repeat. The test windows tile the timeline
   with no gaps and no overlap, so stitching their returns gives one continuous,
   honest OOS equity curve.

The gap between what an in-sample-optimized backtest *claims* and what walk-forward
actually *delivers* is the overfitting tax — the single most useful number a
backtest can give you about itself.

Everything here reuses the same point-in-time simulation as the engine: on every
rebalance the strategy sees returns *strictly before* that day, never after.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics
from .engine import _period_first_days, run_strategy_backtest
from .strategies import equal_weight

Strategy = Callable[..., np.ndarray]

# Default lookbacks to search over, in trading days: ~1mo, 2mo, 3mo, 6mo, 9mo, 1y.
DEFAULT_GRID = (21, 42, 63, 126, 189, 252)


def _apply(strategy_fn: Strategy, param_name: str, value) -> Strategy:
    """Bind a parameter value onto a strategy (or leave it untouched if None)."""
    if value is None:
        return strategy_fn
    return partial(strategy_fn, **{param_name: value})


def _normalize(w: np.ndarray, n: int) -> np.ndarray:
    """Coerce a weight vector to a clean long-only simplex point.

    Falls back to equal weight if the strategy returned nothing usable.
    """
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.repeat(1.0 / n, n)
    return w / total


def _segment_returns(
    returns: pd.DataFrame,
    strategy_fn: Strategy,
    start: int,
    end: int,
    rebalance: str,
    cost: float,
) -> pd.Series:
    """Daily portfolio returns over index positions ``[start, end)``.

    Weights entering the segment come from history *before* ``start``; on each
    rebalance boundary inside the segment the strategy re-chooses from history
    *strictly before* that day. Returned as daily returns (not equity) so segments
    from different folds can be concatenated and compounded seamlessly.
    """
    index = returns.index
    ret_matrix = returns.to_numpy()
    n_assets = returns.shape[1]

    reb_positions = {index.get_loc(d) for d in _period_first_days(index, rebalance)}
    reb_positions = {p for p in reb_positions if start < p < end}

    w = _normalize(strategy_fn(returns.iloc[:start]), n_assets)
    positions = w.copy()          # value enters the segment at 1.0
    prev_value = 1.0
    daily = np.empty(end - start)

    for offset, i in enumerate(range(start, end)):
        positions = positions * (1.0 + ret_matrix[i])
        value = positions.sum()

        if i in reb_positions and value > 0:
            w = _normalize(strategy_fn(returns.iloc[:i]), n_assets)  # past data only
            target = value * w
            traded = np.abs(target - positions).sum()
            value -= cost * traded
            positions = w * value

        daily[offset] = value / prev_value - 1.0
        prev_value = value

    return pd.Series(daily, index=index[start:end], name="oos_return")


@dataclass
class WalkForwardResult:
    """The honest, stitched-together out-of-sample record."""

    equity: pd.Series          # compounded OOS equity curve (starts at 1.0)
    oos_returns: pd.Series     # daily OOS returns
    choices: pd.DataFrame      # one row per fold: spans, chosen param, in-sample score
    param_name: str
    folds: int

    @property
    def oos_sharpe(self) -> float:
        return metrics.sharpe_ratio(self.equity)


def walk_forward(
    returns: pd.DataFrame,
    strategy_fn: Strategy,
    param_values: Sequence = DEFAULT_GRID,
    param_name: str = "lookback",
    train: int = 504,
    test: int = 126,
    rebalance: str = "M",
    cost: float = 0.001,
    warmup: Optional[int] = None,
) -> WalkForwardResult:
    """Roll a train/test split through time, choosing parameters in-sample only.

    Parameters
    ----------
    returns : daily returns, one column per asset.
    strategy_fn : a strategy taking ``(history, <param_name>=value)``.
    param_values : the candidate parameter values to search over each fold.
    param_name : the keyword the parameter is passed as (default ``"lookback"``).
    train, test : rolling window lengths in trading days. Test windows tile the
        timeline; the first OOS day is at position ``train``.
    rebalance : 'W', 'M', 'Q', 'Y'.
    cost : proportional transaction cost per dollar traded.
    warmup : days of history a candidate needs before it trades *inside a train
        window*. Defaults to the longest lookback so every candidate is valid.
    """
    param_values = list(param_values)
    if not param_values:
        raise ValueError("Need at least one candidate parameter value.")
    if warmup is None:
        warmup = max(int(v) for v in param_values)
    if train <= warmup:
        raise ValueError(
            f"train ({train}) must exceed warmup/longest lookback ({warmup})."
        )

    n = len(returns)
    if n < train + 1:
        raise ValueError(
            f"Need at least train+1 = {train + 1} rows; got {n}."
        )

    fold_returns: list[pd.Series] = []
    rows: list[dict] = []

    t0 = 0
    while t0 + train + 1 <= n:
        tr0, tr1 = t0, t0 + train
        te0, te1 = tr1, min(tr1 + test, n)
        train_slice = returns.iloc[tr0:tr1]

        # Score every candidate on the train window; keep the best by Sharpe.
        best_score, best_value = -np.inf, param_values[0]
        for v in param_values:
            fn = _apply(strategy_fn, param_name, v)
            try:
                res = run_strategy_backtest(
                    train_slice, fn, rebalance=rebalance, cost=cost, warmup=warmup
                )
            except ValueError:
                continue
            score = metrics.sharpe_ratio(res.equity)
            score = -np.inf if not np.isfinite(score) else score
            if score > best_score:
                best_score, best_value = score, v

        fn = _apply(strategy_fn, param_name, best_value)
        seg = _segment_returns(returns, fn, te0, te1, rebalance, cost)
        fold_returns.append(seg)
        rows.append(
            {
                "train_start": returns.index[tr0],
                "train_end": returns.index[tr1 - 1],
                "test_start": returns.index[te0],
                "test_end": returns.index[te1 - 1],
                param_name: best_value,
                "in_sample_sharpe": best_score if np.isfinite(best_score) else np.nan,
            }
        )
        t0 += test

    oos_returns = pd.concat(fold_returns)
    equity = (1.0 + oos_returns).cumprod()
    equity.name = "equity"
    choices = pd.DataFrame(rows)

    return WalkForwardResult(
        equity=equity,
        oos_returns=oos_returns,
        choices=choices,
        param_name=param_name,
        folds=len(rows),
    )


class FixedParamFit(NamedTuple):
    """A single parameter chosen by looking at the *whole* sample (hindsight)."""

    param: object
    full_sample_sharpe: float   # the number a naive backtest would proudly report
    equity: pd.Series           # its point-in-time equity over the full sample


def best_fixed_param(
    returns: pd.DataFrame,
    strategy_fn: Strategy,
    param_values: Sequence = DEFAULT_GRID,
    param_name: str = "lookback",
    rebalance: str = "M",
    cost: float = 0.001,
    warmup: Optional[int] = None,
) -> FixedParamFit:
    """Pick the single parameter with the best Sharpe over the *entire* history.

    This is exactly the tempting, dishonest thing walk-forward avoids: the winner
    was chosen with full knowledge of the period it's scored on. Kept here so we can
    quantify how much of its shine survives out of sample.
    """
    param_values = list(param_values)
    if warmup is None:
        warmup = max(int(v) for v in param_values)

    best: Optional[FixedParamFit] = None
    for v in param_values:
        fn = _apply(strategy_fn, param_name, v)
        res = run_strategy_backtest(
            returns, fn, rebalance=rebalance, cost=cost, warmup=warmup
        )
        sharpe = metrics.sharpe_ratio(res.equity)
        sharpe = -np.inf if not np.isfinite(sharpe) else sharpe
        if best is None or sharpe > best.full_sample_sharpe:
            best = FixedParamFit(v, sharpe, res.equity)
    assert best is not None
    return best


@dataclass
class Comparison:
    """Three curves over the same out-of-sample span, plus the headline gap."""

    walk_forward: WalkForwardResult
    naive_oos_equity: pd.Series     # in-sample-optimized param, but shown OOS
    equal_weight_equity: pd.Series  # the humble 1/N benchmark, OOS
    naive_param: object
    naive_full_sample_sharpe: float  # what the naive backtest would have claimed

    @property
    def overfitting_tax(self) -> float:
        """Claimed (full-sample, hindsight) Sharpe minus honest OOS Sharpe."""
        return self.naive_full_sample_sharpe - self.walk_forward.oos_sharpe


def compare(
    returns: pd.DataFrame,
    strategy_fn: Strategy,
    param_values: Sequence = DEFAULT_GRID,
    param_name: str = "lookback",
    train: int = 504,
    test: int = 126,
    rebalance: str = "M",
    cost: float = 0.001,
    warmup: Optional[int] = None,
) -> Comparison:
    """Put honesty and hindsight side by side over one identical OOS span.

    Returns three equity curves aligned on the walk-forward OOS window:
    the honest walk-forward curve, the curve of the parameter an analyst would have
    picked with full-sample hindsight, and a plain equal-weight benchmark.
    """
    param_values = list(param_values)
    if warmup is None:
        warmup = max(int(v) for v in param_values)

    wf = walk_forward(
        returns, strategy_fn, param_values, param_name=param_name,
        train=train, test=test, rebalance=rebalance, cost=cost, warmup=warmup,
    )

    oos_start, oos_end = train, len(returns)
    naive = best_fixed_param(
        returns, strategy_fn, param_values, param_name=param_name,
        rebalance=rebalance, cost=cost, warmup=warmup,
    )
    naive_fn = _apply(strategy_fn, param_name, naive.param)
    naive_oos = _segment_returns(returns, naive_fn, oos_start, oos_end, rebalance, cost)
    ew_oos = _segment_returns(returns, equal_weight, oos_start, oos_end, rebalance, cost)

    return Comparison(
        walk_forward=wf,
        naive_oos_equity=(1.0 + naive_oos).cumprod().rename("equity"),
        equal_weight_equity=(1.0 + ew_oos).cumprod().rename("equity"),
        naive_param=naive.param,
        naive_full_sample_sharpe=naive.full_sample_sharpe,
    )
