"""A simple, honest portfolio backtester.

Portfolio-lab answers "what's the *best* allocation" for a single period. This
answers a different, more realistic question: if you actually *held* an allocation
through time and periodically rebalanced back to it, what would have happened —
after the drift between rebalances and the transaction costs of trading?

The model each day is:

1. Every asset's dollar position grows by that day's return.
2. On a rebalance day, we compute the trades needed to return to the target
   weights, charge a proportional transaction cost on the dollars traded, and
   reset the positions to the target weights on the remaining capital.

Between rebalances the weights are allowed to drift with the market, exactly as a
real buy-and-mostly-hold portfolio would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

# A strategy maps the return history so far -> a target-weight vector.
Strategy = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class BacktestResult:
    """Everything the backtest produced, ready for metrics and plotting."""

    equity: pd.Series          # portfolio value over time (starts at `initial`)
    weights: pd.Series         # weights used (final weights, for strategy runs)
    total_cost: float          # total dollars paid in transaction costs
    n_rebalances: int          # how many times we actually traded
    avg_turnover: float        # mean one-way turnover per rebalance (fraction)
    rebalance: str             # the rebalance rule used
    weights_history: Optional[pd.DataFrame] = None  # weights chosen at each rebalance

    @property
    def returns(self) -> pd.Series:
        """Daily returns of the equity curve."""
        return self.equity.pct_change().dropna()


def _period_first_days(index: pd.DatetimeIndex, rule: str) -> list:
    """The first trading day within each calendar period of `rule`.

    `rule` is a pandas period alias: 'W', 'M', 'Q', 'Y'. 'none' or '' returns [].
    """
    if not rule or rule.lower() == "none":
        return []
    firsts = pd.Series(index, index=index.to_period(rule))
    firsts = firsts.groupby(level=0).first()
    return [pd.Timestamp(d) for d in firsts]


def _rebalance_dates(index: pd.DatetimeIndex, rule: str) -> set:
    """First trading day of each period, excluding the very first day.

    We start already allocated, so the very first date needs no trade.
    """
    dates = set(_period_first_days(index, rule))
    dates.discard(pd.Timestamp(index[0]))
    return dates


def run_backtest(
    returns: pd.DataFrame,
    weights,
    rebalance: str = "M",
    cost: float = 0.001,
    initial: float = 1.0,
) -> BacktestResult:
    """Simulate holding `weights` over `returns`, rebalancing on a schedule.

    Parameters
    ----------
    returns : daily returns, one column per asset.
    weights : target weights (array-like aligned to columns, or a dict/Series).
              Auto-normalized to sum to 1.
    rebalance : 'W', 'M', 'Q', 'Y', or 'none' for buy-and-hold.
    cost : proportional transaction cost per dollar traded (0.001 == 10 bps).
    initial : starting portfolio value.
    """
    tickers = list(returns.columns)

    if isinstance(weights, dict):
        w = np.array([weights[t] for t in tickers], dtype=float)
    elif isinstance(weights, pd.Series):
        w = weights.reindex(tickers).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()

    reb_dates = _rebalance_dates(returns.index, rebalance)

    positions = initial * w            # dollars in each asset
    equity = np.empty(len(returns))
    total_cost = 0.0
    turnovers: list[float] = []

    ret_matrix = returns.to_numpy()
    for i, date in enumerate(returns.index):
        positions = positions * (1.0 + ret_matrix[i])
        value = positions.sum()

        if date in reb_dates and value > 0:
            target = value * w
            trade = target - positions
            traded = np.abs(trade).sum()
            turnovers.append(traded / value)
            fee = cost * traded
            total_cost += fee
            value -= fee
            positions = w * value      # reset to target on post-cost capital

        equity[i] = value

    return BacktestResult(
        equity=pd.Series(equity, index=returns.index, name="equity"),
        weights=pd.Series(w, index=tickers),
        total_cost=float(total_cost),
        n_rebalances=len(turnovers),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        rebalance=rebalance,
    )


def run_strategy_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    rebalance: str = "M",
    cost: float = 0.001,
    initial: float = 1.0,
    warmup: int = 126,
) -> BacktestResult:
    """Backtest a signal-driven strategy that re-chooses weights each rebalance.

    On every rebalance date the strategy is called with the return history
    *strictly before* that date (`returns.iloc[:i]`), so it can never see the
    future — no lookahead bias. The simulation begins at the first rebalance date
    that has at least `warmup` days of history behind it, giving the signal room to
    warm up.

    Parameters
    ----------
    returns : daily returns, one column per asset.
    strategy : callable mapping a return-history DataFrame to a weight vector.
    rebalance : 'W', 'M', 'Q', 'Y'. (Buy-and-hold makes no sense for a strategy.)
    cost, initial : as in `run_backtest`.
    warmup : minimum days of history before the strategy first trades.
    """
    tickers = list(returns.columns)
    index = returns.index
    ret_matrix = returns.to_numpy()

    reb_days = [d for d in _period_first_days(index, rebalance)
                if index.get_loc(d) >= warmup]
    if not reb_days:
        raise ValueError(
            f"Not enough data for warmup={warmup} with rebalance={rebalance!r}."
        )
    reb_set = set(reb_days)
    start = index.get_loc(reb_days[0])

    # Initial weights come from the history available before the start date.
    w = strategy(returns.iloc[:start])
    positions = initial * np.asarray(w, dtype=float)

    equity = np.empty(len(index) - start)
    total_cost = 0.0
    turnovers: list[float] = []
    weights_log: dict[pd.Timestamp, np.ndarray] = {reb_days[0]: np.asarray(w)}

    for offset, i in enumerate(range(start, len(index))):
        date = index[i]
        positions = positions * (1.0 + ret_matrix[i])
        value = positions.sum()

        # Rebalance on period boundaries after the first (which we already set).
        if date in reb_set and date != reb_days[0] and value > 0:
            w = np.asarray(strategy(returns.iloc[:i]), dtype=float)  # past data only
            target = value * w
            trade = target - positions
            traded = np.abs(trade).sum()
            turnovers.append(traded / value)
            fee = cost * traded
            total_cost += fee
            value -= fee
            positions = w * value
            weights_log[date] = w

        equity[offset] = value

    weights_history = pd.DataFrame(weights_log, index=tickers).T
    weights_history.index.name = "date"

    return BacktestResult(
        equity=pd.Series(equity, index=index[start:], name="equity"),
        weights=pd.Series(w, index=tickers),
        total_cost=float(total_cost),
        n_rebalances=len(turnovers),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        rebalance=rebalance,
        weights_history=weights_history,
    )
