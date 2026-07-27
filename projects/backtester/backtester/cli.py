"""Command-line entry point for the backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from functools import partial

from . import metrics
from .data import daily_returns, load_prices
from .engine import run_backtest, run_strategy_backtest
from .plots import plot_curves, plot_drawdown, plot_equity, plot_param_choices
from .strategies import STRATEGIES, equal_weight
from .walkforward import DEFAULT_GRID, compare


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtester",
        description="Backtest a fixed-weight, periodically-rebalanced portfolio.",
    )
    p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT GOOG")
    p.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Target weights (default: equal weight). Auto-normalized to sum to 1.",
    )
    p.add_argument(
        "--rebalance",
        default="M",
        help="Rebalance frequency: W, M, Q, Y, or none (buy-and-hold). Default M.",
    )
    p.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default=None,
        help="Signal-driven strategy that chooses weights each rebalance "
        "(point-in-time, no lookahead). Omit for a fixed-weight backtest.",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback window in trading days for the strategy signal.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=126,
        help="Days of history required before a strategy first trades. Default 126.",
    )
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Validate the strategy honestly: choose its lookback in-sample only, "
        "roll through time, and score purely out-of-sample. Requires --strategy.",
    )
    p.add_argument(
        "--grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID),
        help="Candidate lookbacks to search over each fold (walk-forward mode).",
    )
    p.add_argument(
        "--train",
        type=int,
        default=504,
        help="Walk-forward training window in trading days. Default 504 (~2y).",
    )
    p.add_argument(
        "--test",
        type=int,
        default=126,
        help="Walk-forward test window in trading days. Default 126 (~6mo).",
    )
    p.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Proportional transaction cost per dollar traded (0.001 = 10 bps).",
    )
    p.add_argument(
        "--risk-free", type=float, default=0.0, help="Annual risk-free rate, e.g. 0.04"
    )
    p.add_argument(
        "--assets-dir", default="assets", help="Directory to write charts into"
    )
    return p.parse_args(argv)


def _print_summary(label: str, res, risk_free: float) -> None:
    s = metrics.summary(res.equity, risk_free)
    print(f"\n{label}  (rebalance={res.rebalance})")
    print(f"  Final value (start 1.0): {res.equity.iloc[-1]:7.3f}")
    print(f"  Total return:           {s['total_return']:7.2%}")
    print(f"  CAGR:                   {s['cagr']:7.2%}")
    print(f"  Annualized volatility:  {s['annualized_volatility']:7.2%}")
    print(f"  Sharpe (rf={risk_free:.0%}):          {s['sharpe']:7.2f}")
    print(f"  Max drawdown:           {s['max_drawdown']:7.2%}")
    print(f"  Rebalances / avg turnover: {res.n_rebalances}  /  {res.avg_turnover:.1%}")
    print(f"  Total transaction cost: {res.total_cost:7.4f}")


def _run_walk_forward(args, returns) -> None:
    """Walk-forward validation: honest OOS vs an in-sample-optimized backtest."""
    strat_fn = STRATEGIES[args.strategy]
    cmp = compare(
        returns, strat_fn, param_values=args.grid, param_name="lookback",
        train=args.train, test=args.test, rebalance=args.rebalance, cost=args.cost,
    )
    wf = cmp.walk_forward

    print(f"\nWalk-forward validation: {args.strategy}")
    print(f"  Grid searched: {args.grid}")
    print(f"  Train / test windows: {args.train} / {args.test} trading days")
    print(f"  Folds: {wf.folds}   (rebalance={args.rebalance})")

    print("\nParameter chosen each fold (from past data only):")
    for _, row in wf.choices.iterrows():
        print(
            f"  {row['test_start'].date()} -> {row['test_end'].date()}   "
            f"lookback={int(row['lookback']):>4}   "
            f"(in-sample Sharpe {row['in_sample_sharpe']:+.2f})"
        )

    honest = wf.oos_sharpe
    claimed = cmp.naive_full_sample_sharpe
    naive_oos = metrics.sharpe_ratio(cmp.naive_oos_equity)
    ew_oos = metrics.sharpe_ratio(cmp.equal_weight_equity)

    print("\nSharpe ratios:")
    print(f"  In-sample-optimized claim (lookback={cmp.naive_param}, full-sample): {claimed:+.2f}")
    print(f"  ...that same parameter, but out-of-sample:                          {naive_oos:+.2f}")
    print(f"  Walk-forward (honest, params chosen from past only):                {honest:+.2f}")
    print(f"  Equal-weight benchmark (out-of-sample):                             {ew_oos:+.2f}")
    print(f"\n  Overfitting tax (claimed - honest): {cmp.overfitting_tax:+.2f}")
    print(f"  Walk-forward OOS total return:      {cmp.walk_forward.equity.iloc[-1] - 1.0:+.2%}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    curves = {
        "Walk-forward (honest)": wf.equity,
        f"In-sample-optimized (lookback={cmp.naive_param})": cmp.naive_oos_equity,
        "Equal-weight": cmp.equal_weight_equity,
    }
    c1 = plot_curves(curves, assets / "walkforward_equity.png",
                     title=f"Walk-forward OOS: {args.strategy}")
    c2 = plot_param_choices(wf.choices, "lookback", assets / "walkforward_params.png")
    print("\nSaved charts:")
    for path in (c1, c2):
        print(f"  {path}")
    print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    prices = load_prices(args.tickers, start=args.start, end=args.end)
    tickers = list(prices.columns)
    returns = daily_returns(prices)

    span = f"{prices.index[0].date()} → {prices.index[-1].date()}"
    print(f"\nBacktest: {', '.join(tickers)}   ({span})")
    print(f"  Transaction cost: {args.cost:.2%} per trade")

    if args.walk_forward:
        if args.strategy is None:
            raise SystemExit("--walk-forward requires --strategy (momentum or inverse-vol).")
        _run_walk_forward(args, returns)
        return

    if args.strategy is not None:
        # Signal-driven mode: the strategy re-chooses weights each rebalance,
        # benchmarked against a point-in-time equal-weight portfolio.
        strat_fn = STRATEGIES[args.strategy]
        if args.lookback is not None:
            strat_fn = partial(strat_fn, lookback=args.lookback)
        test = run_strategy_backtest(
            returns, strat_fn, rebalance=args.rebalance, cost=args.cost,
            warmup=args.warmup,
        )
        bench = run_strategy_backtest(
            returns, equal_weight, rebalance=args.rebalance, cost=args.cost,
            warmup=args.warmup,
        )
        lb = f", lookback={args.lookback}" if args.lookback else ""
        print(f"  Strategy: {args.strategy}{lb}  (point-in-time, no lookahead)")
        test_label = f"{args.strategy} ({args.rebalance})"
        _print_summary(test_label, test, args.risk_free)
        _print_summary(f"Equal-weight ({args.rebalance})", bench, args.risk_free)
        results = {test_label: test, f"Equal-weight ({args.rebalance})": bench}
    else:
        # Fixed-weight mode: chosen weights vs. buy-and-hold of the same weights.
        if args.weights is None:
            weights = np.repeat(1.0 / len(tickers), len(tickers))
        else:
            if len(args.weights) != len(tickers):
                raise SystemExit(
                    f"Got {len(args.weights)} weights for {len(tickers)} tickers."
                )
            weights = np.array(args.weights)

        test = run_backtest(returns, weights, rebalance=args.rebalance, cost=args.cost)
        hold = run_backtest(returns, weights, rebalance="none", cost=args.cost)
        wtxt = ", ".join(f"{t} {w:.0%}" for t, w in zip(tickers, test.weights))
        print(f"  Target weights: {wtxt}")
        test_label = f"Rebalanced ({args.rebalance})"
        _print_summary(test_label, test, args.risk_free)
        _print_summary("Buy & hold", hold, args.risk_free)
        results = {test_label: test, "Buy & hold": hold}

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    eq = plot_equity(results, assets / "equity.png")
    dd = plot_drawdown(results, assets / "drawdown.png")

    print("\nSaved charts:")
    for path in (eq, dd):
        print(f"  {path}")
    print()
