"""Command-line entry point for the forecastability study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as dt
from . import forecast as fc
from . import plots

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "co2.csv"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="forecastable",
        description="What can you actually forecast? A predictable series (CO2) vs a "
        "random walk, scored honestly on the future.",
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to co2.csv")
    p.add_argument("--collect", action="store_true",
                   help="Re-download the CO2 series from NOAA first.")
    p.add_argument("--test-len", type=int, default=60,
                   help="Months of future to hold out and forecast. Default 60 (5y).")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def _report(card) -> None:
    print(f"\n{card.name}: forecasting {card.horizon} months ahead")
    print(f"  {'method':16s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>8s}")
    for m in card.mae:
        print(f"  {m:16s} {card.mae[m]:10.3f} {card.rmse[m]:10.3f} {card.mape[m]:8.2f}")
    print(f"  best: {card.best}")


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading CO2 from NOAA Mauna Loa...")
        co2 = dt.fetch_co2()
        dt.save(co2, args.data)
        print(f"Saved {len(co2)} months to {args.data}")
    else:
        path = Path(args.data)
        co2 = dt.load(path) if path.exists() else dt.fetch_co2()

    # A representative random walk (typical of the 300-seed average below), not a
    # cherry-picked lucky one.
    walk = dt.random_walk(n=len(co2), seed=6)

    co2_card = fc.evaluate(co2, "CO2 (real: trend + season)", args.test_len)
    walk_card = fc.evaluate(walk, "Random walk (one example)", args.test_len)
    cards = {co2_card.name: co2_card, walk_card.name: walk_card}
    for card in cards.values():
        _report(card)

    # The honest random-walk claim comes from many walks, not the one plotted above.
    mean_mae, model_win_rate = fc.random_walk_experiment(n_seeds=300, n=len(co2),
                                                         test_len=args.test_len)
    print("\nRandom walk, averaged over 300 seeds (the honest test):")
    print(f"  {'method':16s} {'mean MAE':>10s}")
    for m, v in sorted(mean_mae.items(), key=lambda kv: kv[1]):
        print(f"  {m:16s} {v:10.3f}")

    print("\nThe lesson:")
    print(f"  On CO2, the trend+seasonal model beats naive-last by "
          f"{co2_card.mae['naive-last'] / co2_card.mae['trend+seasonal']:.0f}x — "
          f"real, stable structure really is predictable.")
    best_rw = min(mean_mae, key=mean_mae.get)
    print(f"  On random walks, the lowest average error is '{best_rw}', and the "
          f"flexible model beats naive only {model_win_rate:.0%} of the time — "
          f"no reliable edge exists to find.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_forecast(co2, co2_card, assets / "forecast_co2.png"),
        plots.plot_forecast(walk, walk_card, assets / "forecast_random_walk.png"),
        plots.plot_error_comparison(cards, assets / "error_comparison.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
