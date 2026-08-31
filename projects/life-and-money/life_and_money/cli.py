"""Command-line entry point for the life-and-money capstone."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import features as ft
from . import model as ml
from . import plots
from .collect import collect, save

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "indicators.csv"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="life-and-money",
        description="Does money buy a longer life? A cross-country study of what "
        "predicts life expectancy, and which countries beat the odds.",
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to indicators CSV.")
    p.add_argument("--collect", action="store_true",
                   help="Re-download fresh data from the World Bank API first.")
    p.add_argument("--seed", type=int, default=42, help="Train/test split seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading from the World Bank API...")
        raw = collect()
        save(raw, args.data)
        print(f"Saved {len(raw)} countries to {args.data}")
    else:
        raw = ft.load(args.data)

    df = ft.build_features(raw)
    print(f"\nLife & money: {len(df)} countries with life expectancy and income")
    print(f"  Life expectancy range: {df[ft.TARGET].min():.1f} - {df[ft.TARGET].max():.1f} years")

    # 1. The Preston curve.
    preston = ml.preston_fit(df)
    print("\nDoes income buy life linearly, or with diminishing returns?")
    print(f"  Life ~ income        R² = {preston.r2_linear:.2f}")
    print(f"  Life ~ log(income)   R² = {preston.r2_log:.2f}   <- log wins: diminishing returns")
    print(f"  Slope: +{preston.slope:.1f} years of life per 10x of income")

    # 2. Beyond income.
    res = ml.full_fit(df, seed=args.seed)
    print("\nBeyond income, what predicts life expectancy? (out-of-sample)")
    print(f"  Linear model R²:   {res.linear_r2:.2f}")
    print(f"  Random forest R²:  {res.forest_r2:.2f}")
    print("  Top predictors (random forest):")
    for name, imp in res.importances.head(4).items():
        print(f"    {name:24s} {imp:5.1%}")

    # 3. Who beats the odds?
    resid = ml.income_residuals(df)
    print("\nBeats the odds (lives longest vs. what income predicts):")
    for _, r in resid.head(5).iterrows():
        print(f"  {r['country']:28s} {r['residual']:+.1f} years")
    print("Falls short:")
    for _, r in resid.tail(5).iloc[::-1].iterrows():
        print(f"  {r['country']:28s} {r['residual']:+.1f} years")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    notable = ml.income_residuals(df)
    annotate = notable.head(4).index.union(notable.tail(4).index)
    p1 = plots.plot_preston(df, preston, assets / "preston_curve.png",
                            annotate=df.loc[df.index.isin(annotate), ["country"]])
    p2 = plots.plot_importances(res, assets / "importances.png")
    p3 = plots.plot_residuals(resid, assets / "beats_the_odds.png")
    print("\nSaved charts:")
    for path in (p1, p2, p3):
        print(f"  {path}")
    print()
