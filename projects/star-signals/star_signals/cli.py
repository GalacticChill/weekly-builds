"""Command-line entry point for star-signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import features as ft
from . import model as ml
from .collect import collect, save
from .plots import plot_fit, plot_group_split, plot_importances

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "repos.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="star_signals",
        description="Does GitHub reward substance or surface signal? A study of "
        "what actually predicts a repository's stars.",
    )
    p.add_argument(
        "--data", default=str(DEFAULT_DATA), help="CSV of collected repos to analyze."
    )
    p.add_argument(
        "--collect",
        action="store_true",
        help="Re-fetch data from the GitHub API and overwrite --data before analyzing.",
    )
    p.add_argument(
        "--per-query",
        type=int,
        default=100,
        help="Repos to fetch per (language, star-bucket) when collecting.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    p.add_argument("--assets-dir", default="assets", help="Where to write charts.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Collecting repositories from the GitHub API ...")
        raw = collect(per_query=args.per_query)
        save(raw, args.data)
        print(f"  saved {len(raw)} repos to {args.data}")
    else:
        raw = pd.read_csv(args.data)

    data = ft.build_features(raw)
    res = ml.fit(data, seed=args.seed)

    print(f"\nStar-signals: {len(data)} repos analyzed")
    print(f"  Stars range: {int(data['stars'].min())} – {int(data['stars'].max())}")
    print("\nOut-of-sample fit (predicting log-stars on unseen repos):")
    print(f"  Linear model R²:   {res.linear_r2:6.2f}")
    print(f"  Random forest R²:  {res.forest_r2:6.2f}")

    print("\nShare of predictive power (random forest):")
    for group in ("substance", "signal"):
        print(f"  {group.capitalize():10s} {res.group_importance.get(group, 0):.0%}")

    print("\nTop features by importance:")
    for feat, imp in res.importances.head(6).items():
        print(f"  {feat:18s} {imp:5.1%}   ({ft.group_of(feat)})")

    print(f"\nVerdict: {res.verdict}.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_importances(res, assets / "importances.png"),
        plot_group_split(res, assets / "signal_vs_substance.png"),
        plot_fit(res, assets / "fit.png"),
    ]
    print("\nSaved charts:")
    for path in paths:
        print(f"  {path}")
    print()
