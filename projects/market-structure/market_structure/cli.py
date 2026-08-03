"""Command-line entry point: cluster the market and score it against sectors."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import cluster, evaluate, universe
from .data import daily_returns, load_prices
from .plots import plot_correlation_heatmap, plot_dendrogram


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="market-structure",
        description="Discover market structure by clustering stocks on return "
        "correlation, then compare the groups to real sectors.",
    )
    p.add_argument(
        "tickers",
        nargs="*",
        default=None,
        help="Tickers to cluster (default: the built-in labeled universe).",
    )
    p.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters to cut the tree into (default: number of sectors).",
    )
    p.add_argument(
        "--method",
        default="average",
        choices=["average", "complete", "single", "ward"],
        help="Hierarchical linkage method. Default: average.",
    )
    p.add_argument(
        "--assets-dir", default="assets", help="Directory to write charts into"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    use_universe = not args.tickers
    tickers = universe.tickers() if use_universe else list(args.tickers)
    k = args.clusters or (universe.n_sectors() if use_universe else 4)

    prices = load_prices(tickers, start=args.start, end=args.end)
    tickers = list(prices.columns)
    returns = daily_returns(prices)

    span = f"{prices.index[0].date()} -> {prices.index[-1].date()}"
    print(f"\nMarket structure: {len(tickers)} tickers   ({span})")
    print(f"  Clustering on daily-return correlation, k={k}, method={args.method}")

    corr = cluster.correlation_matrix(returns)
    dist = cluster.correlation_distance(returns)
    z = cluster.linkage_matrix(dist, method=args.method)
    labels = cluster.cluster_labels(dist, k=k, method=args.method)
    order = cluster.leaf_order(dist, method=args.method)

    if use_universe:
        true = universe.sectors_for(tickers)
        pred = labels.loc[tickers].tolist()
        ari = evaluate.adjusted_rand(true, pred)
        pur = evaluate.purity(true, pred)
        print("\nAgreement with real sectors (never shown to the algorithm):")
        print(f"  Adjusted Rand index: {ari:5.2f}   (1 = perfect, 0 = chance)")
        print(f"  Purity:              {pur:5.2f}")

        print("\nDiscovered clusters vs. true sectors:")
        table = evaluate.contingency(true, pred)
        print(table.to_string())

        print("\nWhere the algorithm and reality disagree:")
        comp = evaluate.cluster_composition(tickers, true, pred)
        # Flag names whose cluster's majority sector isn't their own.
        majority = (
            comp.groupby("cluster")["sector"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        odd = comp[comp.apply(lambda r: majority[r["cluster"]] != r["sector"], axis=1)]
        if len(odd):
            for _, r in odd.iterrows():
                print(f"  {r['ticker']:5s} is {r['sector']:12s} but clustered with "
                      f"cluster {r['cluster']} (mostly {majority[r['cluster']]})")
        else:
            print("  (none — every name landed with its own sector)")
    else:
        true = [str(labels.loc[t]) for t in tickers]  # no ground truth available

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    sectors_for_plot = universe.sectors_for(tickers) if use_universe else \
        [f"cluster {labels.loc[t]}" for t in tickers]
    d = plot_dendrogram(z, tickers, sectors_for_plot, assets / "dendrogram.png")
    h = plot_correlation_heatmap(corr, order, assets / "heatmap.png")

    print("\nSaved charts:")
    for path in (d, h):
        print(f"  {path}")
    print()
