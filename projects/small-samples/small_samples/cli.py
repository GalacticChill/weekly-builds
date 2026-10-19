"""Command-line entry point for the small-samples / empirical-Bayes study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import evaluate as ev
from . import plots
from .shrink import empirical_bayes
from .simulate import simulate


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="small-samples",
        description="Why a 100% success rate usually means nothing: empirical-Bayes "
        "shrinkage versus raw rates, judged against known true rates.",
    )
    p.add_argument("--players", type=int, default=1000, help="Number of players.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    df = simulate(n_players=args.players, seed=args.seed)
    shrunk, prior = empirical_bayes(df)

    print(f"\nSmall samples: {len(df)} players, trials ranging "
          f"{df['trials'].min()}-{df['trials'].max()}")
    print(f"  Fitted prior: Beta(a={prior.alpha:.1f}, b={prior.beta:.1f}), "
          f"mean {prior.mean:.3f}, strength {prior.strength:.0f} phantom trials")

    leader = ev.naive_leader(shrunk)
    print("\nThe naive 'best player' (highest raw rate):")
    print(f"  {leader.player}: {leader.successes}/{leader.trials} = "
          f"{leader.raw_rate:.0%}  ->  shrunk to {leader.shrunk_rate:.0%}, "
          f"true rate {leader.true_rate:.0%}")

    print("\nAccuracy vs. the true rates (RMSE, lower is better):")
    print(f"  raw rate:            {ev.rmse(shrunk['raw_rate'], shrunk['true_rate']):.4f}")
    print(f"  shrunken estimate:   {ev.rmse(shrunk['shrunk_rate'], shrunk['true_rate']):.4f}")

    table = ev.rmse_by_sample_size(shrunk)
    print("\n  By sample size:")
    print(f"    {'bucket':16s} {'players':>8s} {'raw':>8s} {'shrunk':>8s} {'better':>8s}")
    for _, r in table.iterrows():
        print(f"    {r['bucket']:16s} {r['players']:8d} {r['raw_rmse']:8.3f} "
              f"{r['shrunk_rmse']:8.3f} {r['improvement']:7.0%}")

    rank = ev.ranking_overlap(shrunk, k=10)
    print(f"\nTop-10 overlap with the true best players:")
    print(f"  by raw rate:        {rank['raw_overlap']:.0%}")
    print(f"  by shrunken rate:   {rank['shrunk_overlap']:.0%}")

    corr = ev.rank_correlation(shrunk)
    print("Rank correlation with the true order (Spearman):")
    print(f"  raw rate:           {corr['raw']:.2f}")
    print(f"  shrunken rate:      {corr['shrunk']:.2f}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_shrinkage(shrunk, prior.mean, assets / "shrinkage.png"),
        plots.plot_rmse_by_size(table, assets / "rmse_by_size.png"),
        plots.plot_examples(shrunk, prior.mean, assets / "examples.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
