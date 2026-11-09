"""Command-line entry point for the recommender study."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as dt
from . import evaluate as ev
from . import plots
from .models import BiasModel, GlobalMean, ItemMean, MatrixFactorization


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="recommender",
        description="Does matrix factorization actually beat recommending popular "
        "movies? A MovieLens study of ratings, cold start, and ranking.",
    )
    p.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--collect", action="store_true", help="Re-download MovieLens first.")
    p.add_argument("--factors", type=int, default=20, help="Latent factors for MF.")
    p.add_argument("--epochs", type=int, default=15, help="SGD epochs for MF.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading MovieLens...")
        ratings, movies = dt.fetch()
        dt.save(ratings, movies, args.data)
    else:
        ratings, movies = dt.load(args.data)
    titles = dict(zip(movies["movieId"], movies["title"]))

    train, test = dt.train_test_split(ratings, test_frac=0.2, seed=args.seed)
    print(f"\nRecommender: {len(ratings):,} ratings, {ratings['userId'].nunique()} "
          f"users, {ratings['movieId'].nunique()} movies")
    print(f"  {len(train):,} train / {len(test):,} test (per-user holdout)")

    models = {
        "GlobalMean": GlobalMean().fit(train),
        "ItemMean": ItemMean().fit(train),
        "BiasModel": BiasModel().fit(train),
        "MF": MatrixFactorization(n_factors=args.factors, epochs=args.epochs,
                                  seed=args.seed).fit(train),
    }
    rmses = {name: ev.rmse(m, test) for name, m in models.items()}

    print("\nRating-prediction error (RMSE, lower is better):")
    for name, r in rmses.items():
        print(f"  {name:12s} {r:.4f}")
    gap = rmses["BiasModel"] - rmses["MF"]
    print(f"  -> matrix factorization beats the plain bias model by just {gap:.4f} "
          f"RMSE ({gap / rmses['BiasModel']:.1%}).")

    mf = models["MF"]
    print("\nCold start (MF RMSE by how much the user has rated):")
    cold = ev.rmse_by_activity(mf, test, train)
    for _, r in cold.iterrows():
        print(f"  {r['bucket']:18s} {r['rmse']:.3f}")

    recall = ev.recall_at_k(mf, train, test, k=10, sample=200, seed=args.seed)
    print(f"\nTop-10 recall of held-out liked movies ({recall['users_scored']} users):")
    print(f"  matrix factorization: {recall['mf']:.3f}")
    print(f"  most popular:         {recall['popularity']:.3f}")
    verdict = ("popularity wins" if recall["popularity"] > recall["mf"]
               else "matrix factorization wins")
    print(f"  -> {verdict}: predicting ratings well is not the same as recommending "
          f"well.")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    charts = [
        plots.plot_rmse_ladder(rmses, assets / "rmse_ladder.png"),
        plots.plot_cold_start(cold, assets / "cold_start.png"),
        plots.plot_ranking(recall, assets / "ranking.png"),
    ]
    print("\nSaved charts:")
    for c in charts:
        print(f"  {c}")
    print()
