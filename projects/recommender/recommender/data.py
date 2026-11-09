"""Load the MovieLens ratings, and split them honestly for evaluation.

The data is the MovieLens "latest-small" set: ~100k ratings (0.5-5 stars) by ~600
users on ~9,700 movies. We cache a compact copy so the study runs offline.

The split is the part that matters. Recommenders are judged on ratings they have
*not* seen, so we hold out a fraction of *each user's* ratings for testing -- never a
global random split, which could leave a user with nothing to learn from. Every test
user is therefore someone the model has already met, just on movies it hasn't.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
_UA = {"User-Agent": "Mozilla/5.0 (recommender research)"}


def fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download MovieLens and return (ratings, movies)."""
    req = urllib.request.Request(URL, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    base = "ml-latest-small"
    with z.open(f"{base}/ratings.csv") as f:
        ratings = pd.read_csv(f)[["userId", "movieId", "rating"]]
    with z.open(f"{base}/movies.csv") as f:
        movies = pd.read_csv(f)[["movieId", "title"]]
    return ratings, movies


def save(ratings: pd.DataFrame, movies: pd.DataFrame, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(data_dir / "ratings.csv.gz", index=False, compression="gzip")
    movies.to_csv(data_dir / "movies.csv", index=False)


def load(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv.gz")
    movies = pd.read_csv(data_dir / "movies.csv")
    return ratings, movies


def train_test_split(ratings: pd.DataFrame, test_frac: float = 0.2,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `test_frac` of each user's ratings for testing."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("userId").groups.items():
        idx = np.asarray(idx)
        n_test = int(round(len(idx) * test_frac))
        if 0 < n_test < len(idx):
            chosen = rng.choice(idx, size=n_test, replace=False)
            test_mask[ratings.index.get_indexer(chosen)] = True
    train = ratings[~test_mask].reset_index(drop=True)
    test = ratings[test_mask].reset_index(drop=True)
    return train, test
