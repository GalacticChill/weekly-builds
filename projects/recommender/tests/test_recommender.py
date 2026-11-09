"""Offline tests — synthetic ratings with known structure, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender import data as dt
from recommender import evaluate as ev
from recommender import models as ml


def _synth(nu=120, ni=60, k=3, noise=0.3, density=0.6, seed=0) -> pd.DataFrame:
    """Ratings generated from known biases and latent factors."""
    rng = np.random.default_rng(seed)
    mu, bu, bi = 3.5, rng.normal(0, 0.5, nu), rng.normal(0, 0.5, ni)
    P, Q = rng.normal(0, 0.5, (nu, k)), rng.normal(0, 0.5, (ni, k))
    rows = []
    for u in range(nu):
        for i in range(ni):
            if rng.random() < density:
                r = mu + bu[u] + bi[i] + P[u] @ Q[i] + rng.normal(0, noise)
                rows.append((u, i, min(5.0, max(0.5, r))))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


# --- split ------------------------------------------------------------------ #

def test_split_keeps_every_user_in_train():
    df = _synth()
    train, test = dt.train_test_split(df, test_frac=0.2, seed=0)
    assert set(test["userId"]).issubset(set(train["userId"]))
    assert len(train) + len(test) == len(df)
    assert 0.15 < len(test) / len(df) < 0.25


# --- baselines -------------------------------------------------------------- #

def test_global_mean_predicts_the_mean():
    df = _synth()
    m = ml.GlobalMean().fit(df)
    assert np.allclose(m.predict(df.head()), df["rating"].mean())


def test_item_mean_shrinks_sparse_items_toward_global():
    # One movie rated once at 5.0; with shrink K it should sit well below 5.
    df = pd.DataFrame({"userId": [1, 2, 3, 4], "movieId": [1, 1, 1, 99],
                       "rating": [3.0, 3.0, 3.0, 5.0]})
    m = ml.ItemMean(shrink=5.0).fit(df)
    pred = m.predict(pd.DataFrame({"userId": [9], "movieId": [99]}))[0]
    assert pred < 4.0 and pred > m.mu            # pulled toward the global mean


def test_bias_model_predictions_are_clipped():
    df = _synth()
    m = ml.BiasModel().fit(df)
    p = m.predict(df)
    assert p.min() >= ml.RATING_MIN and p.max() <= ml.RATING_MAX


# --- the ML claim ----------------------------------------------------------- #

def test_matrix_factorization_beats_baselines_on_structured_data():
    df = _synth(noise=0.3, seed=1)
    train, test = dt.train_test_split(df, test_frac=0.2, seed=1)
    gm = ev.rmse(ml.GlobalMean().fit(train), test)
    bias = ev.rmse(ml.BiasModel().fit(train), test)
    mf = ev.rmse(ml.MatrixFactorization(n_factors=3, epochs=40, seed=1).fit(train), test)
    assert bias < gm                              # biases help
    assert mf < bias                              # latent factors help further


def test_matrix_factorization_handles_unseen_user_and_item():
    train = _synth(seed=2)
    mf = ml.MatrixFactorization(n_factors=3, epochs=5, seed=2).fit(train)
    unseen = pd.DataFrame({"userId": [9999], "movieId": [8888]})
    p = mf.predict(unseen)[0]
    assert ml.RATING_MIN <= p <= ml.RATING_MAX    # falls back to the global mean


def test_recommend_returns_k_items():
    train = _synth(seed=3)
    mf = ml.MatrixFactorization(n_factors=3, epochs=5, seed=3).fit(train)
    recs = mf.recommend(user=0, movie_ids=list(range(40)), k=10)
    assert len(recs) == 10
    assert len(set(recs)) == 10                   # distinct


# --- metrics ---------------------------------------------------------------- #

def test_rmse_hand_value():
    class Const:
        def predict(self, df):
            return np.full(len(df), 3.0)
    test = pd.DataFrame({"userId": [1, 2], "movieId": [1, 2], "rating": [2.0, 4.0]})
    assert np.isclose(ev.rmse(Const(), test), 1.0)   # errors -1, +1 -> RMSE 1


def test_recall_at_k_returns_valid_fractions():
    df = _synth(seed=4)
    train, test = dt.train_test_split(df, test_frac=0.2, seed=4)
    mf = ml.MatrixFactorization(n_factors=3, epochs=10, seed=4).fit(train)
    r = ev.recall_at_k(mf, train, test, k=10, sample=50, seed=4)
    assert 0.0 <= r["mf"] <= 1.0
    assert 0.0 <= r["popularity"] <= 1.0
