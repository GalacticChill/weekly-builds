"""Scoring: rating error overall, error by how active a user is, and top-N recall.

RMSE answers "how close are the predicted stars?" But a recommender's real job is
*ranking* -- putting things you'll love at the top of a short list -- so we also ask
whether the model's top-10 actually surfaces the movies a user rated highly in the
held-out set, and whether that beats the one-size-fits-all popularity list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(model, test: pd.DataFrame) -> float:
    pred = model.predict(test)
    return float(np.sqrt(np.mean((pred - test["rating"].to_numpy()) ** 2)))


def rmse_by_activity(model, test: pd.DataFrame, train: pd.DataFrame,
                     edges=(0, 20, 50, 200, 10_000)) -> pd.DataFrame:
    """RMSE split by how many ratings each test user has in training (cold-start)."""
    counts = train.groupby("userId").size()
    test = test.copy()
    test["pred"] = model.predict(test)
    test["n_train"] = test["userId"].map(counts).fillna(0)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (test["n_train"] >= lo) & (test["n_train"] < hi)
        sub = test[m]
        if len(sub) == 0:
            continue
        err = np.sqrt(np.mean((sub["pred"] - sub["rating"]) ** 2))
        rows.append({"bucket": f"{lo}-{hi - 1} ratings", "test_ratings": int(len(sub)),
                     "rmse": float(err)})
    return pd.DataFrame(rows)


def popularity_scores(train: pd.DataFrame) -> dict:
    """Item -> number of ratings (the popularity ranking, same for everyone)."""
    return train.groupby("movieId").size().to_dict()


def recall_at_k(mf, train: pd.DataFrame, test: pd.DataFrame, k: int = 10,
                like: float = 4.0, sample: int = 200, seed: int = 0) -> dict:
    """Top-K recall of held-out *liked* movies: matrix factorization vs popularity.

    For each sampled user we rank every movie they didn't rate in training, take the
    top K, and measure the share of their held-out 4+ star movies that made the cut.
    """
    rng = np.random.default_rng(seed)
    items = np.array(list(mf.imap.keys()))
    pop = popularity_scores(train)
    pop_vec = np.array([pop.get(m, 0) for m in items], dtype=float)

    seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    liked = (test[test["rating"] >= like].groupby("userId")["movieId"]
             .apply(set).to_dict())

    users = [u for u in liked if u in mf.umap]
    if len(users) > sample:
        users = list(rng.choice(users, size=sample, replace=False))

    mf_recall, pop_recall = [], []
    for u in users:
        want = liked[u]
        unseen = ~np.isin(items, list(seen.get(u, set())))
        cand = items[unseen]
        if len(cand) == 0 or not want:
            continue
        # matrix factorization scores for this user over the candidate items
        uu = mf.umap[u]
        mf_all = mf.mu + mf.bi + mf.bu[uu] + mf.Q @ mf.P[uu]
        mf_scores = mf_all[unseen]
        pop_scores = pop_vec[unseen]

        mf_top = set(cand[np.argsort(-mf_scores)[:k]])
        pop_top = set(cand[np.argsort(-pop_scores)[:k]])
        mf_recall.append(len(mf_top & want) / len(want))
        pop_recall.append(len(pop_top & want) / len(want))

    return {
        "k": k, "users_scored": len(mf_recall),
        "mf": float(np.mean(mf_recall)) if mf_recall else float("nan"),
        "popularity": float(np.mean(pop_recall)) if pop_recall else float("nan"),
    }
