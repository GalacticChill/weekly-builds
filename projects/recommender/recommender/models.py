"""Four predictors, from the humblest baseline to matrix factorization.

Each learns from the training ratings and predicts a rating for any (user, movie)
pair, falling back gracefully when it has never seen the user or the movie. The point
of lining them up is to see how much each extra layer of sophistication actually buys
over the one before it.

- `GlobalMean`   — predict the overall average rating. The floor.
- `ItemMean`     — each movie's average, shrunk toward the global mean for movies with
                   few ratings (the same small-sample fix as shrinkage).
- `BiasModel`    — global + a per-user offset + a per-movie offset. Captures "this
                   user rates high" and "this movie is loved" without any interaction.
- `MatrixFactorization` — the bias model plus learned latent factors (SGD), so it can
                   model *taste*: which users like which kinds of movies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATING_MIN, RATING_MAX = 0.5, 5.0


def _index(ids) -> dict:
    return {v: k for k, v in enumerate(pd.unique(ids))}


class GlobalMean:
    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu)


class ItemMean:
    """Per-movie mean shrunk toward the global mean (K phantom average ratings)."""

    def __init__(self, shrink: float = 5.0):
        self.shrink = shrink

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        g = train.groupby("movieId")["rating"]
        s, c = g.sum(), g.count()
        self.item = ((s + self.shrink * self.mu) / (c + self.shrink)).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.item.get(i, self.mu) for i in df["movieId"]])


class BiasModel:
    """rating ~ mu + user_bias + item_bias, with damped (regularized) estimates."""

    def __init__(self, reg_user: float = 10.0, reg_item: float = 10.0):
        self.reg_user, self.reg_item = reg_user, reg_item

    def fit(self, train: pd.DataFrame):
        self.mu = float(train["rating"].mean())
        g = train.groupby("movieId")["rating"]
        self.bi = ((g.sum() - g.count() * self.mu) / (g.count() + self.reg_item)).to_dict()
        resid = train["rating"] - self.mu - train["movieId"].map(self.bi).fillna(0.0)
        bu_num = resid.groupby(train["userId"]).sum()
        bu_den = train.groupby("userId")["rating"].count() + self.reg_user
        self.bu = (bu_num / bu_den).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        bu = df["userId"].map(self.bu).fillna(0.0).to_numpy()
        bi = df["movieId"].map(self.bi).fillna(0.0).to_numpy()
        return np.clip(self.mu + bu + bi, RATING_MIN, RATING_MAX)


class MatrixFactorization:
    """Biases + latent factors trained by stochastic gradient descent."""

    def __init__(self, n_factors: int = 20, lr: float = 0.01, reg: float = 0.05,
                 epochs: int = 20, seed: int = 0):
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.epochs, self.seed = epochs, seed

    def fit(self, train: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        self.umap, self.imap = _index(train["userId"]), _index(train["movieId"])
        nu, ni = len(self.umap), len(self.imap)
        self.mu = float(train["rating"].mean())
        self.bu = np.zeros(nu)
        self.bi = np.zeros(ni)
        self.P = rng.normal(0, 0.05, (nu, self.n_factors))
        self.Q = rng.normal(0, 0.05, (ni, self.n_factors))

        u = train["userId"].map(self.umap).to_numpy()
        i = train["movieId"].map(self.imap).to_numpy()
        r = train["rating"].to_numpy(dtype=float)
        lr, reg = self.lr, self.reg

        for _ in range(self.epochs):
            order = rng.permutation(len(r))
            for n in order:
                uu, ii = u[n], i[n]
                pred = self.mu + self.bu[uu] + self.bi[ii] + self.P[uu] @ self.Q[ii]
                e = r[n] - pred
                self.bu[uu] += lr * (e - reg * self.bu[uu])
                self.bi[ii] += lr * (e - reg * self.bi[ii])
                puu = self.P[uu].copy()
                self.P[uu] += lr * (e * self.Q[ii] - reg * self.P[uu])
                self.Q[ii] += lr * (e * puu - reg * self.Q[ii])
        return self

    def _predict_one(self, user, item) -> float:
        pred = self.mu
        uu = self.umap.get(user)
        ii = self.imap.get(item)
        if uu is not None:
            pred += self.bu[uu]
        if ii is not None:
            pred += self.bi[ii]
        if uu is not None and ii is not None:
            pred += self.P[uu] @ self.Q[ii]
        return pred

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.array([self._predict_one(u, i)
                        for u, i in zip(df["userId"], df["movieId"])])
        return np.clip(out, RATING_MIN, RATING_MAX)

    def recommend(self, user, movie_ids, k: int = 10) -> list:
        """Top-k movie ids for a user, among the given candidate ids."""
        preds = [(m, self._predict_one(user, m)) for m in movie_ids]
        preds.sort(key=lambda t: t[1], reverse=True)
        return [m for m, _ in preds[:k]]
