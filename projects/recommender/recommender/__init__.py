"""recommender: does matrix factorization beat recommending popular movies?

A MovieLens collaborative-filtering study. Four models from a global-mean baseline up
to matrix factorization, judged on rating error, cold start, and -- the honest test --
top-N ranking against a plain popularity list.
"""

from .data import fetch, load, save, train_test_split
from .evaluate import popularity_scores, recall_at_k, rmse, rmse_by_activity
from .models import BiasModel, GlobalMean, ItemMean, MatrixFactorization

__all__ = [
    "fetch",
    "load",
    "save",
    "train_test_split",
    "GlobalMean",
    "ItemMean",
    "BiasModel",
    "MatrixFactorization",
    "rmse",
    "rmse_by_activity",
    "recall_at_k",
    "popularity_scores",
]
