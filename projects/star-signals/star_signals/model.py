"""Model whether 'signal' or 'substance' better predicts a repo's stars.

Two complementary models, both evaluated **out-of-sample** (train on one split,
score on unseen repos — the same honesty discipline as the backtester):

- A **linear model** on standardized features, for readable coefficients: which
  way does each feature push stars, and how hard?
- A **random forest**, which captures nonlinear effects and gives a feature
  importance ranking that doesn't assume a straight-line relationship.

We then roll the per-feature importances up into the two groups to get the
headline answer: what share of a repo's predictable success traces to substance
versus surface signal?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import features as ft


@dataclass
class ModelResult:
    linear_r2: float                 # out-of-sample R^2 of the linear model
    forest_r2: float                 # out-of-sample R^2 of the random forest
    coefficients: pd.Series          # standardized linear coefficients per feature
    importances: pd.Series           # random-forest importances per feature
    group_importance: pd.Series      # forest importance summed by group (fractions)
    group_coef_share: pd.Series      # |coef| share by group
    y_test: np.ndarray               # actual log-stars on the test set
    y_pred: np.ndarray               # forest-predicted log-stars on the test set

    @property
    def verdict(self) -> str:
        sig = self.group_importance.get("signal", 0.0)
        sub = self.group_importance.get("substance", 0.0)
        leader = "substance" if sub >= sig else "signal"
        share = max(sig, sub)
        return f"{leader} explains {share:.0%} of the model's predictive power"


def fit(data: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> ModelResult:
    """Fit both models on `data` (output of `features.build_features`)."""
    X = data[ft.ALL_FEATURES].to_numpy(dtype=float)
    y = data[ft.TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Linear model on standardized features -> comparable coefficients.
    scaler = StandardScaler().fit(X_train)
    linear = LinearRegression().fit(scaler.transform(X_train), y_train)
    linear_pred = linear.predict(scaler.transform(X_test))
    coefs = pd.Series(linear.coef_, index=ft.ALL_FEATURES)

    # Random forest for nonlinear importance.
    forest = RandomForestRegressor(
        n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=3
    ).fit(X_train, y_train)
    forest_pred = forest.predict(X_test)
    importances = pd.Series(forest.feature_importances_, index=ft.ALL_FEATURES)

    # Roll up into signal vs substance.
    groups = pd.Series({f: ft.group_of(f) for f in ft.ALL_FEATURES})
    group_importance = importances.groupby(groups).sum()
    abs_coef = coefs.abs()
    group_coef_share = abs_coef.groupby(groups).sum() / abs_coef.sum()

    return ModelResult(
        linear_r2=float(r2_score(y_test, linear_pred)),
        forest_r2=float(r2_score(y_test, forest_pred)),
        coefficients=coefs.sort_values(key=lambda s: s.abs(), ascending=False),
        importances=importances.sort_values(ascending=False),
        group_importance=group_importance,
        group_coef_share=group_coef_share,
        y_test=y_test,
        y_pred=forest_pred,
    )
