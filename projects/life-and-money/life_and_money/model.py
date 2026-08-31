"""Three questions, three models.

1. `preston_fit` — does income buy life *linearly*, or with diminishing returns?
   We fit life expectancy against raw GDP and against log-GDP and compare; log
   winning by a wide margin is the quantitative signature of the Preston curve.
2. `full_fit` — beyond income, what predicts longevity? A standardized linear model
   (readable coefficients) and a random forest (nonlinear importances), both scored
   out-of-sample.
3. `income_residuals` — after income has had its say, which countries live far
   longer than their wealth predicts, and which fall short? These residuals are the
   most human part of the story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .collect import MONEY, TARGET
from .features import PREDICTORS


@dataclass
class PrestonFit:
    """The income-only picture: linear vs. log, and the fitted log curve."""

    r2_linear: float        # life ~ gdp
    r2_log: float           # life ~ log10(gdp)
    slope: float            # years of life per decade (10x) of income
    intercept: float

    def predict(self, gdp_pc_ppp) -> np.ndarray:
        """Predicted life expectancy from raw GDP per capita via the log model."""
        x = np.log10(np.asarray(gdp_pc_ppp, dtype=float))
        return self.intercept + self.slope * x


def preston_fit(df: pd.DataFrame) -> PrestonFit:
    """Compare a linear-in-income vs linear-in-log-income fit of life expectancy."""
    y = df[TARGET].to_numpy()
    gdp = df[MONEY].to_numpy().reshape(-1, 1)
    log_gdp = np.log10(df[MONEY].to_numpy()).reshape(-1, 1)

    lin = LinearRegression().fit(gdp, y)
    log = LinearRegression().fit(log_gdp, y)

    return PrestonFit(
        r2_linear=float(r2_score(y, lin.predict(gdp))),
        r2_log=float(r2_score(y, log.predict(log_gdp))),
        slope=float(log.coef_[0]),
        intercept=float(log.intercept_),
    )


@dataclass
class ModelResult:
    """Out-of-sample fit of the multivariate model, plus what drove it."""

    linear_r2: float
    forest_r2: float
    coefficients: pd.Series        # standardized linear coefficients
    importances: pd.Series         # random-forest importances (sum to 1)
    y_test: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    @property
    def top_feature(self) -> str:
        return self.importances.idxmax()


def full_fit(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> ModelResult:
    """Fit life expectancy on all predictors; score on held-out countries."""
    x = df[PREDICTORS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size, random_state=seed)

    scaler = StandardScaler().fit(x_tr)
    linear = LinearRegression().fit(scaler.transform(x_tr), y_tr)
    lin_pred = linear.predict(scaler.transform(x_te))

    forest = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ).fit(x_tr, y_tr)
    rf_pred = forest.predict(x_te)

    coefs = pd.Series(linear.coef_, index=PREDICTORS).sort_values(key=np.abs, ascending=False)
    imps = pd.Series(forest.feature_importances_, index=PREDICTORS).sort_values(ascending=False)

    return ModelResult(
        linear_r2=float(r2_score(y_te, lin_pred)),
        forest_r2=float(r2_score(y_te, rf_pred)),
        coefficients=coefs,
        importances=imps,
        y_test=y_te,
        y_pred=rf_pred,
    )


def income_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Life expectancy minus what income alone predicts, per country.

    Positive = the country lives longer than its wealth would suggest ("beats the
    odds"); negative = it falls short. Fit on the whole sample so every country is
    scored against the same income curve.
    """
    fit = preston_fit(df)
    predicted = fit.predict(df[MONEY].to_numpy())
    out = pd.DataFrame(
        {
            "country": df["country"] if "country" in df else df.index,
            "life_expectancy": df[TARGET].to_numpy(),
            "predicted_from_income": predicted,
            "residual": df[TARGET].to_numpy() - predicted,
        },
        index=df.index,
    )
    if "income_group" in df:
        out["income_group"] = df["income_group"]
    return out.sort_values("residual", ascending=False)
