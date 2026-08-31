"""life-and-money: does money buy a longer life, and who beats the odds?

A cross-country capstone using World Bank data. Trace the Preston curve (income
buys life, with steep diminishing returns), model what predicts longevity beyond
income, and rank the countries that live far longer — or shorter — than their
wealth alone would predict.
"""

from .collect import INDICATORS, MONEY, TARGET, collect, save
from .features import PREDICTORS, build_features, load, missing_report
from .model import (
    ModelResult,
    PrestonFit,
    full_fit,
    income_residuals,
    preston_fit,
)

__all__ = [
    "collect",
    "save",
    "INDICATORS",
    "TARGET",
    "MONEY",
    "load",
    "build_features",
    "missing_report",
    "PREDICTORS",
    "preston_fit",
    "PrestonFit",
    "full_fit",
    "ModelResult",
    "income_residuals",
]
