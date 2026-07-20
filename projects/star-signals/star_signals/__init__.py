"""star-signals: does GitHub reward substance or surface signal?

A capstone study — collect a cross-section of repositories, split their features
into 'signal' (surface polish) and 'substance' (real engineering activity), and
let a model tell us which one actually predicts stars.
"""

from .collect import collect, save
from .features import (
    ALL_FEATURES,
    SIGNAL_FEATURES,
    SUBSTANCE_FEATURES,
    TARGET,
    build_features,
    group_of,
)
from .model import ModelResult, fit

__all__ = [
    "collect",
    "save",
    "build_features",
    "group_of",
    "SIGNAL_FEATURES",
    "SUBSTANCE_FEATURES",
    "ALL_FEATURES",
    "TARGET",
    "fit",
    "ModelResult",
]
