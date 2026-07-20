"""Turn raw repo metadata into 'signal' vs 'substance' features.

The whole project hinges on this split:

- **Signal** features are the *surface presentation* an owner curates to look good
  on paper — the polish a visitor sees before reading a line of code: a long
  description, lots of trendy topic tags, a project homepage, a catchy name.
- **Substance** features proxy for the *actual project*: how long it's been
  maintained, how recently it was worked on, its size, whether it bothers with a
  license, open issues (a sign of real use and engagement).

We then ask a model which group better predicts stars. Note we deliberately do
**not** use forks or open-issue counts as predictors: both are *consequences* of
popularity (more users file more issues and make more forks), not causes, so
including them would just be predicting stars from stars (leakage). Every feature
we keep is something visible on the repo page *before* it gets popular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature groups. Keep these as the single source of truth used everywhere.
SIGNAL_FEATURES = [
    "desc_len",
    "has_description",
    "topics_count",
    "has_homepage",
    "name_len",
]
SUBSTANCE_FEATURES = [
    "age_days",
    "days_since_push",
    "size_kb",
    "has_license",
    "has_wiki",
]
ALL_FEATURES = SIGNAL_FEATURES + SUBSTANCE_FEATURES

TARGET = "log_stars"


def build_features(raw: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Engineer the model-ready feature frame (features + target) from raw rows."""
    if now is None:
        now = pd.Timestamp.now()
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_localize(None)
    df = raw.copy()

    desc = df["description"].fillna("").astype(str)
    topics = df["topics"].fillna("").astype(str)
    homepage = df["homepage"].fillna("").astype(str).str.strip()
    name = df["full_name"].fillna("").astype(str).str.split("/").str[-1]

    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    pushed = pd.to_datetime(df["pushed_at"], errors="coerce", utc=True).dt.tz_localize(None)

    out = pd.DataFrame(index=df.index)
    # Signal / "paper" features
    out["desc_len"] = desc.str.len()
    out["has_description"] = (desc.str.len() > 0).astype(int)
    out["topics_count"] = topics.apply(lambda s: 0 if not s else len(s.split(";")))
    out["has_homepage"] = (homepage.str.len() > 0).astype(int)
    out["name_len"] = name.str.len()
    # Substance features
    out["age_days"] = (now - created).dt.days.clip(lower=0)
    out["days_since_push"] = (now - pushed).dt.days.clip(lower=0)
    out["size_kb"] = df["size"].fillna(0).astype(float)
    out["has_license"] = df["has_license"].astype(int)
    out["has_wiki"] = df["has_wiki"].fillna(False).astype(int)

    # Target: log stars (star counts are extremely skewed, so we model the log).
    out[TARGET] = np.log1p(df["stargazers_count"].fillna(0).astype(float))
    out["stars"] = df["stargazers_count"].fillna(0).astype(float)

    return out.dropna().reset_index(drop=True)


def group_of(feature: str) -> str:
    """Return 'signal' or 'substance' for a feature name."""
    if feature in SIGNAL_FEATURES:
        return "signal"
    if feature in SUBSTANCE_FEATURES:
        return "substance"
    raise KeyError(feature)
