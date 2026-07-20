"""Offline tests for feature engineering — no network, hand-checked answers."""

import numpy as np
import pandas as pd
import pytest

from star_signals import features as ft


def _raw_row(**overrides) -> pd.DataFrame:
    base = {
        "full_name": "octocat/hello-world",
        "stargazers_count": 100,
        "forks_count": 10,
        "open_issues_count": 5,
        "size": 2048,
        "language": "Python",
        "description": "hello",
        "homepage": "https://example.com",
        "has_wiki": True,
        "has_license": True,
        "topics": "cli;tools;python",
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2020-12-31T00:00:00Z",
    }
    base.update(overrides)
    return pd.DataFrame([base])


NOW = pd.Timestamp("2021-01-01")


def test_signal_features_hand_computed():
    f = ft.build_features(_raw_row(), now=NOW).iloc[0]
    assert f["desc_len"] == 5              # len("hello")
    assert f["has_description"] == 1
    assert f["topics_count"] == 3          # cli;tools;python
    assert f["has_homepage"] == 1
    assert f["name_len"] == len("hello-world")


def test_substance_features_hand_computed():
    f = ft.build_features(_raw_row(), now=NOW).iloc[0]
    assert f["age_days"] == 366            # 2020 was a leap year
    assert f["days_since_push"] == 1       # pushed 2020-12-31, now 2021-01-01
    assert f["size_kb"] == 2048
    assert f["has_license"] == 1
    assert f["has_wiki"] == 1


def test_empty_fields_handled():
    f = ft.build_features(
        _raw_row(description=None, homepage="", topics="", has_wiki=False), now=NOW
    ).iloc[0]
    assert f["desc_len"] == 0
    assert f["has_description"] == 0
    assert f["has_homepage"] == 0
    assert f["topics_count"] == 0
    assert f["has_wiki"] == 0


def test_target_is_log_stars():
    f = ft.build_features(_raw_row(stargazers_count=999), now=NOW).iloc[0]
    assert np.isclose(f[ft.TARGET], np.log1p(999))
    assert f["stars"] == 999


def test_forks_and_issues_are_not_predictors():
    # Both are popularity consequences and must stay out of the feature set.
    assert "forks_count" not in ft.ALL_FEATURES
    assert "open_issues_count" not in ft.ALL_FEATURES


def test_feature_groups_partition_cleanly():
    assert set(ft.SIGNAL_FEATURES).isdisjoint(ft.SUBSTANCE_FEATURES)
    assert set(ft.ALL_FEATURES) == set(ft.SIGNAL_FEATURES) | set(ft.SUBSTANCE_FEATURES)
    for f in ft.SIGNAL_FEATURES:
        assert ft.group_of(f) == "signal"
    for f in ft.SUBSTANCE_FEATURES:
        assert ft.group_of(f) == "substance"


def test_group_of_rejects_unknown():
    with pytest.raises(KeyError):
        ft.group_of("not_a_feature")
