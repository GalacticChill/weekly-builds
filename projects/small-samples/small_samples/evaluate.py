"""Score raw vs. shrunken estimates against the true rates we kept.

Because this is a simulation, we know every player's real success probability, so we
can ask the only question that matters: which estimate is closer to the truth? The
answer is reported overall, broken down by sample size (where shrinkage earns its
keep), and as a ranking test (does the 'best' player by each estimate match reality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rmse(estimate, truth) -> float:
    e, t = np.asarray(estimate, float), np.asarray(truth, float)
    return float(np.sqrt(np.mean((e - t) ** 2)))


def rmse_by_sample_size(df: pd.DataFrame, edges=(1, 5, 20, 100, 10_000)) -> pd.DataFrame:
    """RMSE of raw vs. shrunken estimates within trial-count buckets."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["trials"] >= lo) & (df["trials"] < hi)
        sub = df[m]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "bucket": f"{lo}-{hi - 1} trials",
                "players": int(len(sub)),
                "raw_rmse": rmse(sub["raw_rate"], sub["true_rate"]),
                "shrunk_rmse": rmse(sub["shrunk_rate"], sub["true_rate"]),
            }
        )
    out = pd.DataFrame(rows)
    out["improvement"] = 1 - out["shrunk_rmse"] / out["raw_rmse"]
    return out


def ranking_overlap(df: pd.DataFrame, k: int = 10) -> dict:
    """How well each estimate's top-k matches the true top-k players."""
    true_top = set(df.nlargest(k, "true_rate")["player"])
    raw_top = set(df.nlargest(k, "raw_rate")["player"])
    shrunk_top = set(df.nlargest(k, "shrunk_rate")["player"])
    return {
        "k": k,
        "raw_overlap": len(true_top & raw_top) / k,
        "shrunk_overlap": len(true_top & shrunk_top) / k,
    }


def rank_correlation(df: pd.DataFrame) -> dict:
    """Spearman rank correlation of each estimate with the true rate.

    A steadier summary than top-k overlap: how well does each estimate order the
    whole field, not just the extreme tip?
    """
    return {
        "raw": float(spearmanr(df["raw_rate"], df["true_rate"]).statistic),
        "shrunk": float(spearmanr(df["shrunk_rate"], df["true_rate"]).statistic),
    }


def naive_leader(df: pd.DataFrame) -> pd.Series:
    """The player a naive observer would crown (highest raw rate, ties by trials)."""
    return df.sort_values(["raw_rate", "trials"], ascending=False).iloc[0]
