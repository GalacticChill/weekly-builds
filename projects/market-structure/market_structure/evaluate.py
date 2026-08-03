"""Score the discovered clusters against the sectors we held back.

Two complementary numbers:

- **Purity** is the intuitive one: assign each discovered cluster the sector that
  is most common inside it, then ask what fraction of names that gets right. Easy
  to read, but it flatters solutions with many tiny clusters.
- **Adjusted Rand index (ARI)** compares the two groupings pair-by-pair and, unlike
  purity, corrects for the agreement you'd expect by pure chance. 1.0 is a perfect
  match; 0.0 is no better than random; it can even go slightly negative.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def contingency(true_labels, pred_labels) -> pd.DataFrame:
    """Cross-tab of true sector (rows) against discovered cluster (columns)."""
    t = pd.Series(list(true_labels), name="sector")
    p = pd.Series(list(pred_labels), name="cluster")
    return pd.crosstab(t, p)


def purity(true_labels, pred_labels) -> float:
    """Fraction of names whose cluster's majority sector is their true sector."""
    df = pd.DataFrame({"true": list(true_labels), "pred": list(pred_labels)})
    if len(df) == 0:
        return 0.0
    correct = 0
    for _, group in df.groupby("pred"):
        correct += group["true"].value_counts().iloc[0]
    return correct / len(df)


def adjusted_rand(true_labels, pred_labels) -> float:
    """Adjusted Rand index between the true sectors and the discovered clusters."""
    return float(adjusted_rand_score(list(true_labels), list(pred_labels)))


def cluster_composition(tickers, true_labels, pred_labels) -> pd.DataFrame:
    """A readable per-ticker table: ticker, true sector, discovered cluster."""
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "sector": list(true_labels),
            "cluster": list(pred_labels),
        }
    ).sort_values(["cluster", "sector"]).reset_index(drop=True)
