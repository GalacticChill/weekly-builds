"""Attribute each passage to an author, honestly.

The one methodological choice that makes or breaks this study is the train/test
split. If we split passages at random, chunks from the *same book* land in both
train and test, and the model can win by memorizing that book's characters and
places — topic leakage dressed up as style. So we split **by book**: every chunk of
a held-out book is in the test set and none of it is in training. To be recognized,
an author's fingerprint has to survive into a work the model has never opened.

`attribute` trains a classifier under that rule and reports out-of-sample accuracy;
`compare_representations` runs it for both the function-word and full-vocabulary
views so their honesty can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import features as ft


@dataclass
class AttributionResult:
    """Out-of-sample authorship attribution under a leave-books-out split."""

    representation: str
    accuracy: float
    baseline: float                 # accuracy of always guessing the largest class
    labels: list[str]
    confusion: np.ndarray = field(repr=False)
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    @property
    def lift_over_baseline(self) -> float:
        return self.accuracy - self.baseline


def _majority_baseline(y) -> float:
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def attribute(df: pd.DataFrame, representation: str = "function") -> AttributionResult:
    """Leave-books-out cross-validated authorship attribution.

    Every book is held out once (as test) via GroupKFold on `book_id`; each held-out
    chunk is predicted by a model trained only on the *other* books. Predictions are
    pooled across folds for one honest out-of-sample accuracy and confusion matrix.
    """
    texts = df["text"].tolist()
    y = df["author"].to_numpy()
    groups = df["book_id"].to_numpy()

    if representation == "function":
        x, _ = ft.function_word_matrix(texts)
    elif representation == "full":
        x, _ = ft.full_vocabulary_matrix(texts)
    else:
        raise ValueError(f"unknown representation {representation!r}")

    n_books = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_books)          # leave-one-book-out

    y_true, y_pred = [], []
    for train_idx, test_idx in gkf.split(x, y, groups):
        clf = LogisticRegression(max_iter=2000, C=10.0)
        x_tr, x_te = x[train_idx], x[test_idx]
        if representation == "function":
            # Standardize each function word to comparable scale (Burrows-style),
            # fitting the scaler on the training books only (no leakage).
            scaler = StandardScaler().fit(x_tr)
            x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
        clf.fit(x_tr, y[train_idx])
        y_true.extend(y[test_idx])
        y_pred.extend(clf.predict(x_te))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(np.unique(y).tolist())

    return AttributionResult(
        representation=representation,
        accuracy=float(accuracy_score(y_true, y_pred)),
        baseline=float(_majority_baseline(y)),
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        y_true=y_true,
        y_pred=y_pred,
    )


def compare_representations(df: pd.DataFrame) -> dict[str, AttributionResult]:
    """Run attribution for both the function-word and full-vocabulary views."""
    return {
        "function": attribute(df, "function"),
        "full": attribute(df, "full"),
    }


def signature_words(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """The function words each author uses most *distinctively*.

    Distinctiveness is a z-score: how many standard deviations an author's mean
    frequency of a word sits above the corpus-wide mean. Using the z-score rather
    than the raw difference stops the handful of very common words (the, of, and)
    from dominating every author's list and surfaces the genuinely telling ones.
    """
    x, vocab = ft.function_word_matrix(df["text"].tolist())
    freq = pd.DataFrame(x, columns=vocab)
    freq["author"] = df["author"].to_numpy()
    overall_mean = freq[vocab].mean()
    overall_std = freq[vocab].std().replace(0.0, 1.0)

    rows = []
    for author, grp in freq.groupby("author"):
        z = ((grp[vocab].mean() - overall_mean) / overall_std).sort_values(ascending=False)
        for word in z.head(top).index:
            rows.append({"author": author, "word": word, "distinctiveness": float(z[word])})
    return pd.DataFrame(rows)
