"""Offline tests for attribution — synthetic corpora with a known 'right answer'.

Two behavioral tests carry the argument:

- When authors genuinely differ in *how they use function words*, leave-books-out
  attribution should recover them far above chance.
- When authors are identical in function words but each writes about a unique
  *topic*, the function-word view must stay near chance — proving it really does
  discard content rather than sneaking topic in through the back door.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from word_prints import features as ft
from word_prints import model as ml


def _corpus(author_weights, books_per_author=3, chunks_per_book=8,
            words=250, extra=None, seed=0):
    """Build a synthetic chunk table.

    `author_weights[author]` maps a few function words -> sampling weight. `extra`
    optionally maps author -> a content word stuffed into every one of their chunks
    (to simulate topic that the function-word view should ignore).
    """
    rng = np.random.default_rng(seed)
    rows = []
    bid = 0
    for author, weights in author_weights.items():
        vocab = list(weights)
        probs = np.array(list(weights.values()), dtype=float)
        probs = probs / probs.sum()
        for _ in range(books_per_author):
            for c in range(chunks_per_book):
                toks = rng.choice(vocab, size=words, p=probs).tolist()
                if extra and author in extra:
                    toks += [extra[author]] * 40
                rows.append({"author": author, "title": f"b{bid}", "book_id": bid,
                             "chunk_id": c, "text": " ".join(toks)})
            bid += 1
    return pd.DataFrame(rows)


def test_recovers_authors_from_function_word_style():
    # Three authors, each leaning on a different function word.
    weights = {
        "A": {"the": 5, "of": 1, "and": 1, "to": 1, "in": 1},
        "B": {"the": 1, "of": 5, "and": 1, "to": 1, "in": 1},
        "C": {"the": 1, "of": 1, "and": 1, "to": 5, "in": 1},
    }
    df = _corpus(weights, seed=1)
    res = ml.attribute(df, "function")
    assert res.accuracy > 0.75              # well above the 0.33 baseline
    assert res.lift_over_baseline > 0.3


def test_function_view_ignores_topic():
    # Identical function-word usage for everyone, but each author writes about a
    # unique content word. The function-word view carries no author signal, so it
    # must not be able to tell them apart.
    shared = {"the": 1, "of": 1, "and": 1, "to": 1, "in": 1}
    weights = {"A": shared, "B": dict(shared), "C": dict(shared)}
    topics = {"A": "whale", "B": "moor", "C": "raft"}
    df = _corpus(weights, extra=topics, seed=2)
    res = ml.attribute(df, "function")
    assert res.accuracy < 0.55              # near the 0.33 chance level
    # ...but the full-vocabulary view *can* pick up on the topic word.
    assert ml.attribute(df, "full").accuracy > 0.8


def test_result_reports_baseline_and_confusion_shape():
    weights = {
        "A": {"the": 4, "of": 1, "to": 1},
        "B": {"the": 1, "of": 4, "to": 1},
    }
    df = _corpus(weights, seed=3)
    res = ml.attribute(df, "function")
    assert res.confusion.shape == (2, 2)
    assert np.isclose(res.baseline, 0.5, atol=0.05)
    assert set(res.labels) == {"A", "B"}


def test_signature_words_are_per_author_and_finite():
    weights = {
        "A": {"the": 5, "of": 1, "and": 1},
        "B": {"the": 1, "of": 5, "and": 1},
    }
    df = _corpus(weights, seed=4)
    sig = ml.signature_words(df, top=3)
    assert set(sig["author"]) == {"A", "B"}
    assert (sig.groupby("author").size() == 3).all()
    assert np.isfinite(sig["distinctiveness"]).all()
