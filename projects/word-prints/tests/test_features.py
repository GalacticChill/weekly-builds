"""Offline tests for the two text vectorizers — hand-checked."""

from __future__ import annotations

import numpy as np

from word_prints import features as ft


def test_function_word_matrix_relative_frequencies():
    # "the the of cat" -> function words seen: the, the, of (cat is content, ignored).
    # So relative freqs: the = 2/3, of = 1/3, everything else 0.
    x, vocab = ft.function_word_matrix(["the the of cat"])
    row = dict(zip(vocab, x[0]))
    assert np.isclose(row["the"], 2 / 3)
    assert np.isclose(row["of"], 1 / 3)
    assert np.isclose(x[0].sum(), 1.0)          # rows are a distribution
    assert "cat" not in vocab                    # content words are not features


def test_function_word_matrix_all_content_is_zero_row():
    # A passage with no function words at all -> an all-zero row (no divide error).
    x, _ = ft.function_word_matrix(["cat dog elephant"])
    assert np.isclose(x[0].sum(), 0.0)


def test_vocabulary_matches_function_word_list():
    x, vocab = ft.function_word_matrix(["the of and"])
    assert vocab == ft.FUNCTION_WORDS
    assert x.shape[1] == len(ft.FUNCTION_WORDS)


def test_full_vocabulary_includes_content_words():
    texts = ["the whale swam", "the whale dove", "a moor at night", "a moor is cold"]
    x, vocab = ft.full_vocabulary_matrix(texts, max_features=50)
    assert "whale" in vocab or "moor" in vocab    # content survives here (unlike above)
    assert x.shape[0] == 4
