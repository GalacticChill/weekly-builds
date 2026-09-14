"""Offline tests for corpus fetching helpers — no network."""

from __future__ import annotations

from word_prints import corpus


def test_strip_boilerplate_removes_header_and_footer():
    raw = (
        "The Project Gutenberg eBook ... legal blah blah\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
        "Real body text here.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
        "More legal footer, donation info, licensing."
    )
    body = corpus.strip_boilerplate(raw)
    assert body == "Real body text here."


def test_strip_boilerplate_is_noop_without_markers():
    assert corpus.strip_boilerplate("just some text") == "just some text"


def test_tokenize_lowercases_and_keeps_apostrophes():
    assert corpus.tokenize("Don't STOP, he said!") == ["don't", "stop", "he", "said"]


def test_chunk_words_sizes_and_remainder():
    tokens = [f"w{i}" for i in range(25)]
    chunks = corpus.chunk_words(tokens, size=10, max_chunks=None)
    # 25 tokens, size 10 -> two full chunks; the 5-word remainder is dropped.
    assert len(chunks) == 2
    assert all(len(c.split()) == 10 for c in chunks)


def test_chunk_words_respects_max_chunks():
    tokens = [f"w{i}" for i in range(100)]
    chunks = corpus.chunk_words(tokens, size=10, max_chunks=3)
    assert len(chunks) == 3
