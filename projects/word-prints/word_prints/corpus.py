"""Fetch a small corpus of public-domain books from Project Gutenberg.

Five authors with distinctive voices, four books each. We download the plain-text
editions, strip Gutenberg's header/footer boilerplate, and cut each book into
fixed-length word chunks — the samples the model actually sees. Chunking matters:
it turns 20 books into hundreds of labelled passages, and it lets us hold out
*whole books* for testing so the model has to recognize an author's style in a work
it has never seen.

The chunked corpus is cached to `data/chunks.csv.gz` so the analysis runs offline
and reproducibly; `--collect` refreshes it from Gutenberg.
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

# author -> list of (title, Gutenberg id)
CORPUS: dict[str, list[tuple[str, int]]] = {
    "Jane Austen": [
        ("Pride and Prejudice", 1342),
        ("Sense and Sensibility", 161),
        ("Emma", 158),
        ("Persuasion", 105),
    ],
    "Charles Dickens": [
        ("A Tale of Two Cities", 98),
        ("Great Expectations", 1400),
        ("Oliver Twist", 730),
        ("David Copperfield", 766),
    ],
    "Mark Twain": [
        ("Huckleberry Finn", 76),
        ("Tom Sawyer", 74),
        ("A Connecticut Yankee", 86),
        ("The Innocents Abroad", 3176),
    ],
    "Arthur Conan Doyle": [
        ("Adventures of Sherlock Holmes", 1661),
        ("Hound of the Baskervilles", 2852),
        ("The Sign of the Four", 2097),
        ("A Study in Scarlet", 244),
    ],
    "H. G. Wells": [
        ("The War of the Worlds", 36),
        ("The Time Machine", 35),
        ("The Invisible Man", 5230),
        ("The Island of Doctor Moreau", 159),
    ],
}

_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
_UA = {"User-Agent": "Mozilla/5.0 (stylometry research)"}

_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_WORD = re.compile(r"[A-Za-z']+")


def _fetch(book_id: int, retries: int = 3, pause: float = 1.0) -> str:
    url = _URL.format(id=book_id)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch Gutenberg book {book_id}: {last}")


def strip_boilerplate(text: str) -> str:
    """Return only the body between Gutenberg's START and END markers."""
    start = _START.search(text)
    if start:
        text = text[start.end():]
    end = _END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters and apostrophes only)."""
    return _WORD.findall(text.lower())


def chunk_words(tokens: list[str], size: int, max_chunks: int | None) -> list[str]:
    """Split a token list into space-joined chunks of `size` words each.

    Drops a final short remainder so every chunk is the same length; optionally
    caps the number of chunks per book to keep classes roughly balanced.
    """
    chunks = [
        " ".join(tokens[i : i + size])
        for i in range(0, len(tokens) - size + 1, size)
    ]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks


def build_chunks(
    chunk_size: int = 1200, max_chunks: int = 40, pause: float = 1.0
) -> pd.DataFrame:
    """Download every book, clean it, and cut it into labelled word chunks."""
    rows = []
    for author, books in CORPUS.items():
        for title, book_id in books:
            body = strip_boilerplate(_fetch(book_id))
            tokens = tokenize(body)
            for j, chunk in enumerate(chunk_words(tokens, chunk_size, max_chunks)):
                rows.append(
                    {
                        "author": author,
                        "title": title,
                        "book_id": book_id,
                        "chunk_id": j,
                        "text": chunk,
                    }
                )
            time.sleep(pause)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the chunk table to a gzipped CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return path


def load(path: str | Path) -> pd.DataFrame:
    """Read the cached chunk table."""
    return pd.read_csv(path, compression="gzip")
