"""Command-line entry point for the word-prints stylometry capstone."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import corpus, model, plots

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "chunks.csv.gz"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="word-prints",
        description="Can we identify an author from the tiny words they don't think "
        "about? A stylometry study on public-domain books.",
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to chunks.csv.gz")
    p.add_argument("--collect", action="store_true",
                   help="Re-download and re-chunk the books from Project Gutenberg.")
    p.add_argument("--chunk-size", type=int, default=1200, help="Words per passage.")
    p.add_argument("--max-chunks", type=int, default=40, help="Max passages per book.")
    p.add_argument("--assets-dir", default="assets", help="Directory for charts.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.collect:
        print("Downloading books from Project Gutenberg...")
        df = corpus.build_chunks(chunk_size=args.chunk_size, max_chunks=args.max_chunks)
        corpus.save(df, args.data)
        print(f"Saved {len(df)} passages to {args.data}")
    else:
        df = corpus.load(args.data)

    n_authors = df["author"].nunique()
    n_books = df["book_id"].nunique()
    print(f"\nWord-prints: {len(df)} passages, {n_books} books, {n_authors} authors")

    results = model.compare_representations(df)
    fn, full = results["function"], results["full"]

    print("\nCan we name the author of a passage from a book the model never saw?")
    print(f"  Random guessing (largest class):      {fn.baseline:.0%}")
    print(f"  Full vocabulary (content included):    {full.accuracy:.0%}")
    print(f"  FUNCTION WORDS ONLY (style, no topic): {fn.accuracy:.0%}")
    delta = fn.accuracy - full.accuracy
    if delta >= 0:
        print(f"\n  Stripping every content word actually *raises* accuracy by "
              f"{delta:+.0%} — the fingerprint is pure style, not subject matter.")
    else:
        print(f"\n  Stripping every content word costs only {-delta:.0%} — most of "
              f"the signal was style all along.")

    print("\nA few signature words (most distinctive vs. the corpus average):")
    sig = model.signature_words(df, top=3)
    for author in sorted(sig["author"].unique()):
        words = ", ".join(sig[sig["author"] == author]["word"].tolist())
        print(f"  {author:22s} {words}")

    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    sig_full = model.signature_words(df, top=8)
    p1 = plots.plot_confusion(fn, assets / "confusion.png")
    p2 = plots.plot_signatures(sig_full, assets / "signatures.png")
    p3 = plots.plot_style_map(df, assets / "style_map.png")
    print("\nSaved charts:")
    for path in (p1, p2, p3):
        print(f"  {path}")
    print()
