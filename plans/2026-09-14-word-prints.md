# Week of 2026-09-14 — Capstone: Word Prints (stylometry / NLP)

**Theme:** data/ML — an original "words & language" capstone (user's pick).
**Type:** the portfolio's FIRST text/NLP project — a new technique dimension.

## Origin
The user asked for another original capstone and chose the "words & language (new
technique)" direction via AskUserQuestion. Framed as stylometry: can an author be
identified from function words alone?

## The question
Every writer has a fingerprint hidden in the tiny words they don't think about
(the, of, upon, that). Can we identify the author of a passage from function-word
frequencies ALONE — content stripped away — and does that beat using the full
vocabulary?

## Scope (shipped)
- `corpus.py` — download 20 public-domain novels (5 authors x 4 books) from Project
  Gutenberg, strip license boilerplate, tokenize, cut into fixed 1200-word passages
  (max 40/book). Cached to `data/chunks.csv.gz` (committed, ~1.6MB, runs offline).
- `features.py` — two views: relative frequencies of ~170 FUNCTION_WORDS
  (content-free style) vs full-vocabulary TF-IDF (content included).
- `model.py` — logistic regression scored **leave-one-book-out** (`GroupKFold` on
  book_id — the anti-leakage crux: test books never seen in training). Function-word
  features standardized within each fold (Burrows-style). `signature_words` ranks
  each author's function words by z-score distinctiveness.
- `plots.py` — confusion matrix, per-author signature small-multiples, PCA style map.
- CLI + committed corpus.
- 13 offline synthetic tests, incl. two behavioral ones: distinct function-word
  usage IS recovered above chance; topic-only differences are NOT (proving the
  function-word view discards content).

## Findings (5 authors, ~780 passages, leave-books-out)
- **Function words ALONE: 87% accuracy** (chance 20%). Full vocabulary: 82%.
  **Stripping all content words slightly RAISES accuracy (+5pts)** — because content
  encodes topic, which doesn't generalize across an author's different books, while
  function-word style does. A lesson about honest evaluation, not just style.
- Jane Austen identified 97%; every author >= 80%.
- Signatures are historically real: Wells "towards/into" (British), Twain
  "toward/around" (American); Doyle archaic "upon"; Austen "her/herself/she".

## Why this matters for the portfolio story
First NLP/text project — a genuinely new technique (tokenization, function-word
vectors, TF-IDF, GroupKFold). Reuses the leakage-awareness discipline in a new
guise (split-by-book, style-vs-topic). Memorable, counterintuitive, essay-ready
result; strong visuals. Reliable data (Gutenberg) + committed snapshot. No new deps
beyond the installed sklearn stack.

## Deliberately deferred / next
- Could graduate to its own standalone repo (would be a 4th pinned repo; vivid).
- Add Burrows's Delta as a from-scratch baseline; add more authors / harder cases;
  attribute a genuinely disputed/anonymous text as a demo.

## Status
Built and shipped 2026-09-14. 13 tests pass; verified end-to-end on a freshly
collected 20-book corpus. Three charts generated. New dependency: none.
