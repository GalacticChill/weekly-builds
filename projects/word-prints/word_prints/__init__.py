"""word-prints: every writer has a fingerprint, hidden in the words they don't notice.

A stylometry capstone. Identify the author of a passage from the frequencies of
function words alone -- the, of, and, upon -- content stripped away, and the test
set drawn from whole books the model never saw.
"""

from .corpus import CORPUS, build_chunks, load, save, tokenize
from .features import FUNCTION_WORDS, full_vocabulary_matrix, function_word_matrix
from .model import (
    AttributionResult,
    attribute,
    compare_representations,
    signature_words,
)

__all__ = [
    "CORPUS",
    "build_chunks",
    "load",
    "save",
    "tokenize",
    "FUNCTION_WORDS",
    "function_word_matrix",
    "full_vocabulary_matrix",
    "attribute",
    "AttributionResult",
    "compare_representations",
    "signature_words",
]
