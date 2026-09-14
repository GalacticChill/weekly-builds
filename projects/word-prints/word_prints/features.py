"""Two ways to turn a passage into numbers.

The whole argument of this project lives in the contrast between them:

- `function_word_matrix` counts only ~170 *function words* — the, of, and, to, that,
  upon, which... — the connective tissue of English that carries almost no topic.
  If an author can be identified from these alone, the fingerprint is *style*, not
  subject matter.
- `full_vocabulary_matrix` is an ordinary TF-IDF over all words, content included.
  It's usually more accurate, but some of that accuracy is really just detecting
  *what* an author wrote about (whales, moors, the French Revolution) rather than
  *how* they wrote — which is why we lean on the function-word view for the honest
  claim.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# A compact, deliberately content-free list of English function words: articles,
# pronouns, prepositions, conjunctions, auxiliaries, and a few high-frequency
# adverbs/determiners. These are the classic stylometric features (the same family
# Mosteller & Wallace used to settle the disputed Federalist Papers).
FUNCTION_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "either", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "might", "more", "most", "much", "must", "my", "myself", "neither",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "shall", "she",
    "should", "since", "so", "some", "still", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "though", "through", "to", "too", "under", "until", "up",
    "upon", "us", "very", "was", "we", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whom", "whose", "why", "will", "with", "would", "yet",
    "you", "your", "yours", "yourself", "yourselves", "aboard", "amid", "among",
    "around", "behind", "beneath", "beside", "besides", "beyond", "near", "toward",
    "towards", "within", "without", "unless", "whereas", "hence", "thus",
    "therefore", "however", "moreover", "indeed", "perhaps", "rather", "quite",
]


def function_word_matrix(texts):
    """Relative frequencies of the function words in each text (rows sum to ~1).

    Normalizing by each passage's own function-word count removes length effects,
    so what's left is *how the author distributes* their function words.
    """
    vec = CountVectorizer(vocabulary=FUNCTION_WORDS, lowercase=True)
    counts = vec.fit_transform(texts).toarray().astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals, list(FUNCTION_WORDS)


def full_vocabulary_matrix(texts, max_features: int = 3000):
    """TF-IDF over the full vocabulary (content words included), for comparison."""
    vec = TfidfVectorizer(lowercase=True, max_features=max_features, min_df=2)
    x = vec.fit_transform(texts)
    return x, vec.get_feature_names_out().tolist()
