"""
Shared stopword sets for the Healthcare QA pipeline.

All XAI, retrieval, and pipeline modules import from here so there is
one canonical list instead of five separate inline definitions.
"""

ENGLISH_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can",
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "into",
    "this", "that", "these", "those", "it", "its",
    "and", "or", "but", "not", "so", "yet", "nor",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "his", "their", "our", "your", "my",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "only", "same", "than", "too", "very",
    "just", "also", "about", "above", "after", "before", "between",
    "during", "through", "over", "under", "again", "then", "here",
    "there", "once", "if", "as", "up", "out", "because", "while",
    # Negations (kept separate so callers can optionally include)
    "not", "no", "never", "neither", "nor", "aren't", "won't", "without",
})

# Extended set used in term-overlap scoring — excludes negations
# so "not effective" ≠ "effective" in grounding checks.
OVERLAP_STOPWORDS: frozenset = frozenset({
    w for w in ENGLISH_STOPWORDS
    if w not in {"not", "no", "never", "neither", "nor", "without"}
})

# Minimal set for BM25 tokenisation — only the highest-frequency function words
BM25_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "at", "to", "for", "by", "with", "and", "or", "it", "this", "that",
})
