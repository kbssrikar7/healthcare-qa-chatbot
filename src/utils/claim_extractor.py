"""
Unified medical claim extraction utility.

Consolidates three separate claim extraction implementations that previously lived
independently across:
  - SourceAttributor.extract_claims()        (src/xai/source_attribution.py)
  - FactualConsistencyChecker._extract_claims() (src/xai/factual_consistency.py)
  - HealthcareRAGNodes._extract_claims()     (src/langgraph/langgraph_nodes.py)

All three produced different outputs from the same input text, making XAI signals
inconsistent. This module provides a single canonical implementation based on the
best features of all three.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Shared skip patterns (union of FactualConsistencyChecker + LangGraph nodes)
# ---------------------------------------------------------------------------

_SKIP_WORDS = frozenset({
    "consult", "recommend", "important", "disclaimer", "please",
    "physician", "doctor", "healthcare", "professional", "provider",
    "advice", "treatment", "medical",  # too generic to be a meaningful claim
})

# Sentence-boundary abbreviations: protect from splitting mid-abbreviation
_ABBR_PROTECT = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "mg.", "ml.", "vs.", "i.e.", "e.g.", "etc.",
    "approx.", "dept.", "Fig.", "Tab.", "vol.", "no.",
]

# Minimum character length for a claim sentence
_MIN_CLAIM_LEN = 20


def _protect_abbreviations(text: str) -> str:
    """Replace dots in known abbreviations to avoid false sentence splits."""
    protected = text
    for abbr in _ABBR_PROTECT:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def extract_medical_claims(
    text: str,
    min_length: int = _MIN_CLAIM_LEN,
    max_claims: int = 10,
    skip_questions: bool = True,
    skip_disclaimer_sentences: bool = True,
) -> List[str]:
    """Extract factual medical claims from a block of text.

    This is the single canonical claim extractor used across SourceAttributor,
    FactualConsistencyChecker, HallucinationDetector, and LangGraph nodes.

    Parameters
    ----------
    text : str
        The medical text to extract claims from (answer or document content).
    min_length : int
        Minimum character length for a sentence to be considered a claim.
    max_claims : int
        Maximum number of claims to return.
    skip_questions : bool
        If True, skip sentences ending with "?" (rhetorical questions).
    skip_disclaimer_sentences : bool
        If True, skip sentences containing disclaimer / advisory language
        (consult, recommend, etc.) — these are not verifiable medical facts.

    Returns
    -------
    List[str]
        Ordered list of factual claim sentences extracted from the text.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviation dots before splitting
    protected = _protect_abbreviations(text)

    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

    claims: List[str] = []
    for raw_sent in raw_sentences:
        sent = _restore_abbreviations(raw_sent.strip())
        if not sent:
            continue

        # Length filter
        if len(sent) < min_length:
            continue

        # Question filter
        if skip_questions and sent.endswith("?"):
            continue

        # Disclaimer / advisory filter (too general, not falsifiable)
        if skip_disclaimer_sentences:
            sent_lower = sent.lower()
            if any(word in sent_lower for word in _SKIP_WORDS):
                continue

        # Deduplicate (avoid the same claim appearing twice)
        if sent not in claims:
            claims.append(sent)

        if len(claims) >= max_claims:
            break

    return claims


def extract_claims_from_answer(answer: str) -> List[str]:
    """Convenience wrapper for extracting claims from a generated answer.

    Uses conservative settings: skips questions and disclaimers, returns at most 8 claims.
    """
    return extract_medical_claims(
        answer,
        min_length=_MIN_CLAIM_LEN,
        max_claims=8,
        skip_questions=True,
        skip_disclaimer_sentences=True,
    )


def extract_claims_from_document(doc_content: str) -> List[str]:
    """Convenience wrapper for extracting claims from a retrieved document.

    More permissive: keeps advisory sentences (documents may contain clinical
    guidelines), returns up to 15 claims.
    """
    return extract_medical_claims(
        doc_content,
        min_length=_MIN_CLAIM_LEN,
        max_claims=15,
        skip_questions=True,
        skip_disclaimer_sentences=False,
    )
