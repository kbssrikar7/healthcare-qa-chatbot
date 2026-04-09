"""
Source attribution for medical QA.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

# Try to import spacy for better NLP, fall back to regex
try:
    import spacy

    NLP_AVAILABLE = True
except Exception:
    NLP_AVAILABLE = False
    spacy = None

# Lazy load spacy model
_nlp = None


def _get_nlp():
    """Lazy load spacy model for sentence segmentation."""
    global _nlp
    if _nlp is None and NLP_AVAILABLE:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, fallback to regex
            pass
    return _nlp


@dataclass
class Attribution:
    """A single source attribution."""

    claim: str
    source: str
    evidence: str
    similarity_score: float
    url: Optional[str] = None


class SourceAttributor:
    """
    Attribute generated claims to source documents.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        embedding_threshold: float = 0.55,
        embedder=None,
        use_nlp: bool = True,
    ):
        # Text overlap threshold (Jaccard-like: 0–1, lower scores expected)
        self.similarity_threshold = similarity_threshold
        # Cosine similarity threshold for embedding-based matching
        self.embedding_threshold = embedding_threshold
        self.embedder = embedder
        self.use_nlp = use_nlp and NLP_AVAILABLE

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from generated text using NLP sentence segmentation.

        Uses spacy for proper sentence boundary detection when available,
        with regex fallback that handles medical abbreviations.
        """
        # Try NLP-based extraction first
        if self.use_nlp:
            nlp = _get_nlp()
            if nlp:
                return self._extract_claims_nlp(text, nlp)

        # Fallback to improved regex-based extraction
        return self._extract_claims_regex(text)

    def _extract_claims_nlp(self, text: str, nlp) -> List[str]:
        """Extract claims using spacy NLP pipeline."""
        doc = nlp(text)
        claims = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Filter criteria for factual claims
            if self._is_factual_claim(sent_text):
                claims.append(sent_text)

        return claims

    def _extract_claims_regex(self, text: str) -> List[str]:
        """
        Extract claims using improved regex that handles medical abbreviations.

        Handles cases like: "Dr.", "mg.", "ml.", "vs.", "i.e.", "e.g."
        """
        # Protect common medical abbreviations from sentence splitting
        protected = text
        # (pattern, placeholder) pairs — placeholders must not contain periods
        abbrev_protections = [
            (r"\bDr\.", "DR_ABBR"),
            (r"\bMr\.", "MR_ABBR"),
            (r"\bMs\.", "MS_ABBR"),
            (r"\bvs\.", "VS_ABBR"),
            (r"\bi\.e\.", "IE_ABBR"),
            (r"\be\.g\.", "EG_ABBR"),
            (r"\betc\.", "ETC_ABBR"),
            (r"(\d+)\s*mg\.", r"\1 MG_ABBR"),
            (r"(\d+)\s*ml\.", r"\1 ML_ABBR"),
        ]

        for pattern, replacement in abbrev_protections:
            protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)

        # Split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", protected)

        # Restore abbreviations and filter
        # (placeholder, restored_text) pairs
        restore_map = [
            ("DR_ABBR", "Dr."),
            ("MR_ABBR", "Mr."),
            ("MS_ABBR", "Ms."),
            ("VS_ABBR", "vs."),
            ("IE_ABBR", "i.e."),
            ("EG_ABBR", "e.g."),
            ("ETC_ABBR", "etc."),
            ("MG_ABBR", "mg."),
            ("ML_ABBR", "ml."),
        ]

        claims = []
        for sent in sentences:
            restored = sent
            for placeholder, original in restore_map:
                restored = restored.replace(placeholder, original)

            restored = restored.strip()
            if self._is_factual_claim(restored):
                claims.append(restored)

        return claims

    def _is_factual_claim(self, text: str) -> bool:
        """Check if text is likely a factual claim worth attributing."""
        # Minimum length check
        if len(text) < 20:
            return False

        # Skip questions
        if text.endswith("?"):
            return False

        # Skip hedged/uncertain statements
        hedges = ("i think", "maybe", "perhaps", "possibly", "might be", "could be")
        if text.lower().startswith(hedges):
            return False

        # Skip imperatives/instructions (common in medical text but not claims)
        imperatives = ("please", "consult", "see your doctor", "always consult")
        if text.lower().startswith(imperatives):
            return False

        return True

    def find_evidence(self, claim: str, documents: List[Dict]) -> Optional[Dict]:
        """Find best matching evidence for a claim."""
        best_match = None
        best_score = 0.0

        for doc in documents:
            content = doc.get("content", str(doc))
            source = doc.get("source", "Unknown")

            # Calculate similarity (using embedder if available)
            if self.embedder:
                try:
                    score = self._embedding_similarity(claim, content)
                except Exception as e:
                    logger.warning(
                        f"Embedding similarity failed, falling back to text overlap: {e}"
                    )
                    score = self._text_overlap(claim, content)
            else:
                score = self._text_overlap(claim, content)

            if score > best_score:
                best_score = score
                best_match = {
                    "source": source,
                    "evidence": self._extract_relevant_snippet(claim, content),
                    "score": score,
                    "url": doc.get("url", ""),
                }

        threshold = self.embedding_threshold if self.embedder else self.similarity_threshold
        if best_score >= threshold:
            return best_match
        return None

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between embeddings."""
        emb1 = self.embedder.embed_query(text1)
        emb2 = self.embedder.embed_query(text2)
        # Cosine similarity (normalized dot product)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    # Common English stopwords to exclude from overlap scoring
    _STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "no",
            "so",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "we",
            "you",
            "i",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
        }
    )

    def _text_overlap(self, claim: str, content: str) -> float:
        """Calculate word overlap similarity (stopwords excluded)."""
        claim_words = set(claim.lower().split()) - self._STOP_WORDS
        content_words = set(content.lower().split()) - self._STOP_WORDS

        if not claim_words:
            return 0.0

        overlap = claim_words.intersection(content_words)
        return len(overlap) / len(claim_words)

    def _extract_relevant_snippet(self, claim: str, content: str, max_length: int = 200) -> str:
        """Extract the most relevant snippet from content."""
        # Find best matching window
        claim_words = claim.lower().split()

        best_start = 0
        best_score = 0

        words = content.split()
        for i in range(max(1, len(words) - 10)):
            window = " ".join(words[i : i + 20]).lower()
            score = sum(1 for w in claim_words if w in window)
            if score > best_score:
                best_score = score
                best_start = i

        # Extract snippet
        snippet_words = words[best_start : best_start + 30]
        snippet = " ".join(snippet_words)

        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def attribute_answer(self, answer: str, documents: List[Dict]) -> List[Attribution]:
        """Attribute an entire answer to sources."""
        claims = self.extract_claims(answer)
        attributions = []

        for claim in claims:
            evidence = self.find_evidence(claim, documents)

            if evidence:
                attributions.append(
                    Attribution(
                        claim=claim,
                        source=evidence["source"],
                        evidence=evidence["evidence"],
                        similarity_score=evidence["score"],
                        url=evidence.get("url"),
                    )
                )
            else:
                # Mark as unsupported
                attributions.append(
                    Attribution(
                        claim=claim,
                        source="Unsupported",
                        evidence="No matching evidence found",
                        similarity_score=0.0,
                    )
                )

        return attributions

    def calculate_attribution_coverage(self, attributions: List[Attribution]) -> float:
        """Calculate what percentage of claims are attributed."""
        if not attributions:
            return 0.0

        supported = sum(1 for a in attributions if a.source != "Unsupported")
        return supported / len(attributions)
