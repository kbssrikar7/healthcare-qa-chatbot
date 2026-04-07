"""
Hallucination detection for medical QA.

Detects potential hallucinations using:
1. Source-grounding check (token overlap with retrieved documents)
2. Medical sanity rules (plausibility of vital signs / dosages)
3. Self-contradiction detection (negation + high term overlap)
4. Fabricated citation detection (regex for unverifiable references)
5. Optional NLI-based entailment verification (cross-encoder/nli-deberta)

All methods are CPU-safe; heavy NLI model is lazily loaded only if
``use_nli=True`` is passed at construction time.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from loguru import logger


class HallucinationType(Enum):
    FABRICATION = "fabrication"  # Contradicts known medical facts
    EXTRINSIC   = "extrinsic"   # Not supported by retrieved sources
    INTRINSIC   = "intrinsic"   # Internally contradicts the answer itself
    NONE        = "none"


@dataclass
class HallucinationResult:
    """Result of hallucination detection analysis."""
    has_hallucination: bool
    hallucination_type: HallucinationType
    hallucination_score: float          # 0.0 = clean, 1.0 = definite hallucination
    flagged_claims: List[Dict[str, Any]]
    medical_accuracy_flags: List[str]
    explanation: str


# ---------------------------------------------------------------------------
# Sanity rules: (regex pattern, validator callable, description)
# ---------------------------------------------------------------------------
def _make_rules() -> List[Dict]:
    return [
        {
            "pattern":     r"normal\s+(?:body\s+)?temperature[^.]*?(\d+\.?\d*)\s*°?F",
            "n_groups":    1,
            "check":       lambda v: 96.0 <= float(v) <= 100.4,
            "description": "Normal body temperature (°F)",
        },
        {
            "pattern":     r"normal\s+(?:body\s+)?temperature[^.]*?(\d+\.?\d*)\s*°?C",
            "n_groups":    1,
            "check":       lambda v: 35.5 <= float(v) <= 38.0,
            "description": "Normal body temperature (°C)",
        },
        {
            "pattern":     r"normal\s+(?:resting\s+)?heart\s+rate[^.]*?(\d+)\s*(?:bpm|beats)",
            "n_groups":    1,
            "check":       lambda v: 40 <= int(v) <= 130,
            "description": "Normal heart rate (bpm)",
        },
        {
            "pattern":     r"normal\s+blood\s+pressure[^.]*?(\d{2,3})\s*/\s*(\d{2,3})",
            "n_groups":    2,
            "check":       lambda v: 60 <= int(v[0]) <= 200 and 40 <= int(v[1]) <= 130,
            "description": "Plausible blood pressure (mmHg)",
        },
    ]


class HallucinationDetector:
    """
    Multi-method hallucination detection for medical answers.

    Usage
    -----
    >>> detector = HallucinationDetector()
    >>> result = detector.detect(answer, retrieved_documents, query)
    >>> if result.has_hallucination:
    ...     print(result.explanation)
    """

    NEGATION_WORDS = frozenset({
        "not", "no", "never", "neither", "nor",
        "cannot", "can't", "don't", "doesn't",
        "isn't", "aren't", "won't", "without",
    })
    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be",
        "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall",
        "of", "in", "on", "at", "to", "for", "by", "with",
        "this", "that", "these", "those",
    })

    def __init__(self, use_nli: bool = False):
        """
        Parameters
        ----------
        use_nli : Load a DeBERTa NLI model for entailment checking.
                  Requires ~500 MB disk space. Default is False (CPU-safe).
        """
        self.use_nli = use_nli
        self._nli_pipeline = None
        self._rules = _make_rules()

    def warm_up(self) -> None:
        """Pre-load NLI model to avoid first-query latency spike."""
        if self.use_nli and self._nli_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._nli_pipeline = hf_pipeline(
                    "text-classification",
                    model="microsoft/deberta-base-mnli",
                    device=-1,
                )
                logger.info("NLI pipeline pre-loaded successfully")
            except Exception as e:
                logger.warning(f"NLI pipeline warm-up failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
        query: str = "",
    ) -> HallucinationResult:
        """
        Run all hallucination checks on a generated answer.

        Parameters
        ----------
        answer : The LLM-generated answer text.
        retrieved_documents : List of dicts with at least a ``content`` key.
        query : The original user query (used for context, optional).

        Returns
        -------
        HallucinationResult with detailed flags and score.
        """
        if not answer:
            return HallucinationResult(
                has_hallucination=False,
                hallucination_type=HallucinationType.NONE,
                hallucination_score=0.0,
                flagged_claims=[],
                medical_accuracy_flags=[],
                explanation="Empty answer — nothing to check.",
            )

        flagged_claims: List[Dict[str, Any]] = []
        medical_flags: List[str] = []
        scores: List[float] = []

        # Combine retrieved text for grounding checks
        context = "\n".join(
            doc.get("content", "") if isinstance(doc, dict) else str(doc)
            for doc in retrieved_documents
        )

        # 1. Source-grounding check (term overlap)
        grounding_score = self._check_source_grounding(answer, context)
        scores.append(1.0 - grounding_score)  # invert: 0 = well-grounded
        if grounding_score < 0.25:
            flagged_claims.append({
                "check": "source_grounding",
                "detail": f"Low overlap with sources ({grounding_score:.2f})",
            })

        # 2. Medical sanity rules
        rule_flags = self._check_medical_sanity(answer)
        medical_flags.extend(rule_flags)
        if rule_flags:
            scores.append(0.8)

        # 3. Self-contradiction detection
        contradictions = self._check_self_contradiction(answer)
        if contradictions:
            flagged_claims.extend(contradictions)
            scores.append(0.7)

        # 4. Fabricated citation detection
        fab_citations = self._check_fabricated_citations(answer)
        if fab_citations:
            flagged_claims.extend(fab_citations)
            scores.append(0.6)

        # 5. Optional NLI entailment
        if self.use_nli:
            nli_score = self._nli_check(answer, context)
            scores.append(1.0 - nli_score)

        # Aggregate
        hallucination_score = max(scores) if scores else 0.0
        has_hallucination = hallucination_score >= 0.5

        # Determine type
        if medical_flags:
            h_type = HallucinationType.FABRICATION
        elif any(c.get("check") == "self_contradiction" for c in flagged_claims):
            h_type = HallucinationType.INTRINSIC
        elif has_hallucination:
            h_type = HallucinationType.EXTRINSIC
        else:
            h_type = HallucinationType.NONE

        explanation_parts = []
        if not has_hallucination:
            explanation_parts.append("No hallucination signals detected.")
        else:
            if grounding_score < 0.25:
                explanation_parts.append(
                    f"Answer poorly grounded in sources (overlap={grounding_score:.0%})."
                )
            if medical_flags:
                explanation_parts.append(
                    f"Medical sanity issues: {', '.join(medical_flags)}"
                )
            if any(c.get("check") == "self_contradiction" for c in flagged_claims):
                explanation_parts.append("Self-contradictory statements detected.")
            if any(c.get("check") == "fabricated_citation" for c in flagged_claims):
                explanation_parts.append("Potentially fabricated citations found.")

        return HallucinationResult(
            has_hallucination=has_hallucination,
            hallucination_type=h_type,
            hallucination_score=round(hallucination_score, 3),
            flagged_claims=flagged_claims,
            medical_accuracy_flags=medical_flags,
            explanation=" ".join(explanation_parts),
        )

    # ------------------------------------------------------------------
    # 1. Source-grounding (term overlap)
    # ------------------------------------------------------------------
    def _check_source_grounding(self, answer: str, context: str) -> float:
        """Return fraction of answer key-terms found in context (0..1)."""
        answer_terms = self._key_terms(answer)
        if not answer_terms:
            return 1.0
        context_lower = context.lower()
        found = sum(1 for t in answer_terms if t in context_lower)
        return found / len(answer_terms)

    # ------------------------------------------------------------------
    # 2. Medical sanity rules
    # ------------------------------------------------------------------
    def _check_medical_sanity(self, answer: str) -> List[str]:
        flags: List[str] = []
        for rule in self._rules:
            match = re.search(rule["pattern"], answer, re.IGNORECASE)
            if not match:
                continue
            try:
                if rule["n_groups"] == 1:
                    value = match.group(1)
                    if not rule["check"](value):
                        flags.append(
                            f"{rule['description']}: {value} is implausible"
                        )
                elif rule["n_groups"] == 2:
                    values = (match.group(1), match.group(2))
                    if not rule["check"](values):
                        flags.append(
                            f"{rule['description']}: {values} is implausible"
                        )
            except (ValueError, IndexError):
                continue
        return flags

    # ------------------------------------------------------------------
    # 3. Self-contradiction detection
    # ------------------------------------------------------------------
    def _check_self_contradiction(self, answer: str) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if len(sentences) < 2:
            return []

        contradictions: List[Dict[str, Any]] = []
        for i, s1 in enumerate(sentences):
            for s2 in sentences[i + 1:]:
                terms1 = self._key_terms(s1)
                terms2 = self._key_terms(s2)
                if not terms1 or not terms2:
                    continue
                overlap = len(terms1 & terms2) / max(len(terms1), len(terms2))
                neg1 = any(w in s1.lower().split() for w in self.NEGATION_WORDS)
                neg2 = any(w in s2.lower().split() for w in self.NEGATION_WORDS)
                if overlap > 0.5 and neg1 != neg2:
                    contradictions.append({
                        "check": "self_contradiction",
                        "sentence_a": s1[:80],
                        "sentence_b": s2[:80],
                    })
        return contradictions

    # ------------------------------------------------------------------
    # 4. Fabricated citation detection
    # ------------------------------------------------------------------
    _CITATION_PATTERNS = [
        r'\(\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\s*\)',  # (Smith et al., 2023)
        r'according\s+to\s+(?:a\s+)?(?:recent\s+)?(?:study|research|paper)',
        r'published\s+in\s+(?:the\s+)?[A-Z]',
        r'\b(?:doi|DOI)\s*:\s*10\.\d{4,}',
    ]

    def _check_fabricated_citations(self, answer: str) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        for pattern in self._CITATION_PATTERNS:
            for match in re.finditer(pattern, answer):
                flags.append({
                    "check": "fabricated_citation",
                    "matched": match.group()[:60],
                })
        return flags

    # ------------------------------------------------------------------
    # 5. Optional NLI entailment
    # ------------------------------------------------------------------
    def _nli_check(self, answer: str, context: str) -> float:
        """Return average entailment probability (0..1) using DeBERTa NLI.

        Uses microsoft/deberta-base-mnli which is cached locally.
        Input format: {"text": premise, "text_pair": hypothesis} for correct
        NLI pair encoding (avoids manual [SEP] injection which deberta ignores).
        """
        if self._nli_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                # microsoft/deberta-base-mnli is cached locally under ~/.cache/huggingface/hub/
                # Labels: CONTRADICTION / NEUTRAL / ENTAILMENT
                self._nli_pipeline = hf_pipeline(
                    "text-classification",
                    model="microsoft/deberta-base-mnli",
                    device=-1,
                )
            except Exception as e:
                logger.warning(f"NLI pipeline load failed, returning neutral score: {e}")
                return 0.5  # neutral fallback

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 15]
        if not sentences:
            return 0.5

        entail_scores: List[float] = []
        premise = context[:512]
        for sent in sentences[:5]:
            try:
                # Pass as a dict so the tokenizer uses proper pair encoding
                result = self._nli_pipeline({"text": premise, "text_pair": sent})[0]
                if "ENTAIL" in result["label"].upper():
                    entail_scores.append(result["score"])
                else:
                    entail_scores.append(0.0)
            except Exception as e:
                logger.warning(f"NLI sentence check failed: {e}")
                entail_scores.append(0.5)

        return sum(entail_scores) / len(entail_scores) if entail_scores else 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _key_terms(self, text: str) -> set:
        words = set(text.lower().split())
        return {w for w in words - self.STOP_WORDS if len(w) > 3}
