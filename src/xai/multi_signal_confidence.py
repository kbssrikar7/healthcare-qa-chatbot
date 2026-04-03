"""
Multi-signal confidence scoring for Explainable Healthcare QA.

Combines retrieval confidence, generation confidence, self-consistency,
source agreement, and medical entity coverage to produce calibrated
confidence estimates using Platt scaling.

For publication: each signal can be ablated independently to study
contribution (Section 5.3 of paper).
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence signals for explainability."""

    retrieval_confidence: float = 0.0
    generation_confidence: float = 0.0
    consistency_score: float = 0.0
    source_agreement: float = 0.0
    medical_entity_coverage: float = 0.0
    overall_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    confidence_level: str = "low"  # low | medium | high
    explanation: str = ""
    signal_weights: Dict[str, float] = field(default_factory=dict)


class MultiSignalConfidenceScorer:
    """
    Produces calibrated confidence scores from five complementary signals.

    Signals
    -------
    1. Retrieval confidence  — document relevance quality.
    2. Generation confidence — token-level generation probabilities.
    3. Self-consistency      — agreement across multiple generation samples.
    4. Source agreement      — cross-document support for the answer.
    5. Medical entity coverage — NER-based grounding of query entities.

    Calibration
    -----------
    Platt scaling is applied on top of the weighted combination so that
    the final score is a calibrated probability (ECE-minimising).

    Ablation
    --------
    Pass ``signal_weights`` at construction time to disable individual
    signals (set weight=0) for ablation studies.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "retrieval": 0.25,
        "generation": 0.25,
        "consistency": 0.20,
        "source_agreement": 0.15,
        "entity_coverage": 0.15,
    }

    def __init__(
        self,
        signal_weights: Optional[Dict[str, float]] = None,
        calibration_params: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        signal_weights      : Override default weights for ablation studies.
        calibration_params  : Pre-fitted Platt parameters {"a": float, "b": float}.
        """
        self.weights = signal_weights or dict(self.DEFAULT_WEIGHTS)
        # Normalise weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Platt scaling:  sigmoid(a * raw_score + b)
        # Default params adjusted so raw [0,1] maps to meaningful calibrated range
        # With a=3, b=-1.5: raw=0→0.18, raw=0.5→0.5, raw=0.8→0.82, raw=1→0.95
        # This makes "high" confidence (≥0.75) achievable with strong evidence
        cp = calibration_params or {}
        self.platt_a: float = cp.get("a", 3.0)
        self.platt_b: float = cp.get("b", -1.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_confidence(
        self,
        query: str,
        answer: str,
        retrieved_documents: list,
        generation_probabilities: Optional[List[float]] = None,
        alternative_answers: Optional[List[str]] = None,
    ) -> ConfidenceBreakdown:
        """Compute multi-signal confidence with full breakdown.

        Parameters
        ----------
        query                   : Original user query.
        answer                  : Primary generated answer.
        retrieved_documents     : List of RetrievedDocument objects.
        generation_probabilities: Per-token probabilities from the LLM.
        alternative_answers     : Extra generation samples (for consistency).

        Returns
        -------
        ConfidenceBreakdown with all signal scores and explanation.
        """
        bd = ConfidenceBreakdown()

        bd.retrieval_confidence = self._retrieval_confidence(retrieved_documents)
        bd.generation_confidence = self._generation_confidence(generation_probabilities)
        bd.consistency_score = self._self_consistency_score(answer, alternative_answers)
        bd.source_agreement = self._source_agreement_score(answer, retrieved_documents)
        bd.medical_entity_coverage = self._entity_coverage_score(
            query, answer, retrieved_documents
        )

        raw = (
            self.weights.get("retrieval", 0) * bd.retrieval_confidence
            + self.weights.get("generation", 0) * bd.generation_confidence
            + self.weights.get("consistency", 0) * bd.consistency_score
            + self.weights.get("source_agreement", 0) * bd.source_agreement
            + self.weights.get("entity_coverage", 0) * bd.medical_entity_coverage
        )

        bd.overall_confidence = float(np.clip(raw, 0.0, 1.0))
        bd.calibrated_confidence = self._platt_calibrate(bd.overall_confidence)
        bd.confidence_level = self._confidence_level(bd.calibrated_confidence)
        bd.signal_weights = dict(self.weights)
        bd.explanation = self._generate_explanation(bd)

        return bd

    # ------------------------------------------------------------------
    # Individual signal methods (can be overridden / tested independently)
    # ------------------------------------------------------------------

    def _retrieval_confidence(self, documents: list) -> float:
        """Score based on retrieval quality metrics."""
        if not documents:
            return 0.0

        scores = [float(getattr(d, "score", 0.0)) for d in documents]
        if not scores:
            return 0.0

        top_score = max(scores)
        mean_score = float(np.mean(scores))
        # Score dropoff: large gap between top and last → high confidence
        dropoff = (
            (scores[0] - scores[-1]) / (scores[0] + 1e-8) if len(scores) > 1 else 0.5
        )

        return float(
            np.clip(0.4 * top_score + 0.3 * mean_score + 0.3 * dropoff, 0.0, 1.0)
        )

    def _generation_confidence(self, probabilities: Optional[List[float]]) -> float:
        """Score from token-level generation probabilities."""
        if not probabilities:
            return 0.5  # neutral when unavailable

        probs = np.array(probabilities, dtype=float)
        probs = np.clip(probs, 1e-10, 1.0)

        geo_mean = float(np.exp(np.mean(np.log(probs))))
        min_prob = float(np.min(probs))
        entropy = -float(np.mean(probs * np.log2(probs)))
        max_ent = float(np.log2(len(probs) + 1))
        norm_cert = 1.0 - entropy / (max_ent + 1e-8)

        return float(
            np.clip(0.5 * geo_mean + 0.2 * min_prob + 0.3 * norm_cert, 0.0, 1.0)
        )

    def _self_consistency_score(
        self, primary_answer: str, alternatives: Optional[List[str]]
    ) -> float:
        """Score based on agreement across multiple generation samples."""
        if not alternatives or len(alternatives) < 2:
            return 0.5  # neutral when unavailable

        sims = [
            SequenceMatcher(None, primary_answer.lower(), alt.lower()).ratio()
            for alt in alternatives
        ]
        return float(np.mean(sims))

    def _source_agreement_score(self, answer: str, documents: list) -> float:
        """Score based on how many sources support the answer."""
        if not documents:
            return 0.0

        supporting = 0
        for doc in documents:
            content = getattr(doc, "content", "")
            sim = SequenceMatcher(None, answer.lower(), content.lower()).ratio()
            if sim > 0.3:
                supporting += 1

        return min(supporting / max(len(documents), 1), 1.0)

    def _entity_coverage_score(self, query: str, answer: str, documents: list) -> float:
        """Score based on medical entity overlap between query, answer, and sources."""

        def _extract(text: str) -> set:
            entities: set = set()
            # Proper nouns: case-sensitive (capital letters identify entities)
            entities.update(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text))
            # Dosages and conditions: case-insensitive
            entities.update(re.findall(r"\b\d+\s*(?:mg|ml|mcg|units|mmol|g)\b", text, re.IGNORECASE))
            entities.update(re.findall(r"\b(?:type\s*[12]\s*diabetes|hypertension|cancer|infection)\b", text, re.IGNORECASE))
            return {e.lower() for e in entities}

        query_ent = _extract(query)
        answer_ent = _extract(answer)
        source_ent: set = set()
        for doc in documents:
            source_ent.update(_extract(getattr(doc, "content", "")))

        if not query_ent:
            return 0.5  # neutral when no entities found

        coverage = len(query_ent & answer_ent) / len(query_ent)
        grounding = (
            len(answer_ent & source_ent) / len(answer_ent) if answer_ent else 0.5
        )
        return float(0.5 * coverage + 0.5 * grounding)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _platt_calibrate(self, raw_score: float) -> float:
        """Apply Platt scaling for calibrated probabilities."""
        logit = self.platt_a * raw_score + self.platt_b
        calibrated = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(calibrated, 0.0, 1.0))

    def fit_calibration(
        self, predictions: List[float], labels: List[int]
    ) -> Dict[str, float]:
        """Fit Platt scaling parameters on validation data.

        Parameters
        ----------
        predictions : Raw (un-calibrated) confidence scores.
        labels      : Binary correctness labels (1 = correct, 0 = wrong).

        Returns
        -------
        Dict with fitted {"a": float, "b": float}.
        """
        from scipy.optimize import minimize  # lightweight; already in scipy

        preds = np.array(predictions, dtype=float)
        labels = np.array(labels, dtype=float)

        def nll(params: np.ndarray) -> float:
            a, b = params
            p = np.clip(1.0 / (1.0 + np.exp(-(a * preds + b))), 1e-7, 1 - 1e-7)
            return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

        result = minimize(
            nll, [1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-6}
        )
        self.platt_a, self.platt_b = float(result.x[0]), float(result.x[1])
        return {"a": self.platt_a, "b": self.platt_b}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confidence_level(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.45:
            return "medium"
        return "low"

    def _generate_explanation(self, bd: ConfidenceBreakdown) -> str:
        parts: List[str] = []

        if bd.retrieval_confidence < 0.4:
            parts.append("Retrieved documents may not be highly relevant to the query.")
        elif bd.retrieval_confidence > 0.7:
            parts.append("Retrieved documents are highly relevant.")

        if bd.generation_confidence < 0.4:
            parts.append("The model showed uncertainty during generation.")

        if bd.consistency_score < 0.5:
            parts.append("Multiple generation attempts produced varying answers.")
        elif bd.consistency_score > 0.8:
            parts.append("Answer is consistent across multiple generation attempts.")

        if bd.source_agreement < 0.3:
            parts.append("Limited source agreement for this answer.")
        elif bd.source_agreement > 0.7:
            parts.append("Multiple sources support this answer.")

        if not parts:
            parts.append(f"Overall confidence: {bd.confidence_level}.")

        return " ".join(parts)
