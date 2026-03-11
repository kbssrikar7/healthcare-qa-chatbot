"""
Confidence scoring and calibration for medical QA.

Fix (v2): All numpy operations now explicitly cast to Python-native float()
so that ConfidenceResult fields are always plain Python floats.
This prevents msgpack / JSON serialization errors when LangGraph's
MemorySaver tries to checkpoint the state (numpy.float64 is not
msgpack-serializable).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ConfidenceResult:
    """Confidence scoring result."""

    score: float
    level: str  # high, medium, low
    calibrated_score: float
    low_confidence_spans: List[Dict]
    explanation: str


class ConfidenceScorer:
    """
    Calculate and calibrate confidence scores for generated answers.

    All internal numpy computations are explicitly cast to Python float
    before being stored so that callers never receive numpy scalar types.
    """

    # Thresholds for confidence levels
    HIGH_THRESHOLD = 0.8
    MEDIUM_THRESHOLD = 0.5

    def __init__(
        self,
        calibration_temperature: float = 1.0,
        retrieval_weight: float = 0.4,
        generation_weight: float = 0.6,
    ):
        self.calibration_temperature = float(calibration_temperature)
        self.retrieval_weight = float(retrieval_weight)
        self.generation_weight = float(generation_weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        generation_probs: Optional[List[float]] = None,
        retrieval_scores: Optional[List[float]] = None,
        num_sources: int = 0,
    ) -> ConfidenceResult:
        """Calculate overall confidence score.

        Returns a ConfidenceResult whose numeric fields are guaranteed to
        be plain Python floats (not numpy.float64).
        """

        # --- Generation confidence (from token probabilities) ---
        if generation_probs and len(generation_probs) > 0:
            # float() ensures numpy.float64 → Python float
            gen_confidence = float(np.mean(generation_probs))
            low_conf_indices: List[int] = [
                i for i, p in enumerate(generation_probs) if p < self.MEDIUM_THRESHOLD
            ]
        else:
            gen_confidence = 0.7  # Python float default
            low_conf_indices = []

        # --- Retrieval confidence (from similarity / RRF scores) ---
        if retrieval_scores and len(retrieval_scores) > 0:
            # float() ensures numpy.float64 → Python float
            ret_confidence = float(np.mean(retrieval_scores[:3]))
        else:
            ret_confidence = 0.5  # Python float default

        # --- Source-count bonus ---
        source_bonus = float(min(num_sources * 0.05, 0.2))  # max 0.2

        # --- Combined raw score (all operands are now Python float) ---
        raw_score = float(
            self.generation_weight * gen_confidence
            + self.retrieval_weight * ret_confidence
            + source_bonus
        )

        # --- Calibrate ---
        calibrated_score = self._calibrate(raw_score)

        # --- Level & explanation ---
        level = self._get_confidence_level(calibrated_score)
        explanation = self._build_explanation(
            calibrated_score, level, num_sources, len(low_conf_indices)
        )

        low_conf_spans = [{"index": int(i)} for i in low_conf_indices[:5]]

        return ConfidenceResult(
            score=raw_score,
            level=level,
            calibrated_score=calibrated_score,
            low_confidence_spans=low_conf_spans,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self, score: float) -> float:
        """Apply temperature scaling for calibration.

        Always returns a plain Python float.
        """
        # score ** (1/T)  — if score happened to be numpy type, float() cleans it
        calibrated = float(score) ** (1.0 / self.calibration_temperature)
        # clamp to [0, 1] using plain Python min/max to avoid numpy return types
        return float(min(max(calibrated, 0.0), 1.0))

    def _get_confidence_level(self, score: float) -> str:
        """Map a calibrated score to a human-readable level string."""
        if score >= self.HIGH_THRESHOLD:
            return "high"
        elif score >= self.MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    def _build_explanation(
        self,
        score: float,
        level: str,
        num_sources: int,
        num_low_conf_tokens: int,
    ) -> str:
        """Build a human-readable confidence explanation."""
        explanations = {
            "high": (
                f"High confidence ({score:.0%}). "
                f"This answer is well-supported by {num_sources} authoritative sources."
            ),
            "medium": (
                f"Medium confidence ({score:.0%}). "
                f"This answer is based on {num_sources} sources, but some uncertainty "
                "remains. Consider consulting additional sources or a healthcare "
                "professional."
            ),
            "low": (
                f"Low confidence ({score:.0%}). "
                "Limited supporting evidence was found. "
                "Please consult a healthcare professional for reliable information."
            ),
        }

        explanation = explanations.get(level, explanations["medium"])

        if num_low_conf_tokens > 3:
            explanation += f" ({num_low_conf_tokens} uncertain parts detected)"

        return explanation

    def get_confidence_color(self, level: str) -> str:
        """Return a CSS hex colour for the given confidence level."""
        colors = {
            "high": "#28a745",  # green
            "medium": "#ffc107",  # amber
            "low": "#dc3545",  # red
        }
        return colors.get(level, "#6c757d")
