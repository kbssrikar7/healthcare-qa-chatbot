"""
Confidence scoring and calibration for medical QA.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
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
    """
    
    # Thresholds for confidence levels
    HIGH_THRESHOLD = 0.8
    MEDIUM_THRESHOLD = 0.5
    
    def __init__(
        self,
        calibration_temperature: float = 1.0,
        retrieval_weight: float = 0.4,
        generation_weight: float = 0.6
    ):
        self.calibration_temperature = calibration_temperature
        self.retrieval_weight = retrieval_weight
        self.generation_weight = generation_weight
    
    def calculate_confidence(
        self,
        generation_probs: Optional[List[float]] = None,
        retrieval_scores: Optional[List[float]] = None,
        num_sources: int = 0
    ) -> ConfidenceResult:
        """Calculate overall confidence score."""
        
        # Generation confidence (from token probabilities)
        if generation_probs and len(generation_probs) > 0:
            gen_confidence = np.mean(generation_probs)
            # Find low confidence tokens
            low_conf_indices = [
                i for i, p in enumerate(generation_probs) 
                if p < self.MEDIUM_THRESHOLD
            ]
        else:
            gen_confidence = 0.7  # Default
            low_conf_indices = []
        
        # Retrieval confidence (from similarity scores)
        if retrieval_scores and len(retrieval_scores) > 0:
            ret_confidence = np.mean(retrieval_scores[:3])  # Top 3 sources
        else:
            ret_confidence = 0.5  # Default
        
        # Source count bonus
        source_bonus = min(num_sources * 0.05, 0.2)  # Max 0.2 bonus
        
        # Combined confidence
        raw_score = (
            self.generation_weight * gen_confidence +
            self.retrieval_weight * ret_confidence +
            source_bonus
        )
        
        # Calibrate score
        calibrated_score = self._calibrate(raw_score)
        
        # Determine level
        level = self._get_confidence_level(calibrated_score)
        
        # Build explanation
        explanation = self._build_explanation(
            calibrated_score, level, num_sources, len(low_conf_indices)
        )
        
        # Low confidence spans (placeholder - would need token positions)
        low_conf_spans = [{"index": i} for i in low_conf_indices[:5]]
        
        return ConfidenceResult(
            score=raw_score,
            level=level,
            calibrated_score=calibrated_score,
            low_confidence_spans=low_conf_spans,
            explanation=explanation
        )
    
    def _calibrate(self, score: float) -> float:
        """Apply temperature scaling for calibration."""
        # Simple temperature scaling: score^(1/T)
        calibrated = score ** (1 / self.calibration_temperature)
        return min(max(calibrated, 0.0), 1.0)
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= self.HIGH_THRESHOLD:
            return "high"
        elif score >= self.MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def _build_explanation(
        self,
        score: float,
        level: str,
        num_sources: int,
        num_low_conf_tokens: int
    ) -> str:
        """Build human-readable confidence explanation."""
        explanations = {
            "high": (
                f"High confidence ({score:.0%}). "
                f"This answer is well-supported by {num_sources} authoritative sources."
            ),
            "medium": (
                f"Medium confidence ({score:.0%}). "
                f"This answer is based on {num_sources} sources, but some uncertainty remains. "
                "Consider consulting additional sources or a healthcare professional."
            ),
            "low": (
                f"Low confidence ({score:.0%}). "
                "Limited supporting evidence was found. "
                "Please consult a healthcare professional for reliable information."
            )
        }
        
        explanation = explanations.get(level, explanations["medium"])
        
        if num_low_conf_tokens > 3:
            explanation += f" ({num_low_conf_tokens} uncertain parts detected)"
        
        return explanation
    
    def get_confidence_color(self, level: str) -> str:
        """Get color code for confidence level."""
        colors = {
            "high": "#28a745",    # Green
            "medium": "#ffc107",  # Yellow
            "low": "#dc3545"      # Red
        }
        return colors.get(level, "#6c757d")
