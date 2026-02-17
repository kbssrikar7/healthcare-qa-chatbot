"""
Explainable AI (XAI) module for Healthcare QA.

Provides:
- Confidence scoring and calibration
- Source attribution for claims
- Rationale generation
- Token importance analysis (SHAP-like)
- Attention visualization
- Counterfactual explanations
"""

from .confidence_scorer import ConfidenceScorer
from .source_attribution import SourceAttributor
from .rationale_generator import RationaleGenerator

__all__ = [
    'ConfidenceScorer',
    'SourceAttributor', 
    'RationaleGenerator'
]

# Lazy imports for heavier modules
def get_attention_visualizer():
    from .attention_visualizer import TokenImportanceAnalyzer, AttentionExtractor
    return TokenImportanceAnalyzer, AttentionExtractor

def get_counterfactual_explainer():
    from .attention_visualizer import CounterfactualExplainer
    return CounterfactualExplainer
