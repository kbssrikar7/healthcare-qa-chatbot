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

try:
    from .confidence_scorer import ConfidenceScorer
except Exception:
    ConfidenceScorer = None  # type: ignore

try:
    from .source_attribution import SourceAttributor
except Exception:
    SourceAttributor = None  # type: ignore

try:
    from .rationale_generator import RationaleGenerator
except Exception:
    RationaleGenerator = None  # type: ignore

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
