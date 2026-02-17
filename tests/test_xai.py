"""
Unit tests for XAI components.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.confidence_scorer import ConfidenceScorer, ConfidenceResult
from src.xai.source_attribution import SourceAttributor, Attribution


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""
    
    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()
    
    def test_high_confidence(self, scorer):
        """Test high confidence calculation."""
        result = scorer.calculate_confidence(
            generation_probs=[0.95, 0.93, 0.91, 0.89],
            retrieval_scores=[0.92, 0.88, 0.85],
            num_sources=5
        )
        
        assert result.level == "high"
        assert result.calibrated_score > 0.8
    
    def test_low_confidence(self, scorer):
        """Test low confidence calculation."""
        result = scorer.calculate_confidence(
            generation_probs=[0.3, 0.25, 0.2],
            retrieval_scores=[0.4, 0.35, 0.3],
            num_sources=1
        )
        
        assert result.level in ["low", "medium"]
        assert result.calibrated_score < 0.8
    
    def test_default_values(self, scorer):
        """Test with no probabilities/scores."""
        result = scorer.calculate_confidence()
        
        assert result.score > 0
        assert result.level in ["high", "medium", "low"]
        assert result.explanation is not None
    
    def test_confidence_levels(self, scorer):
        """Test confidence level determination."""
        assert scorer._get_confidence_level(0.9) == "high"
        assert scorer._get_confidence_level(0.6) == "medium"
        assert scorer._get_confidence_level(0.3) == "low"
    
    def test_confidence_colors(self, scorer):
        """Test confidence color mapping."""
        assert scorer.get_confidence_color("high") == "#28a745"
        assert scorer.get_confidence_color("medium") == "#ffc107"
        assert scorer.get_confidence_color("low") == "#dc3545"
    
    def test_calibration(self, scorer):
        """Test score calibration."""
        calibrated = scorer._calibrate(0.8)
        assert 0.0 <= calibrated <= 1.0
    
    def test_explanation_generated(self, scorer):
        """Test that explanation is generated."""
        result = scorer.calculate_confidence(
            generation_probs=[0.85, 0.82],
            retrieval_scores=[0.9],
            num_sources=3
        )
        
        assert len(result.explanation) > 0
        assert "3" in result.explanation or "sources" in result.explanation.lower()


class TestSourceAttributor:
    """Tests for SourceAttributor."""
    
    @pytest.fixture
    def attributor(self):
        return SourceAttributor()
    
    def test_extract_claims(self, attributor):
        """Test claim extraction from text."""
        text = "Diabetes affects blood sugar levels. It can cause increased thirst. What about treatment?"
        claims = attributor.extract_claims(text)
        
        assert len(claims) >= 1
        assert "?" not in claims[0]  # Should not include questions
    
    def test_extract_claims_filters_short(self, attributor):
        """Test that short sentences are filtered."""
        text = "OK. Yes. Diabetes is a condition that affects millions of people worldwide."
        claims = attributor.extract_claims(text)
        
        # Should only include the long sentence
        assert all(len(c) > 20 for c in claims)
    
    def test_text_overlap_similarity(self, attributor):
        """Test text overlap calculation."""
        claim = "Diabetes causes increased thirst"
        content = "Common symptoms include increased thirst and frequent urination"
        
        similarity = attributor._text_overlap(claim, content)
        assert similarity > 0
    
    def test_find_evidence(self, attributor):
        """Test finding evidence for a claim."""
        claim = "Diabetes symptoms include increased thirst"
        documents = [
            {"content": "Diabetes symptoms include increased thirst and urination", "source": "CDC"},
            {"content": "Heart disease is a leading cause of death", "source": "AHA"}
        ]
        
        evidence = attributor.find_evidence(claim, documents)
        
        assert evidence is not None
        assert evidence["source"] == "CDC"
        assert evidence["score"] > 0
    
    def test_attribute_answer(self, attributor):
        """Test full answer attribution."""
        answer = "Diabetes causes high blood sugar. It can lead to serious complications."
        documents = [
            {"content": "Diabetes is characterized by high blood sugar levels", "source": "WHO"},
            {"content": "Complications of diabetes include kidney disease", "source": "NIH"}
        ]
        
        attributions = attributor.attribute_answer(answer, documents)
        
        assert len(attributions) >= 1
        assert all(isinstance(a, Attribution) for a in attributions)
    
    def test_attribution_coverage(self, attributor):
        """Test attribution coverage calculation."""
        attributions = [
            Attribution(claim="Claim 1", source="Source1", evidence="Ev1", similarity_score=0.8),
            Attribution(claim="Claim 2", source="Unsupported", evidence="", similarity_score=0.0),
            Attribution(claim="Claim 3", source="Source2", evidence="Ev2", similarity_score=0.7)
        ]
        
        coverage = attributor.calculate_attribution_coverage(attributions)
        
        assert coverage == pytest.approx(2/3, rel=0.01)
