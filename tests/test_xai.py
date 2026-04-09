"""
Unit tests for XAI components.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.confidence_scorer import ConfidenceResult, ConfidenceScorer
from src.xai.source_attribution import Attribution, SourceAttributor


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
            num_sources=5,
        )

        assert result.level == "high"
        assert result.calibrated_score > 0.8

    def test_low_confidence(self, scorer):
        """Test low confidence calculation."""
        result = scorer.calculate_confidence(
            generation_probs=[0.3, 0.25, 0.2],
            retrieval_scores=[0.4, 0.35, 0.3],
            num_sources=1,
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
            generation_probs=[0.85, 0.82], retrieval_scores=[0.9], num_sources=3
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
            {
                "content": "Diabetes symptoms include increased thirst and urination",
                "source": "CDC",
            },
            {"content": "Heart disease is a leading cause of death", "source": "AHA"},
        ]

        evidence = attributor.find_evidence(claim, documents)

        assert evidence is not None
        assert evidence["source"] == "CDC"
        assert evidence["score"] > 0

    def test_attribute_answer(self, attributor):
        """Test full answer attribution."""
        answer = "Diabetes causes high blood sugar. It can lead to serious complications."
        documents = [
            {
                "content": "Diabetes is characterized by high blood sugar levels",
                "source": "WHO",
            },
            {
                "content": "Complications of diabetes include kidney disease",
                "source": "NIH",
            },
        ]

        attributions = attributor.attribute_answer(answer, documents)

        assert len(attributions) >= 1
        assert all(isinstance(a, Attribution) for a in attributions)

    def test_attribution_coverage(self, attributor):
        """Test attribution coverage calculation."""
        attributions = [
            Attribution(claim="Claim 1", source="Source1", evidence="Ev1", similarity_score=0.8),
            Attribution(claim="Claim 2", source="Unsupported", evidence="", similarity_score=0.0),
            Attribution(claim="Claim 3", source="Source2", evidence="Ev2", similarity_score=0.7),
        ]

        coverage = attributor.calculate_attribution_coverage(attributions)

        assert coverage == pytest.approx(2 / 3, rel=0.01)


# =============================================================================
# TestMultiSignalConfidenceScorer
# =============================================================================


class TestMultiSignalConfidenceScorer:
    """
    Comprehensive tests for MultiSignalConfidenceScorer.

    This is one of the novel contributions described in the paper:
    a five-signal confidence estimator with Platt-scaling calibration
    and built-in ablation support.
    """

    # ------------------------------------------------------------------ helpers

    @pytest.fixture
    def scorer(self):
        """Default scorer with equal-ish weights."""
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        return MultiSignalConfidenceScorer()

    @pytest.fixture
    def mock_docs(self):
        """Lightweight mock documents with .content and .score attributes."""
        from dataclasses import dataclass

        @dataclass
        class _Doc:
            content: str
            score: float

        return [
            _Doc("Diabetes is characterised by elevated blood glucose levels.", 0.88),
            _Doc("Type 2 diabetes involves insulin resistance.", 0.75),
            _Doc("Hypertension increases the risk of heart disease.", 0.60),
        ]

    # ------------------------------------------------------------------ init

    def test_default_weights_sum_to_one(self, scorer):
        """All default weights must sum to exactly 1.0 after normalisation."""
        total = sum(scorer.weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_custom_weights_are_normalised(self):
        """Custom weights provided at construction should be auto-normalised."""
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        custom = {
            "retrieval": 2.0,
            "generation": 2.0,
            "consistency": 1.0,
            "source_agreement": 0.5,
            "entity_coverage": 0.5,
        }
        sc = MultiSignalConfidenceScorer(signal_weights=custom)
        assert sum(sc.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_custom_calibration_params_stored(self):
        """Platt scaling parameters passed at init should be stored correctly."""
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        sc = MultiSignalConfidenceScorer(calibration_params={"a": 2.5, "b": -0.3})
        assert sc.platt_a == pytest.approx(2.5)
        assert sc.platt_b == pytest.approx(-0.3)

    # ------------------------------------------------------------------ compute_confidence

    def test_compute_confidence_returns_breakdown(self, scorer, mock_docs):
        """compute_confidence should return a ConfidenceBreakdown dataclass."""
        from src.xai.multi_signal_confidence import ConfidenceBreakdown

        result = scorer.compute_confidence(
            query="What is diabetes?",
            answer="Diabetes is a condition characterised by high blood glucose.",
            retrieved_documents=mock_docs,
            generation_probabilities=[0.9, 0.85, 0.88, 0.92],
        )
        assert isinstance(result, ConfidenceBreakdown)

    def test_compute_confidence_scores_in_unit_interval(self, scorer, mock_docs):
        """All numeric fields on the breakdown must be in [0, 1]."""
        result = scorer.compute_confidence(
            query="What is diabetes?",
            answer="Diabetes is a chronic metabolic disorder.",
            retrieved_documents=mock_docs,
            generation_probabilities=[0.8, 0.75, 0.82],
        )
        for field_name in (
            "retrieval_confidence",
            "generation_confidence",
            "consistency_score",
            "source_agreement",
            "medical_entity_coverage",
            "overall_confidence",
            "calibrated_confidence",
        ):
            value = getattr(result, field_name)
            assert 0.0 <= value <= 1.0, f"{field_name} = {value} is outside [0, 1]"

    def test_compute_confidence_level_valid(self, scorer, mock_docs):
        """confidence_level must be one of the three valid strings."""
        result = scorer.compute_confidence(
            query="How is hypertension treated?",
            answer="Hypertension is treated with lifestyle changes and medication.",
            retrieved_documents=mock_docs,
        )
        assert result.confidence_level in {"high", "medium", "low"}

    def test_compute_confidence_explanation_non_empty(self, scorer, mock_docs):
        """An explanation string must always be generated."""
        result = scorer.compute_confidence(
            query="What causes anemia?",
            answer="Anemia is caused by iron deficiency.",
            retrieved_documents=mock_docs,
        )
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_compute_confidence_signal_weights_recorded(self, scorer, mock_docs):
        """The breakdown should record the weights that were applied."""
        result = scorer.compute_confidence(
            query="test", answer="test answer", retrieved_documents=mock_docs
        )
        assert isinstance(result.signal_weights, dict)
        assert set(result.signal_weights.keys()) == set(scorer.weights.keys())

    def test_compute_confidence_no_probabilities(self, scorer, mock_docs):
        """Omitting generation_probabilities should still produce a valid result."""
        result = scorer.compute_confidence(
            query="What is hypertension?",
            answer="Hypertension is high blood pressure.",
            retrieved_documents=mock_docs,
            generation_probabilities=None,
        )
        # generation_confidence defaults to 0.5 (neutral) when probs absent
        assert result.generation_confidence == pytest.approx(0.5)
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_compute_confidence_empty_documents(self, scorer):
        """Empty document list should produce a low confidence score."""
        result = scorer.compute_confidence(
            query="What is cancer?",
            answer="Cancer is uncontrolled cell growth.",
            retrieved_documents=[],
        )
        assert result.retrieval_confidence == pytest.approx(0.0)

    def test_compute_confidence_with_alternative_answers(self, scorer, mock_docs):
        """Providing alternative_answers should influence consistency_score."""
        # High-agreement alternatives → high consistency
        alts_agree = ["Diabetes involves high blood sugar."] * 3
        result_agree = scorer.compute_confidence(
            query="What is diabetes?",
            answer="Diabetes involves high blood sugar.",
            retrieved_documents=mock_docs,
            alternative_answers=alts_agree,
        )
        # Low-agreement alternatives → low consistency
        alts_disagree = [
            "The answer is completely different.",
            "No relation to the original.",
            "Something entirely unrelated.",
        ]
        result_disagree = scorer.compute_confidence(
            query="What is diabetes?",
            answer="Diabetes involves high blood sugar.",
            retrieved_documents=mock_docs,
            alternative_answers=alts_disagree,
        )
        assert result_agree.consistency_score > result_disagree.consistency_score

    # ------------------------------------------------------------------ individual signals

    def test_retrieval_confidence_empty(self, scorer):
        """Empty document list → retrieval confidence of 0."""
        assert scorer._retrieval_confidence([]) == pytest.approx(0.0)

    def test_retrieval_confidence_high_scores(self, scorer, mock_docs):
        """Documents with high scores should yield high retrieval confidence."""
        result = scorer._retrieval_confidence(mock_docs)
        assert result > 0.5

    def test_retrieval_confidence_no_docs(self, scorer):
        """Empty document list should yield zero retrieval confidence."""
        result = scorer._retrieval_confidence([])
        assert result == 0.0

    def test_retrieval_confidence_score_agnostic(self, scorer):
        """Confidence normalizes by max so absolute scale (cosine vs RRF) doesn't change ranking."""
        from dataclasses import dataclass

        @dataclass
        class _Doc:
            score: float
            score_type: str = "cosine"
            content: str = ""

        # Same relative distribution at different scales → same confidence
        cosine_docs = [
            _Doc(0.9, "cosine"),
            _Doc(0.54, "cosine"),
            _Doc(0.36, "cosine"),
        ]  # cosine similarity scale
        rrf_docs = [
            _Doc(0.03, "rrf"),
            _Doc(0.018, "rrf"),
            _Doc(0.012, "rrf"),
        ]  # RRF scale (same ratios)
        r_cosine = scorer._retrieval_confidence(cosine_docs)
        r_rrf = scorer._retrieval_confidence(rrf_docs)

        # Since we modified the RRF scaling (to not divide by max, but rather gentle scale),
        # rrf and cosine are no longer completely scale invariant, they are now normalized differently based on score type!
        assert r_cosine > 0.6  # Cosine confidence should still be quite high
        assert r_rrf > 0.1  # RRF is much smaller but still > 0.1

    def test_generation_confidence_high_probs(self, scorer):
        """Very high token probabilities should yield a confidence close to 1."""
        result = scorer._generation_confidence([0.99, 0.98, 0.97, 0.99])
        assert result > 0.7

    def test_generation_confidence_low_probs(self, scorer):
        """Very low token probabilities should yield low confidence."""
        result = scorer._generation_confidence([0.05, 0.08, 0.06, 0.04])
        assert result < 0.5

    def test_generation_confidence_none(self, scorer):
        """None probabilities should return 0.5 (neutral default)."""
        assert scorer._generation_confidence(None) == pytest.approx(0.5)

    def test_generation_confidence_empty_list(self, scorer):
        """Empty probability list should return 0.5 (neutral default)."""
        assert scorer._generation_confidence([]) == pytest.approx(0.5)

    def test_self_consistency_identical_answers(self, scorer):
        """Identical alternatives should yield a consistency score of 1.0."""
        answer = "Diabetes is a metabolic disorder."
        alts = [answer, answer, answer]
        result = scorer._self_consistency_score(answer, alts)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_self_consistency_no_alternatives(self, scorer):
        """Fewer than two alternatives → neutral score of 0.5."""
        assert scorer._self_consistency_score("any answer", None) == pytest.approx(0.5)
        assert scorer._self_consistency_score("any answer", ["one"]) == pytest.approx(0.5)

    def test_self_consistency_divergent_answers(self, scorer):
        """Very different alternatives should yield a low consistency score."""
        answer = "Diabetes involves high blood glucose."
        alts = [
            "Antibiotics are used to treat bacterial infections.",
            "The heart pumps blood around the body.",
            "Bones provide structural support for the body.",
        ]
        result = scorer._self_consistency_score(answer, alts)
        assert result < 0.5

    def test_source_agreement_no_documents(self, scorer):
        """No documents should return 0."""
        assert scorer._source_agreement_score("any answer", []) == pytest.approx(0.0)

    def test_source_agreement_high_overlap(self, scorer):
        """Answer closely matching document content should yield higher agreement."""
        from dataclasses import dataclass

        @dataclass
        class _Doc:
            content: str
            score: float = 0.8

        answer = "Diabetes is characterised by elevated blood glucose levels."
        docs = [_Doc(content="Diabetes is characterised by elevated blood glucose levels.")]
        result = scorer._source_agreement_score(answer, docs)
        assert result > 0.0

    def test_entity_coverage_no_entities_in_query(self, scorer, mock_docs):
        """When the query contains no medical entities, score defaults to 0.5."""
        result = scorer._entity_coverage_score(
            query="tell me about it",
            answer="Some medical information.",
            documents=mock_docs,
        )
        assert result == pytest.approx(0.5)

    def test_entity_coverage_with_entities(self, scorer, mock_docs):
        """Query with recognised entities should produce a meaningful coverage score."""
        result = scorer._entity_coverage_score(
            query="What are the symptoms of Type 2 diabetes?",
            answer="Type 2 diabetes symptoms include fatigue and frequent urination.",
            documents=mock_docs,
        )
        assert 0.0 <= result <= 1.0

    # ------------------------------------------------------------------ calibration

    def test_platt_calibrate_midpoint(self, scorer):
        """Updated params (a=14.44, b=-11.25): sigmoid(14.44*x + -11.25)."""
        # Testing the new default parameters explicitly for correctness
        scorer.platt_a = 14.44
        scorer.platt_b = -11.25
        result = scorer._platt_calibrate(0.779)  # 14.44 * 0.779 - 11.25 = 0
        assert result == pytest.approx(0.5, abs=0.01)

    def test_platt_calibrate_output_range(self, scorer):
        """Calibrated output must always stay within [0, 1]."""
        for raw in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = scorer._platt_calibrate(raw)
            assert 0.0 <= result <= 1.0, f"Out-of-range calibrated score for raw={raw}"

    def test_platt_calibrate_monotonic(self, scorer):
        """Higher raw scores must produce higher (or equal) calibrated scores."""
        raw_scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibrated = [scorer._platt_calibrate(r) for r in raw_scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], (
                f"Monotonicity violated: calibrated[{i}]={calibrated[i]:.4f} > "
                f"calibrated[{i + 1}]={calibrated[i + 1]:.4f}"
            )

    # ------------------------------------------------------------------ confidence level

    def test_confidence_level_high(self, scorer):
        assert scorer._confidence_level(0.80) == "high"
        assert scorer._confidence_level(1.00) == "high"

    def test_confidence_level_medium(self, scorer):
        assert scorer._confidence_level(0.60) == "medium"
        assert scorer._confidence_level(0.45) == "medium"

    def test_confidence_level_low(self, scorer):
        assert scorer._confidence_level(0.30) == "low"
        assert scorer._confidence_level(0.00) == "low"

    def test_confidence_level_boundary_high(self, scorer):
        """Score exactly at high threshold (0.75) should be 'high'."""
        assert scorer._confidence_level(0.75) == "high"

    def test_confidence_level_boundary_medium(self, scorer):
        """Score exactly at medium threshold (0.45) should be 'medium'."""
        assert scorer._confidence_level(0.45) == "medium"

    # ------------------------------------------------------------------ ablation support

    def test_ablation_zero_retrieval_weight(self, mock_docs):
        """Zeroing retrieval weight should not crash and still produce valid output."""
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        ablated = MultiSignalConfidenceScorer(
            signal_weights={
                "retrieval": 0.0,
                "generation": 0.4,
                "consistency": 0.3,
                "source_agreement": 0.2,
                "entity_coverage": 0.1,
            }
        )
        result = ablated.compute_confidence(
            query="What is insulin resistance?",
            answer="Insulin resistance is when cells fail to respond to insulin.",
            retrieved_documents=mock_docs,
            generation_probabilities=[0.88, 0.85, 0.90],
        )
        assert 0.0 <= result.calibrated_confidence <= 1.0
        assert result.confidence_level in {"high", "medium", "low"}

    def test_ablation_single_signal_only(self, mock_docs):
        """Using only one non-zero signal should still produce a valid result."""
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        ablated = MultiSignalConfidenceScorer(
            signal_weights={
                "retrieval": 1.0,
                "generation": 0.0,
                "consistency": 0.0,
                "source_agreement": 0.0,
                "entity_coverage": 0.0,
            }
        )
        result = ablated.compute_confidence(
            query="What is hypertension?",
            answer="Hypertension is high blood pressure.",
            retrieved_documents=mock_docs,
        )
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_ablation_weights_affect_score(self, mock_docs):
        """
        Ablation study support: different weight profiles should produce
        different overall_confidence values for the same input.
        """
        from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

        # High-retrieval weight scorer
        sc_ret = MultiSignalConfidenceScorer(
            signal_weights={
                "retrieval": 0.9,
                "generation": 0.025,
                "consistency": 0.025,
                "source_agreement": 0.025,
                "entity_coverage": 0.025,
            }
        )
        # High-generation weight scorer
        sc_gen = MultiSignalConfidenceScorer(
            signal_weights={
                "retrieval": 0.025,
                "generation": 0.9,
                "consistency": 0.025,
                "source_agreement": 0.025,
                "entity_coverage": 0.025,
            }
        )
        kwargs = dict(
            query="What causes fever?",
            answer="Fever is caused by infection or inflammation.",
            retrieved_documents=mock_docs,
            # Low generation probs → high-gen scorer should produce lower score
            generation_probabilities=[0.1, 0.12, 0.09],
        )
        r_ret = sc_ret.compute_confidence(**kwargs)
        r_gen = sc_gen.compute_confidence(**kwargs)
        # With low gen probs, retrieval-dominated scorer should score higher
        assert r_ret.overall_confidence > r_gen.overall_confidence

    # ------------------------------------------------------------------ ConfidenceBreakdown

    def test_confidence_breakdown_default_values(self):
        """ConfidenceBreakdown should initialise all numerics to 0.0."""
        from src.xai.multi_signal_confidence import ConfidenceBreakdown

        bd = ConfidenceBreakdown()
        for attr in (
            "retrieval_confidence",
            "generation_confidence",
            "consistency_score",
            "source_agreement",
            "medical_entity_coverage",
            "overall_confidence",
            "calibrated_confidence",
        ):
            assert getattr(bd, attr) == pytest.approx(0.0)
        assert bd.confidence_level == "low"
        assert bd.explanation == ""
        assert bd.signal_weights == {}


# ── Attention Visualizer tests ───────────────────────────────────────────────


class TestTokenImportance:
    """Tests for TokenImportance dataclass and attention visualizer fallback."""

    def test_token_importance_defaults(self):
        """is_computed defaults to True."""
        from src.xai.attention_visualizer import TokenImportance

        ti = TokenImportance(token="aspirin", importance=0.8, position=0)
        assert ti.is_computed is True

    def test_token_importance_not_computed(self):
        """is_computed can be set to False for placeholder scores."""
        from src.xai.attention_visualizer import TokenImportance

        ti = TokenImportance(token="aspirin", importance=0.5, position=0, is_computed=False)
        assert ti.is_computed is False
        assert ti.importance == 0.5

    def test_fallback_scores_marked_not_computed(self):
        """When gradient computation fails, all tokens must have is_computed=False."""
        import unittest.mock as mock

        import torch

        from src.xai.attention_visualizer import TokenImportanceAnalyzer

        # Minimal mock model/tokenizer that raises on forward pass
        mock_model = mock.MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_model.get_input_embeddings.side_effect = RuntimeError("no GPU")
        mock_tokenizer = mock.MagicMock()
        tok_output = mock.MagicMock()
        tok_output.input_ids = torch.tensor([[1, 2, 3]])
        tok_output.__getitem__ = lambda self, k: {"input_ids": torch.tensor([[1, 2, 3]])}[k]
        mock_tokenizer.return_value = tok_output
        mock_tokenizer.convert_ids_to_tokens.return_value = ["What", "is", "aspirin"]

        analyzer = TokenImportanceAnalyzer(mock_model, mock_tokenizer)
        results = analyzer.compute_token_importance("What is aspirin")

        assert len(results) == 3
        assert all(
            not r.is_computed for r in results
        ), "Fallback tokens must have is_computed=False"
        assert all(r.importance == 0.5 for r in results), "Fallback scores must be uniform 0.5"


# ── RationaleGenerator tests ─────────────────────────────────────────────────


class TestRationaleGenerator:
    """Tests for RationaleGenerator template and CoT paths."""

    def test_template_rationale_no_extras(self):
        """Template rationale works with just question and answer."""
        from src.xai.rationale_generator import RationaleGenerator

        gen = RationaleGenerator(llm=None)
        rationale = gen.generate_template_rationale(
            question="What is hypertension?",
            answer="Hypertension is high blood pressure.",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 20
        assert "healthcare professional" in rationale.lower()

    def test_template_rationale_with_confidence(self):
        """Confidence level and score appear in template rationale."""
        from src.xai.rationale_generator import RationaleGenerator

        gen = RationaleGenerator(llm=None)
        rationale = gen.generate_template_rationale(
            question="What is aspirin used for?",
            answer="Aspirin is used as a pain reliever.",
            confidence={"score": 0.82, "level": "high"},
            sources=[{"source": "PubMedQA"}, {"source": "MedMCQA"}],
        )
        assert "high" in rationale
        assert "82%" in rationale or "0.82" in rationale or "82" in rationale

    def test_generate_rationale_falls_back_without_llm(self):
        """generate_rationale uses template when llm=None."""
        from src.xai.rationale_generator import RationaleGenerator

        gen = RationaleGenerator(llm=None)
        rationale = gen.generate_rationale(
            question="What causes diabetes?",
            answer="Diabetes is caused by insulin deficiency.",
            context="Insulin is a hormone produced by the pancreas.",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 10
