"""
Integration tests for LangChain and LangGraph pipelines.

Verifies that both pipeline variants produce valid QAResponse-compatible
results using mock components without requiring GPU or model downloads.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import MockDocument, MockLLM, MockRetriever

# =========================================================================
# Shared test data
# =========================================================================
SAMPLE_DOCS = [
    {
        "content": "Diabetes mellitus is a metabolic disease characterized by high blood sugar levels. "
        "Type 2 diabetes is the most common form and involves insulin resistance.",
        "source": "MedQuAD",
        "score": 0.85,
        "metadata": {"source": "MedQuAD", "score": 0.85, "id": "doc_1"},
    },
    {
        "content": "Common symptoms of diabetes include increased thirst, frequent urination, "
        "blurred vision, and fatigue.",
        "source": "PubMedQA",
        "score": 0.78,
        "metadata": {"source": "PubMedQA", "score": 0.78, "id": "doc_2"},
    },
    {
        "content": "Treatment for type 2 diabetes includes lifestyle changes, metformin, "
        "and insulin therapy when needed.",
        "source": "MedMCQA",
        "score": 0.72,
        "metadata": {"source": "MedMCQA", "score": 0.72, "id": "doc_3"},
    },
]


@pytest.fixture
def mock_retriever():
    retriever = MockRetriever(documents=SAMPLE_DOCS)
    # MockRetriever.retrieve() needs to accept **kwargs for LangChain wrapper
    original_retrieve = retriever.retrieve

    def patched_retrieve(query, k=5, **kwargs):
        return original_retrieve(query, k)

    retriever.retrieve = patched_retrieve
    return retriever


@pytest.fixture
def mock_llm():
    return MockLLM(
        responses={
            "diabetes": "Diabetes is a chronic condition that affects how the body processes blood sugar. "
            "Common symptoms include increased thirst and frequent urination. "
            "Treatment typically includes lifestyle changes and medication.",
        }
    )


# =========================================================================
# LangChain Pipeline Tests
# =========================================================================
class TestLangChainPipeline:
    """Tests for the LangChain LCEL pipeline."""

    def test_langchain_pipeline_init(self, mock_retriever, mock_llm):
        """Test that LangChain pipeline initializes without errors."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        pipeline = LangChainHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm, k=3)
        assert pipeline is not None
        assert pipeline.k == 3
        assert pipeline.enable_grounding_gate is True

    def test_langchain_pipeline_invoke(self, mock_retriever, mock_llm):
        """Test basic invoke returns valid result."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        pipeline = LangChainHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm, k=3)
        result = pipeline.invoke("What are the symptoms of diabetes?")
        assert result is not None
        # Result can be a dict or dataclass — just check it has answer
        answer = (
            result.get("answer", "") if isinstance(result, dict) else getattr(result, "answer", "")
        )
        assert answer

    def test_langchain_pipeline_answer(self, mock_retriever, mock_llm):
        """Test the answer() method returns a proper result dict."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        pipeline = LangChainHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm, k=3)
        result = pipeline.answer("What are the symptoms of diabetes?")
        assert result is not None

    def test_langchain_adaptive_threshold(self, mock_retriever, mock_llm):
        """Test that adaptive threshold handles RRF-scale scores."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        pipeline = LangChainHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm, k=3)
        # Verify adaptive threshold params exist
        assert hasattr(pipeline, "adaptive_threshold_ratio")
        assert hasattr(pipeline, "absolute_score_floor")
        assert pipeline.absolute_score_floor == 0.01

    def test_langchain_grounding_gate_with_low_rrf_scores(self, mock_llm):
        """Test grounding gate passes with RRF-scale scores (~0.016)."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        # Create retriever with RRF-scale scores
        low_score_docs = [
            {**doc, "score": 0.016, "metadata": {**doc["metadata"], "score": 0.016}}
            for doc in SAMPLE_DOCS
        ]
        retriever = MockRetriever(documents=low_score_docs)
        original_retrieve = retriever.retrieve

        def patched_retrieve(query, k=5, **kwargs):
            return original_retrieve(query, k)

        retriever.retrieve = patched_retrieve

        pipeline = LangChainHealthcareQAPipeline(retriever=retriever, llm=mock_llm, k=3)

        result = pipeline.invoke("What is diabetes?")
        # Check is_answerable via dict or attribute access
        is_ans = (
            result.get("is_answerable", True)
            if isinstance(result, dict)
            else getattr(result, "is_answerable", True)
        )
        assert is_ans is True

    def test_langchain_unanswerable(self, mock_llm):
        """Test unanswerable response when no documents found."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        empty_retriever = MockRetriever(documents=[])
        original_retrieve = empty_retriever.retrieve

        def patched_retrieve(query, k=5, **kwargs):
            return original_retrieve(query, k)

        empty_retriever.retrieve = patched_retrieve

        pipeline = LangChainHealthcareQAPipeline(retriever=empty_retriever, llm=mock_llm, k=3)
        result = pipeline.invoke("Something with no documents?")
        is_ans = (
            result.get("is_answerable")
            if isinstance(result, dict)
            else getattr(result, "is_answerable", None)
        )
        assert is_ans is False


# =========================================================================
# LangGraph Pipeline Tests
# =========================================================================
class TestLangGraphPipeline:
    """Tests for the LangGraph StateGraph pipeline."""

    def test_langgraph_pipeline_init(self, mock_retriever, mock_llm):
        """Test that LangGraph pipeline initializes and compiles graph."""
        from src.langgraph.langgraph_pipeline import LangGraphHealthcareQAPipeline

        pipeline = LangGraphHealthcareQAPipeline(
            retriever=mock_retriever, llm=mock_llm, k=3, enable_checkpointing=False
        )
        assert pipeline is not None
        assert pipeline._graph is not None

    def test_langgraph_pipeline_invoke(self, mock_retriever, mock_llm):
        """Test basic invoke returns valid state dict."""
        from src.langgraph.langgraph_pipeline import LangGraphHealthcareQAPipeline

        pipeline = LangGraphHealthcareQAPipeline(
            retriever=mock_retriever, llm=mock_llm, k=3, enable_checkpointing=False
        )
        result = pipeline.invoke("What are the symptoms of diabetes?")
        assert result is not None
        # LangGraph returns LangGraphQAResult dataclass, not a dict
        assert hasattr(result, "answer") or (isinstance(result, dict) and "answer" in result)

    def test_langgraph_to_qa_response(self, mock_retriever, mock_llm):
        """Test QAResponse conversion for API compatibility."""
        from src.langgraph.langgraph_pipeline import LangGraphHealthcareQAPipeline

        pipeline = LangGraphHealthcareQAPipeline(
            retriever=mock_retriever, llm=mock_llm, k=3, enable_checkpointing=False
        )
        result = pipeline.invoke("What is diabetes?")
        qa_response = pipeline.to_qa_response(result)
        assert qa_response is not None
        assert hasattr(qa_response, "answer")
        assert hasattr(qa_response, "sources")

    def test_langgraph_unanswerable_route(self, mock_llm):
        """Test that empty retrieval routes to unanswerable."""
        from src.langgraph.langgraph_pipeline import LangGraphHealthcareQAPipeline

        empty_retriever = MockRetriever(documents=[])
        pipeline = LangGraphHealthcareQAPipeline(
            retriever=empty_retriever, llm=mock_llm, k=3, enable_checkpointing=False
        )
        result = pipeline.invoke("Unknown question with no docs")
        # LangGraph returns LangGraphQAResult dataclass
        is_ans = (
            result.get("is_answerable")
            if isinstance(result, dict)
            else getattr(result, "is_answerable", None)
        )
        assert is_ans is False


# =========================================================================
# Routing Logic Tests
# =========================================================================
class TestLangGraphRouting:
    """Tests for LangGraph routing functions."""

    def test_route_after_grading_generate(self):
        """Test routing to generate when enough relevant docs."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": True, "retry_count": 0, "documents": [1, 2, 3]}
        assert route_after_grading(state) == "generate"

    def test_route_after_grading_refine(self):
        """Test routing to refine when not enough docs but can retry."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": False, "retry_count": 0, "documents": [1]}
        assert route_after_grading(state) == "refine"

    def test_route_after_grading_unanswerable(self):
        """Test routing to unanswerable when no docs and max retries."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": False, "retry_count": 5, "documents": []}
        assert route_after_grading(state) == "unanswerable"

    def test_route_after_verify_grounded(self):
        """Test routing after verification when grounded."""
        from src.langgraph.langgraph_routing import route_after_verify

        state = {"is_grounded": True, "retry_count": 0}
        assert route_after_verify(state) == "enrich_xai"

    def test_route_after_verify_not_grounded_retry(self):
        """Test routing to refine when not grounded and can retry."""
        from src.langgraph.langgraph_routing import route_after_verify

        state = {"is_grounded": False, "retry_count": 0}
        assert route_after_verify(state) == "refine"

    def test_route_after_verify_exhausted_retries(self):
        """Test routing to enrich_xai when retries exhausted."""
        from src.langgraph.langgraph_routing import route_after_verify

        state = {"is_grounded": False, "retry_count": 5}
        assert route_after_verify(state) == "enrich_xai"


# =========================================================================
# Text Cleaning Tests
# =========================================================================
class TestTextCleaning:
    """Tests for the shared text cleaning utility."""

    def test_clean_response_removes_prefixes(self):
        from src.utils.text_cleaning import clean_llm_response

        result = clean_llm_response("Answer: The patient should rest.")
        assert result == "The patient should rest."

    def test_clean_response_truncates_at_stop_pattern(self):
        from src.utils.text_cleaning import clean_llm_response

        text = "Diabetes is a chronic condition. It affects blood sugar management. Best regards, Doctor"
        result = clean_llm_response(text)
        assert "Best regards" not in result

    def test_clean_response_handles_empty(self):
        from src.utils.text_cleaning import clean_llm_response

        assert clean_llm_response("") == ""
        assert clean_llm_response(None) is None

    def test_clean_response_collapses_whitespace(self):
        from src.utils.text_cleaning import clean_llm_response

        result = clean_llm_response("Multiple   spaces   here.  And more.")
        assert "  " not in result


# =========================================================================
# Hallucination Detector Tests
# =========================================================================
class TestHallucinationDetector:
    """Tests for the completed HallucinationDetector."""

    def test_detect_empty_answer(self):
        from src.xai.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        result = detector.detect("", [])
        assert result.has_hallucination is False
        assert result.hallucination_score == 0.0

    def test_detect_grounded_answer(self):
        from src.xai.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        docs = [
            {"content": "Diabetes is a chronic metabolic disease affecting blood sugar levels."}
        ]
        answer = (
            "Diabetes is a chronic metabolic disease. It affects blood sugar levels in the body."
        )
        result = detector.detect(answer, docs)
        # Well-grounded answer should have low hallucination score
        assert result.hallucination_score < 0.8

    def test_detect_medical_sanity_violation(self):
        from src.xai.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        answer = "Normal body temperature is 120.5 °F which is perfectly healthy."
        result = detector.detect(answer, [{"content": "body temperature"}])
        assert len(result.medical_accuracy_flags) > 0

    def test_detect_fabricated_citation(self):
        from src.xai.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector()
        answer = "According to a recent study (Smith et al., 2024), diabetes affects millions."
        result = detector.detect(answer, [{"content": "diabetes facts"}])
        fabricated = [c for c in result.flagged_claims if c.get("check") == "fabricated_citation"]
        assert len(fabricated) > 0
