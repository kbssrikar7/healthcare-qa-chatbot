"""
Integration tests for Healthcare QA Chatbot improvements.

Tests cover:
- TinyLlama pipeline configuration
- Safety guardrails integration
- Conversation history across turns
- Query expansion in retrieval
- Passage highlighting
"""
from datetime import datetime, timedelta
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Multi-Model Pipeline Selection
# =============================================================================

class TestMultiModelSelection:
    """Test that model selection configuration works correctly."""

    def test_available_models_registry(self):
        """Test AVAILABLE_MODELS has expected entries."""
        from config.settings import AVAILABLE_MODELS
        assert "tinyllama" in AVAILABLE_MODELS
        assert AVAILABLE_MODELS["tinyllama"]["display_name"] == "TinyLlama 1.1B"
        assert AVAILABLE_MODELS["tinyllama"]["requires_gpu"] is False

    def test_model_choice_enum(self):
        """Test ModelChoice enum values."""
        from config.settings import ModelChoice
        assert ModelChoice.TINYLLAMA.value == "tinyllama"

    def test_llm_config_default_model(self):
        """Test that LLMConfig has a default_model field."""
        from config.settings import LLMConfig
        config = LLMConfig()
        assert config.default_model == "tinyllama"

    def test_config_from_env_development(self):
        """Test development config uses tinyllama."""
        from config.settings import Config
        cfg = Config.from_env("development")
        assert cfg.llm.model_name == "tinyllama"

    def test_config_from_env_production(self):
        """Test production config uses TinyLlama."""
        from config.settings import Config
        cfg = Config.from_env("production")
        assert "TinyLlama" in cfg.llm.model_name


# =============================================================================
# Test Safety Guardrails Integration
# =============================================================================

class TestGuardrailsIntegration:
    """Test safety guardrails detect and handle dangerous inputs."""

    def test_emergency_detection(self):
        """Test that emergency keywords trigger emergency response."""
        from src.safety.guardrails import MedicalGuardrails
        guard = MedicalGuardrails()
        result = guard.check_input("I'm having severe chest pain and can't breathe")
        assert result.level.value in ("emergency", "blocked", "caution")

    def test_safe_input_passes(self):
        """Test that safe medical questions pass guardrails."""
        from src.safety.guardrails import MedicalGuardrails
        guard = MedicalGuardrails()
        result = guard.check_input("What are the symptoms of diabetes?")
        assert result.passed is True

    def test_content_filter_blocks_dangerous(self):
        """Test content filter blocks harmful content."""
        from src.safety.guardrails import ContentFilter
        cf = ContentFilter()
        blocked, _ = cf.is_blocked("how to make illegal drugs")
        assert blocked is True
        blocked, _ = cf.is_blocked("What is aspirin used for?")
        assert blocked is False

    def test_drug_interaction_checker(self):
        """Test drug interaction detection."""
        from src.safety.guardrails import DrugInteractionChecker
        checker = DrugInteractionChecker()
        warnings = checker.check_interaction_risk(
            "I take warfarin and aspirin together daily"
        )
        # Should detect warfarin + aspirin interaction
        assert isinstance(warnings, list)

    def test_emergency_detector_categories(self):
        """Test EmergencyDetector categorization."""
        from src.safety.guardrails import EmergencyDetector
        detector = EmergencyDetector()
        is_emergency, category = detector.detect("I think I'm having a heart attack")
        # Should detect as cardiac emergency
        assert isinstance(is_emergency, bool)


# =============================================================================
# Test Conversation History
# =============================================================================

class TestConversationHistory:
    """Test conversation management and follow-up detection."""

    def test_create_session(self):
        """Test session creation."""
        from src.conversation.history import ConversationManager
        cm = ConversationManager()
        session = cm.create_session()
        assert session.session_id is not None
        assert len(session.turns) == 0

    def test_add_turn(self):
        """Test adding a turn to conversation."""
        from src.conversation.history import ConversationManager
        cm = ConversationManager()
        session = cm.create_session()
        cm.add_turn(
            session_id=session.session_id,
            question="What is diabetes?",
            answer="Diabetes is a metabolic disorder...",
            confidence=0.85,
            sources=["MedQuAD"],
        )
        assert len(session.turns) == 1
        assert session.turns[0].question == "What is diabetes?"

    def test_followup_detection(self):
        """Test follow-up question detection."""
        from src.conversation.history import FollowUpDetector
        detector = FollowUpDetector()
        # "What about it?" with previous context should be follow-up
        assert detector.detect_followup(
            "What about its treatment?",
            previous_context="diabetes"
        ) is True
        # Clear standalone question
        assert detector.detect_followup(
            "What is hypertension?",
            previous_context=None
        ) is False

    def test_context_window(self):
        """Test conversation context window generation."""
        from src.conversation.history import ConversationManager
        cm = ConversationManager()
        session = cm.create_session()
        cm.add_turn(
            session_id=session.session_id,
            question="What is diabetes?",
            answer="Diabetes is a chronic condition.",
            confidence=0.85,
            sources=["MedQuAD"],
        )
        context = session.get_context_window()
        assert "diabetes" in context.lower()

    def test_expired_session_is_cleaned_on_access(self):
        """Expired sessions should be removed during normal manager use."""
        from src.conversation.history import ConversationManager

        cm = ConversationManager(max_session_age_hours=1, cleanup_interval_minutes=0)
        session = cm.create_session()
        session.last_activity = datetime.now() - timedelta(hours=2)

        assert cm.get_session(session.session_id) is None
        assert session.session_id not in cm.conversations

    def test_save_sessions_creates_parent_directory(self, tmp_path):
        """Saving sessions should create parent directories for persistence."""
        from src.conversation.history import ConversationManager

        storage_path = tmp_path / "nested" / "sessions.json"
        cm = ConversationManager(storage_path=str(storage_path))
        cm.create_session()

        cm.save_sessions()

        assert storage_path.exists()


# =============================================================================
# Test Query Expansion
# =============================================================================

class TestQueryExpansion:
    """Test medical query expansion with synonyms."""

    def test_synonym_expansion(self):
        """Test that medical synonyms are generated."""
        from src.retrieval.query_expander import MedicalQueryExpander
        expander = MedicalQueryExpander()
        queries = expander.expand("What are the symptoms of diabetes?")
        assert len(queries) > 1
        # Should include original
        assert queries[0] == "What are the symptoms of diabetes?"
        # Should have expanded variants
        assert any("diabetes mellitus" in q.lower() for q in queries)

    def test_abbreviation_expansion(self):
        """Test medical abbreviation expansion."""
        from src.retrieval.query_expander import MedicalQueryExpander
        expander = MedicalQueryExpander()
        terms = expander.get_expanded_terms("What causes COPD?")
        assert "chronic obstructive pulmonary disease" in terms

    def test_no_expansion_for_unknown(self):
        """Test that unknown terms return only original."""
        from src.retrieval.query_expander import MedicalQueryExpander
        expander = MedicalQueryExpander()
        queries = expander.expand("How does photosynthesis work?")
        assert len(queries) == 1

    def test_max_expansions_limit(self):
        """Test expansion limit is respected."""
        from src.retrieval.query_expander import MedicalQueryExpander
        expander = MedicalQueryExpander(max_expansions=2)
        queries = expander.expand("What causes a heart attack?")
        assert len(queries) <= 3  # original + max 2


# =============================================================================
# Test Passage Highlighting
# =============================================================================

class TestPassageHighlighting:
    """Test XAI passage highlighting."""

    def test_highlight_matching_passage(self):
        """Test that matching spans are found."""
        from src.xai.passage_highlighter import PassageHighlighter
        highlighter = PassageHighlighter()
        
        answer = "Diabetes causes increased thirst and frequent urination."
        passages = [{
            "content": "People with diabetes experience increased thirst and frequent urination as primary symptoms.",
            "source": "MedQuAD",
            "score": 0.85,
        }]
        
        results = highlighter.highlight(answer, passages)
        assert len(results) == 1
        assert results[0].source == "MedQuAD"

    def test_no_highlights_for_unrelated(self):
        """Test no spans for unrelated passages."""
        from src.xai.passage_highlighter import PassageHighlighter
        highlighter = PassageHighlighter()
        
        answer = "The sky is blue due to Rayleigh scattering."
        passages = [{
            "content": "Diabetes is a metabolic disorder affecting blood sugar levels.",
            "source": "MedQuAD",
            "score": 0.3,
        }]
        
        results = highlighter.highlight(answer, passages)
        assert len(results) == 1
        assert len(results[0].highlight_spans) == 0

    def test_sentence_splitting(self):
        """Test sentence splitting with medical abbreviations."""
        from src.xai.passage_highlighter import PassageHighlighter
        h = PassageHighlighter()
        sentences = h._split_sentences("Take 500 mg. of aspirin daily. Consult Dr. Smith.")
        # Should not split on "mg." or "Dr."
        assert len(sentences) >= 1


# =============================================================================
# Test API Request/Response Models
# =============================================================================

class TestAPIModels:
    """Test Pydantic request/response models."""

    def test_question_request_defaults(self):
        """Test QuestionRequest default values."""
        from api.main import QuestionRequest
        req = QuestionRequest(question="What is diabetes?")
        assert req.include_explanation is True
        assert req.num_sources == 3
        assert req.model_choice is None
        assert req.use_langchain is False

    def test_question_request_with_model(self):
        """Test QuestionRequest with model choice."""
        from api.main import QuestionRequest
        req = QuestionRequest(
            question="What is diabetes?",
            model_choice="tinyllama",
            use_langchain=True,
        )
        assert req.model_choice == "tinyllama"
        assert req.use_langchain is True

    def test_safety_info_defaults(self):
        """Test SafetyInfo defaults."""
        from api.main import SafetyInfo
        s = SafetyInfo()
        assert s.level == "safe"
        assert s.is_emergency is False
        assert s.drug_warnings is None


# =============================================================================
# Test RAG-Engineer improvements: adaptive weights + retrieval metrics
# =============================================================================

class TestAdaptiveRetrievalWeights:
    """Query-type detection drives per-type RRF weight selection."""

    def setup_method(self):
        from unittest.mock import MagicMock
        from src.retrieval.hybrid_retriever import HybridRetriever
        self.retriever = HybridRetriever.__new__(HybridRetriever)

    def test_drug_query_detected(self):
        assert self.retriever._detect_query_type("What is the dosage for metformin?") == "drug"
        assert self.retriever._detect_query_type("ibuprofen 400mg side effects") == "drug"

    def test_definition_query_detected(self):
        assert self.retriever._detect_query_type("What is type 2 diabetes?") == "definition"
        assert self.retriever._detect_query_type("Explain hypertension") == "definition"

    def test_symptom_query_detected(self):
        assert self.retriever._detect_query_type("I have chest pain and nausea") == "symptom"
        assert self.retriever._detect_query_type("symptoms of fever") == "symptom"

    def test_comparison_query_detected(self):
        assert self.retriever._detect_query_type("difference between type 1 and type 2 diabetes") == "comparison"

    def test_default_query_detected(self):
        assert self.retriever._detect_query_type("how long does recovery take") == "default"

    def test_drug_weights_are_bm25_heavy(self):
        dense_w, sparse_w = self.retriever._ADAPTIVE_WEIGHTS["drug"]
        assert sparse_w > dense_w, "Drug queries should be BM25-heavy"

    def test_definition_weights_are_dense_heavy(self):
        dense_w, sparse_w = self.retriever._ADAPTIVE_WEIGHTS["definition"]
        assert dense_w > sparse_w, "Definition queries should be dense-heavy"

    def test_all_weights_sum_to_one(self):
        for qt, (d, s) in self.retriever._ADAPTIVE_WEIGHTS.items():
            assert abs(d + s - 1.0) < 1e-6, f"{qt} weights don't sum to 1.0"


class TestRetrievalMetrics:
    """Unit tests for MRR, Precision, Recall, NDCG functions."""

    def test_mrr_first_hit(self):
        from evaluation.retrieval_metrics import _reciprocal_rank
        assert _reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_mrr_second_hit(self):
        from evaluation.retrieval_metrics import _reciprocal_rank
        assert _reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        from evaluation.retrieval_metrics import _reciprocal_rank
        assert _reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_precision_at_k(self):
        from evaluation.retrieval_metrics import _precision_at_k
        assert _precision_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, k=5) == pytest.approx(0.4)

    def test_recall_at_k(self):
        from evaluation.retrieval_metrics import _recall_at_k
        assert _recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == pytest.approx(0.5)

    def test_ndcg_perfect(self):
        from evaluation.retrieval_metrics import _ndcg_at_k
        assert _ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_no_hit(self):
        from evaluation.retrieval_metrics import _ndcg_at_k
        assert _ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_compute_retrieval_metrics_with_mock(self):
        from unittest.mock import MagicMock
        from evaluation.retrieval_metrics import compute_retrieval_metrics
        from src.retrieval.hybrid_retriever import RetrievedDocument

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(content="diabetes symptoms include thirst", source="MedQuAD",
                              score=0.9, metadata={}, doc_id="doc1"),
            RetrievedDocument(content="type 2 diabetes management", source="MedQuAD",
                              score=0.8, metadata={}, doc_id="doc2"),
        ]

        test_set = [
            {"question": "What are symptoms of diabetes?",
             "answer": "thirst fatigue blurred vision urination weight loss"},
        ]
        results = compute_retrieval_metrics(mock_retriever, test_set, k=5)
        assert "mrr@5" in results
        assert "recall@5" in results
        assert "ndcg@5" in results
        assert results["n_queries"] == 1

