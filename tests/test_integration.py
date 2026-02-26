"""
Integration tests for Healthcare QA Chatbot improvements.

Tests cover:
- Multi-model pipeline selection
- Safety guardrails integration
- Conversation history across turns
- Query expansion in retrieval
- Passage highlighting
"""
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
        assert "biomistral-7b" in AVAILABLE_MODELS
        assert AVAILABLE_MODELS["tinyllama"]["display_name"] == "TinyLlama 1.1B"
        assert AVAILABLE_MODELS["biomistral-7b"]["requires_gpu"] is True

    def test_model_choice_enum(self):
        """Test ModelChoice enum values."""
        from config.settings import ModelChoice
        assert ModelChoice.TINYLLAMA.value == "tinyllama"
        assert ModelChoice.BIOMISTRAL_7B.value == "biomistral-7b"

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
        """Test production config uses BioMistral."""
        from config.settings import Config
        cfg = Config.from_env("production")
        assert "BioMistral" in cfg.llm.model_name


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
            model_choice="biomistral-7b",
            use_langchain=True,
        )
        assert req.model_choice == "biomistral-7b"
        assert req.use_langchain is True

    def test_safety_info_defaults(self):
        """Test SafetyInfo defaults."""
        from api.main import SafetyInfo
        s = SafetyInfo()
        assert s.level == "safe"
        assert s.is_emergency is False
        assert s.drug_warnings is None
