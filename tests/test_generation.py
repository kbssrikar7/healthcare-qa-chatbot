"""
Unit tests for generation components.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.prompt_manager import MedicalPromptManager


class TestMedicalPromptManager:
    """Tests for MedicalPromptManager."""

    @pytest.fixture
    def prompt_manager(self):
        return MedicalPromptManager()

    def test_build_prompt_default(self, prompt_manager):
        """Test building prompt with default template."""
        prompt = prompt_manager.build_prompt(
            question="What is diabetes?", context="Diabetes is a metabolic disorder."
        )

        assert "What is diabetes?" in prompt
        assert "Diabetes is a metabolic disorder." in prompt

    def test_build_prompt_with_template(self, prompt_manager):
        """Test building prompt with specific template."""
        prompt = prompt_manager.build_prompt(
            question="Test question", context="Test context", template_name="simple"
        )

        assert "Test question" in prompt
        assert "Test context" in prompt

    def test_build_context_from_documents(self, prompt_manager):
        """Test building context from documents."""
        documents = [
            {"content": "First document content", "source": "Source1"},
            {"content": "Second document content", "source": "Source2"},
        ]

        context = prompt_manager.build_context_from_documents(documents)

        assert "First document content" in context
        assert "Source1" in context

    def test_context_respects_max_length(self, prompt_manager):
        """Test that context respects max length."""
        documents = [
            {"content": "A" * 500, "source": "Source1"},
            {"content": "B" * 500, "source": "Source2"},
        ]

        context = prompt_manager.build_context_from_documents(documents, max_length=600)

        assert len(context) <= 700  # Allow some buffer for source labels

    def test_get_medical_disclaimer(self, prompt_manager):
        """Test medical disclaimer is returned."""
        disclaimer = prompt_manager.get_medical_disclaimer()

        assert "DISCLAIMER" in disclaimer or "disclaimer" in disclaimer.lower()
        assert "NOT a substitute" in disclaimer or "substitute" in disclaimer.lower()

    def test_add_custom_template(self, prompt_manager):
        """Test adding custom template."""
        prompt_manager.add_template(
            name="custom", template="Q: {question}\nC: {context}\nA:", description="Custom template"
        )

        prompt = prompt_manager.build_prompt(
            question="Test Q", context="Test C", template_name="custom"
        )

        assert "Q: Test Q" in prompt
        assert "C: Test C" in prompt


class TestLLMWrapper:
    """Tests for LLM wrapper (import tests only, no model loading)."""

    def test_llm_imports(self):
        """Test LLM wrapper can be imported."""
        from src.generation.llm_wrapper import GenerationResult, MedicalLLM

        assert MedicalLLM is not None
        assert GenerationResult is not None

    def test_generation_result_dataclass(self):
        """Test GenerationResult dataclass."""
        from src.generation.llm_wrapper import GenerationResult

        result = GenerationResult(
            response="Test response",
            input_tokens=10,
            generated_tokens=20,
            probabilities=[0.9, 0.8, 0.7],
        )

        assert result.response == "Test response"
        assert result.input_tokens == 10
        assert result.generated_tokens == 20
        assert len(result.probabilities) == 3

    def test_supported_models_exist(self):
        """Test that supported models are defined."""
        from src.generation.llm_wrapper import MedicalLLM

        assert "tinyllama" in MedicalLLM.SUPPORTED_MODELS
        assert "biomistral" in MedicalLLM.SUPPORTED_MODELS
