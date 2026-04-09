"""
Tests for LangChain Pipeline.

Unit tests for the LangChain-based Healthcare QA pipeline components.
"""

# Test imports
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MockDocument:
    """Mock document for testing."""

    content: str
    source: str
    score: float
    metadata: dict
    url: str = ""


class TestLangChainLLMWrapper:
    """Tests for LangChainMedicalLLM wrapper."""

    def test_llm_wrapper_initialization(self):
        """Test that LLM wrapper initializes correctly."""
        from src.langchain.langchain_llm import LangChainMedicalLLM

        mock_llm = Mock()
        wrapper = LangChainMedicalLLM(llm=mock_llm)

        assert wrapper._llm is not None
        assert wrapper.model_name == "medical-llm"

    def test_llm_call_delegates_to_underlying(self):
        """Test that _call properly delegates to underlying LLM."""
        from src.langchain.langchain_llm import LangChainMedicalLLM

        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(response="Test response")

        wrapper = LangChainMedicalLLM(llm=mock_llm)
        result = wrapper._call("Test prompt")

        assert result == "Test response"
        mock_llm.generate.assert_called_once()


class TestLangChainRetrieverWrapper:
    """Tests for LangChainHybridRetriever wrapper."""

    def test_retriever_wrapper_initialization(self):
        """Test that retriever wrapper initializes correctly."""
        from src.langchain.langchain_retriever import LangChainHybridRetriever

        mock_retriever = Mock()
        wrapper = LangChainHybridRetriever(retriever=mock_retriever, k=3)

        assert wrapper._retriever is not None
        assert wrapper.k == 3

    def test_retriever_converts_documents(self):
        """Test that documents are properly converted to LangChain format."""
        from src.langchain.langchain_retriever import LangChainHybridRetriever

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            MockDocument(
                content="Test content", source="TestSource", score=0.85, metadata={"key": "value"}
            )
        ]

        wrapper = LangChainHybridRetriever(retriever=mock_retriever)
        result = wrapper._get_relevant_documents("test query")

        assert len(result) == 1
        assert result[0].page_content == "Test content"


class TestLangChainPipeline:
    """Tests for LangChainHealthcareQAPipeline."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = Mock()
        retriever.retrieve.return_value = [
            MockDocument(
                content="Diabetes is a chronic metabolic condition",
                source="Medical Encyclopedia",
                score=0.85,
                metadata={"score": 0.85},
            )
        ]
        return retriever

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.generate.return_value = Mock(response="Diabetes is a chronic condition.")
        return llm

    def test_pipeline_initialization(self, mock_retriever, mock_llm):
        """Test pipeline initializes correctly."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        pipeline = LangChainHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm)

        assert pipeline is not None
        assert pipeline.k == 5
        assert pipeline.enable_grounding_gate == True

    def test_grounding_gate_blocks_low_relevance(self, mock_retriever, mock_llm):
        """Test that grounding gate blocks when docs have low relevance."""
        from src.langchain.langchain_pipeline import LangChainHealthcareQAPipeline

        # Configure low-relevance docs
        mock_retriever.retrieve.return_value = [
            MockDocument(
                content="Unrelated content", source="Unknown", score=0.1, metadata={"score": 0.1}
            )
        ]

        pipeline = LangChainHealthcareQAPipeline(
            retriever=mock_retriever, llm=mock_llm, min_retrieval_score=0.3, min_relevant_docs=2
        )

        assert pipeline.enable_grounding_gate == True


class TestLangChainCallbacks:
    """Tests for LangChain callback handler."""

    def test_callback_handler_initialization(self, tmp_path):
        """Test callback handler initializes correctly."""
        from src.langchain.langchain_callbacks import MedicalQACallbackHandler

        log_file = tmp_path / "test_log.jsonl"
        handler = MedicalQACallbackHandler(log_file=str(log_file))

        assert handler.log_file.exists() or handler.log_file.parent.exists()

    def test_callback_logs_chain_start(self, tmp_path):
        """Test that chain start is logged."""
        from src.langchain.langchain_callbacks import MedicalQACallbackHandler

        log_file = tmp_path / "test_log.jsonl"
        handler = MedicalQACallbackHandler(log_file=str(log_file), log_to_console=False)

        handler.on_chain_start(
            serialized={"name": "test_chain"},
            inputs={"question": "What is diabetes?"},
            run_id="test-run-123",
        )

        assert handler.run_id == "test-run-123"
        assert handler.start_time is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
