"""
Tests for LangGraph Pipeline.

Unit tests for the LangGraph-based Healthcare RAG pipeline components.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MockDocument:
    """Mock document for testing."""

    page_content: str
    metadata: dict


class TestLangGraphState:
    """Tests for LangGraph state management."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        from src.langgraph.langgraph_state import create_initial_state

        state = create_initial_state("What is diabetes?")

        assert state["question"] == "What is diabetes?"
        assert state["query_history"] == ["What is diabetes?"]
        assert state["retry_count"] == 0
        assert state["documents"] == []

    def test_state_has_required_fields(self):
        """Test that state schema has all required fields."""
        from src.langgraph.langgraph_state import HealthcareRAGState

        # Check annotations exist
        annotations = HealthcareRAGState.__annotations__
        required_fields = [
            "question",
            "documents",
            "answer",
            "is_answerable",
            "confidence",
        ]

        for field in required_fields:
            assert field in annotations


class TestLangGraphRouting:
    """Tests for LangGraph conditional routing."""

    def test_route_after_grading_generate(self):
        """Test routing to generate when docs are relevant."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": True, "retry_count": 0, "documents": [Mock()]}

        result = route_after_grading(state)
        assert result == "generate"

    def test_route_after_grading_refine(self):
        """Test routing to refine when docs not relevant enough."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": False, "retry_count": 0, "documents": [Mock()]}

        result = route_after_grading(state)
        assert result == "refine"

    def test_route_after_grading_unanswerable(self):
        """Test routing to unanswerable after max retries."""
        from src.langgraph.langgraph_routing import route_after_grading

        state = {"is_answerable": False, "retry_count": 3, "documents": []}

        result = route_after_grading(state)
        assert result == "unanswerable"

    def test_route_after_verify_grounded(self):
        """Test routing when answer is grounded."""
        from src.langgraph.langgraph_routing import route_after_verify

        state = {"is_grounded": True}
        result = route_after_verify(state)
        assert result == "enrich_xai"

    def test_route_after_verify_not_grounded(self):
        """Test routing when answer is not grounded."""
        from src.langgraph.langgraph_routing import route_after_verify

        state = {"is_grounded": False, "retry_count": 0}
        result = route_after_verify(state)
        assert result == "refine"


class TestLangGraphNodes:
    """Tests for LangGraph node implementations."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = Mock()
        retriever.retrieve.return_value = [
            Mock(
                content="Diabetes is a metabolic disorder",
                source="Medical Reference",
                score=0.85,
                url="",
            )
        ]
        return retriever

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.generate.return_value = Mock(response="Diabetes affects blood sugar levels.")
        return llm

    def test_nodes_initialization(self, mock_retriever, mock_llm):
        """Test nodes class initializes correctly."""
        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        assert nodes is not None
        assert nodes.k == 5

    def test_retrieve_documents_node(self, mock_retriever, mock_llm):
        """Test document retrieval node."""
        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        state = {
            "question": "What is diabetes?",
            "query_history": ["What is diabetes?"],
        }
        result = nodes.retrieve_documents(state)

        assert "documents" in result
        assert len(result["documents"]) > 0

    def test_grade_relevance_node(self, mock_retriever, mock_llm):
        """Test document relevance grading node."""
        from langchain_core.documents import Document

        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        docs = [
            Document(
                page_content="Diabetes is a chronic metabolic condition",
                metadata={"score": 0.85, "source": "Medical"},
            )
        ]

        state = {"question": "What is diabetes?", "documents": docs}

        result = nodes.grade_relevance(state)

        assert "doc_grades" in result
        assert "is_answerable" in result

    def test_verify_grounding_node(self, mock_retriever, mock_llm):
        """Test grounding verification node."""
        from langchain_core.documents import Document

        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        state = {
            "answer": "Diabetes is a chronic metabolic condition that affects blood sugar.",
            "context": "Diabetes is a chronic metabolic condition characterized by elevated blood glucose levels.",
            "documents": [
                Document(
                    page_content="Diabetes is a chronic metabolic condition",
                    metadata={},
                )
            ],
        }

        result = nodes.verify_grounding(state)

        assert "is_grounded" in result
        assert "grounding_score" in result

    def test_unanswerable_response_node(self, mock_retriever, mock_llm):
        """Test unanswerable response node."""
        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        state = {"question": "What is xyz?"}
        result = nodes.unanswerable_response(state)

        assert result["is_answerable"] == False
        assert result["confidence"]["score"] == 0.0
        assert "I don't have enough" in result["answer"]

    def test_handle_error_node(self, mock_retriever, mock_llm):
        """Test error handling node."""
        from src.langgraph.langgraph_nodes import HealthcareRAGNodes

        nodes = HealthcareRAGNodes(retriever=mock_retriever, llm=mock_llm)

        state = {"question": "Test?", "error": "Test error message"}
        result = nodes.handle_error(state)

        assert result["is_answerable"] == False
        assert result["needs_review"] == True
        assert "error" in result["confidence"]["explanation"].lower()


class TestLangGraphPipeline:
    """Tests for the complete LangGraph pipeline."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = Mock()
        retriever.retrieve.return_value = [
            Mock(content="Test medical content", source="Test", score=0.8, url="")
        ]
        return retriever

    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.generate.return_value = Mock(response="Test answer")
        return llm

    def test_pipeline_builds_graph(self, mock_retriever, mock_llm):
        """Test that pipeline builds a valid graph."""
        from src.langgraph.langgraph_pipeline import LangGraphHealthcareQAPipeline

        pipeline = LangGraphHealthcareQAPipeline(retriever=mock_retriever, llm=mock_llm)

        assert pipeline._graph is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
