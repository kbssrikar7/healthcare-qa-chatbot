"""
Pytest fixtures for Healthcare QA Chatbot tests.

Includes mock components for testing without requiring full model loading.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Mock Components
# =============================================================================


class MockEmbedder:
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache = {}

    def embed_query(self, text: str) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        if text not in self._cache:
            # Use hash for deterministic but varied embeddings
            np.random.seed(hash(text) % 2**32)
            self._cache[text] = np.random.randn(self.dimension).astype(np.float32)
            self._cache[text] /= np.linalg.norm(self._cache[text])
        return self._cache[text]

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed_query(t) for t in texts])


@dataclass
class MockDocument:
    """Mock document for testing retrieval."""

    content: str
    source: str
    score: float
    metadata: Dict


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []

    def retrieve(self, query: str, k: int = 5) -> List[MockDocument]:
        """Return mock documents."""
        return [
            MockDocument(
                content=doc.get("content", ""),
                source=doc.get("source", "Unknown"),
                score=doc.get("score", 0.8),
                metadata=doc.get("metadata", {}),
            )
            for doc in self.documents[:k]
        ]

    def retrieve_with_context(self, query: str, k: int = 5):
        """Return documents and concatenated context."""
        docs = self.retrieve(query, k)
        context = "\n\n".join(doc.content for doc in docs)
        return docs, context


@dataclass
class MockGenerationResult:
    """Mock LLM generation result."""

    response: str
    input_tokens: int = 100
    generated_tokens: int = 50
    probabilities: Optional[List[float]] = None


class MockLLM:
    """Mock LLM for testing without GPU."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.default_response = "This is a test response about medical conditions."

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        return_probabilities: bool = False,
        **kwargs,
    ) -> MockGenerationResult:
        """Return mock generation result."""
        # Check for keyword matches in responses
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return MockGenerationResult(
                    response=response,
                    probabilities=[0.9] * 10 if return_probabilities else None,
                )

        return MockGenerationResult(
            response=self.default_response,
            probabilities=[0.85] * 10 if return_probabilities else None,
        )


class MockPromptManager:
    """Mock prompt manager for testing."""

    def build_prompt(self, question: str, context: str, template_name: str = "medical_qa") -> str:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    def get_medical_disclaimer(self) -> str:
        return "This is for educational purposes only. Consult a healthcare professional."


class MockConfidenceScorer:
    """Mock confidence scorer for testing."""

    @dataclass
    class Result:
        calibrated_score: float = 0.75
        level: str = "medium"
        explanation: str = "Mock confidence score"

    def calculate_confidence(self, **kwargs):
        return self.Result()


class MockSourceAttributor:
    """Mock source attributor for testing."""

    @dataclass
    class Attribution:
        claim: str
        source: str
        evidence: str
        similarity_score: float

    def attribute_answer(self, answer: str, documents: List[Dict]):
        """Return mock attributions."""
        return [
            self.Attribution(
                claim="Test claim",
                source="Test Source",
                evidence="Test evidence",
                similarity_score=0.85,
            )
        ]


# =============================================================================
# Fixtures - Mock Components
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Provide mock embedder."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def mock_retriever(sample_documents):
    """Provide mock retriever with sample documents."""
    docs = [
        {"content": d["content"], "source": d["source"], "score": 0.85, "metadata": {}}
        for d in sample_documents
    ]
    return MockRetriever(documents=docs)


@pytest.fixture
def mock_llm():
    """Provide mock LLM with medical responses."""
    return MockLLM(
        responses={
            "diabetes": "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "hypertension": "Hypertension is treated with lifestyle changes and medications like ACE inhibitors.",
            "headache": "Common headache causes include tension, dehydration, and lack of sleep.",
        }
    )


@pytest.fixture
def mock_prompt_manager():
    """Provide mock prompt manager."""
    return MockPromptManager()


@pytest.fixture
def mock_confidence_scorer():
    """Provide mock confidence scorer."""
    return MockConfidenceScorer()


@pytest.fixture
def mock_source_attributor():
    """Provide mock source attributor."""
    return MockSourceAttributor()


@pytest.fixture
def mock_pipeline(mock_retriever, mock_llm, mock_prompt_manager):
    """Provide mock pipeline for testing."""
    from src.pipeline.qa_pipeline import HealthcareQAPipeline

    return HealthcareQAPipeline(
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_manager=mock_prompt_manager,
        enable_grounding_gate=False,  # Disable for predictable testing
    )


# =============================================================================
# Fixtures - Sample Data
# =============================================================================


@pytest.fixture
def sample_question():
    return "What are the symptoms of diabetes?"


@pytest.fixture
def sample_documents():
    return [
        {
            "content": "Diabetes symptoms include increased thirst, frequent urination, and fatigue.",
            "source": "CDC",
        },
        {
            "content": "Type 2 diabetes often develops slowly with symptoms that are easy to miss.",
            "source": "Mayo Clinic",
        },
        {
            "content": "Common signs of diabetes include blurred vision and slow wound healing.",
            "source": "WebMD",
        },
    ]


@pytest.fixture
def sample_qa_pairs():
    return [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision.",
            "source": "CDC",
        },
        {
            "question": "How is high blood pressure treated?",
            "answer": "High blood pressure is treated through lifestyle changes (diet, exercise) and medications like ACE inhibitors, beta-blockers, or diuretics.",
            "source": "American Heart Association",
        },
    ]


@pytest.fixture
def edge_case_questions():
    """Edge case questions for robustness testing."""
    return [
        "",  # Empty question
        "a",  # Too short
        "What is " + "very " * 100 + "important?",  # Very long
        "症状は何ですか",  # Non-English
        "What are the s1de eff3cts?",  # Typos/leetspeak
        "WHAT ARE THE SYMPTOMS OF DIABETES?!?!",  # Shouting
    ]


@pytest.fixture
def emergency_inputs():
    """Inputs that should trigger emergency detection."""
    return [
        "I want to kill myself",
        "I'm having a heart attack",
        "I can't breathe and my chest hurts",
        "I took an overdose of pills",
    ]


@pytest.fixture
def safe_inputs():
    """Clearly safe inputs for testing."""
    return [
        "What causes headaches?",
        "How much water should I drink daily?",
        "What vitamins are good for energy?",
    ]
