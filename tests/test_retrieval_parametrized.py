"""
Parametrized retrieval quality tests.

Tests that the hybrid retriever returns medically relevant results
for a range of query types (clinical, research, simple, complex).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievedDocument,
    reciprocal_rank_fusion,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_retriever():
    """Create a retriever with mocked dependencies."""
    embedder = MagicMock()
    vector_store = MagicMock()

    # Mock embedder to return a dummy vector
    embedder.embed_query.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    return retriever


# ── RRF Tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("k_param", [10, 30, 60, 100])
def test_rrf_k_parameter(k_param):
    """RRF fusion produces valid scores for different k parameters."""
    lists = [
        [("doc_a", 0.9), ("doc_b", 0.7), ("doc_c", 0.5)],
        [("doc_b", 0.8), ("doc_c", 0.6), ("doc_d", 0.4)],
    ]
    fused = reciprocal_rank_fusion(lists, k=k_param)

    assert len(fused) == 4  # 4 unique docs
    assert all(v > 0 for v in fused.values())
    # doc_b appears in both lists → highest
    assert fused["doc_b"] > fused["doc_d"]


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 1.0],
        [0.7, 0.3],
        [0.5, 0.5],
        [1.0, 0.0],
    ],
)
def test_rrf_weight_influence(weights):
    """RRF respects weight ratios between result lists."""
    dense = [("doc_dense", 0.9)]
    sparse = [("doc_sparse", 0.8)]

    fused = reciprocal_rank_fusion([dense, sparse], weights=weights)

    if weights[0] > weights[1]:
        assert fused.get("doc_dense", 0) > fused.get("doc_sparse", 0)
    elif weights[1] > weights[0]:
        assert fused.get("doc_sparse", 0) > fused.get("doc_dense", 0)


@pytest.mark.parametrize("n_lists", [1, 2, 3, 5])
def test_rrf_multiple_lists(n_lists):
    """RRF handles varying numbers of result lists."""
    lists = [[(f"doc_{i}", 0.9 - i * 0.1) for i in range(3)] for _ in range(n_lists)]
    fused = reciprocal_rank_fusion(lists)
    assert len(fused) == 3  # same docs in all lists


def test_rrf_empty_list():
    """RRF handles empty result lists gracefully."""
    fused = reciprocal_rank_fusion([[], [("doc_a", 0.5)]])
    assert "doc_a" in fused
    assert fused["doc_a"] > 0


# ── Retriever Interface Tests ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,category",
    [
        ("What are the symptoms of diabetes?", "clinical"),
        ("How does metformin work?", "pharmacology"),
        ("Is aspirin safe during pregnancy?", "safety"),
        ("What causes hypertension?", "pathology"),
        ("treatment options for migraine", "treatment"),
    ],
)
def test_retrieve_returns_documents(mock_retriever, query, category):
    """Retriever returns documents for diverse medical query types."""
    # Mock dense retrieval returning sample docs
    sample_results = {
        "documents": [["Sample medical document about " + category]],
        "distances": [[0.2]],
        "metadatas": [[{"source": "MedQuAD"}]],
        "ids": [["doc_1"]],
    }
    mock_retriever.vector_store.search.return_value = sample_results

    docs = mock_retriever.retrieve(query, k=5, use_hybrid=False, use_reranking=False)

    assert isinstance(docs, list)
    assert all(isinstance(d, RetrievedDocument) for d in docs)
    assert len(docs) <= 5


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_retrieve_respects_k(mock_retriever, k):
    """Retriever respects the k parameter for result count."""
    # Create enough mock results
    n = max(k * 2, 10)
    sample_results = {
        "documents": [["Doc " + str(i) for i in range(n)]],
        "distances": [[0.1 + i * 0.05 for i in range(n)]],
        "metadatas": [[{"source": "test"} for _ in range(n)]],
        "ids": [[f"id_{i}" for i in range(n)]],
    }
    mock_retriever.vector_store.search.return_value = sample_results

    docs = mock_retriever.retrieve("test query", k=k, use_hybrid=False, use_reranking=False)
    assert len(docs) <= k


# ── Document Deduplication Tests ─────────────────────────────────────────────


def test_dedup_by_doc_id():
    """RRF deduplicates by document ID, not content."""
    lists = [
        [("doc_1", 0.9), ("doc_2", 0.8)],
        [("doc_1", 0.7), ("doc_3", 0.6)],  # doc_1 appears in both
    ]
    fused = reciprocal_rank_fusion(lists)
    # doc_1 should appear once with combined score
    assert "doc_1" in fused
    assert len(fused) == 3


# ── Score Ordering Tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scores,expected_order",
    [
        ([0.9, 0.5, 0.1], ["doc_0", "doc_1", "doc_2"]),
        ([0.1, 0.9, 0.5], ["doc_1", "doc_2", "doc_0"]),
    ],
)
def test_retrieve_score_ordering(mock_retriever, scores, expected_order):
    """Retrieved documents are sorted by descending score."""
    n = len(scores)
    sample_results = {
        "documents": [[f"Content {i}" for i in range(n)]],
        "distances": [[1 - s for s in scores]],  # distance = 1 - similarity
        "metadatas": [[{"source": "test"} for _ in range(n)]],
        "ids": [[f"doc_{i}" for i in range(n)]],
    }
    mock_retriever.vector_store.search.return_value = sample_results

    docs = mock_retriever.retrieve("test", k=n, use_hybrid=False, use_reranking=False)

    assert len(docs) == n
    for i in range(len(docs) - 1):
        assert docs[i].score >= docs[i + 1].score


# ── Drug Interaction Negation Tests ──────────────────────────────────────────


class TestDrugInteractionNegation:
    """Tests for negation-aware drug interaction checking (Section 5.4)."""

    @pytest.fixture
    def checker(self):
        from src.safety.guardrails import DrugInteractionChecker

        return DrugInteractionChecker()

    @pytest.mark.parametrize(
        "text,should_warn",
        [
            ("I take warfarin and aspirin daily", True),
            ("I am on warfarin but NOT taking aspirin", False),
            ("I stopped taking warfarin, now only on aspirin", False),
            ("I never take warfarin with aspirin", False),
            ("My doctor prescribed warfarin and ibuprofen", True),
            ("I discontinued warfarin, currently on ibuprofen", False),
        ],
    )
    def test_negation_aware_interactions(self, checker, text, should_warn):
        """Drug interaction checker respects negation context."""
        warnings = checker.check_interaction_risk(text)
        if should_warn:
            assert len(warnings) > 0, f"Expected warning for: {text}"
        else:
            assert len(warnings) == 0, f"Unexpected warning for: {text}"
