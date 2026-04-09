"""
Unit tests for RAG metrics module.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rag_metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_retrieval_metrics,
)


class TestPrecisionAtK:
    """Tests for Precision@k calculation."""

    def test_all_relevant(self):
        """All retrieved docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert calculate_precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        """No retrieved docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert calculate_precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_relevant(self):
        """Some retrieved docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc5"}
        # 2 relevant in top 5 = 0.4
        assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.4

    def test_k_greater_than_retrieved(self):
        """k is larger than retrieved list."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # 2 relevant in top 5 = 0.4
        assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.4


class TestRecallAtK:
    """Tests for Recall@k calculation."""

    def test_all_relevant_retrieved(self):
        """All relevant docs are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert calculate_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self):
        """Some relevant docs retrieved."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc5", "doc7"}
        # 2 of 3 relevant = 0.67
        assert abs(calculate_recall_at_k(retrieved, relevant, k=5) - 0.667) < 0.01

    def test_no_relevant_set(self):
        """Empty relevant set returns 0."""
        retrieved = ["doc1", "doc2"]
        relevant = set()
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank calculation."""

    def test_first_is_relevant(self):
        """First retrieved doc is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert calculate_mrr(retrieved, relevant) == 1.0

    def test_second_is_relevant(self):
        """Second retrieved doc is first relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        assert calculate_mrr(retrieved, relevant) == 0.5

    def test_no_relevant(self):
        """No relevant docs in retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc5"}
        assert calculate_mrr(retrieved, relevant) == 0.0


class TestHitRate:
    """Tests for Hit Rate calculation."""

    def test_hit(self):
        """At least one relevant in top k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        assert calculate_hit_rate(retrieved, relevant, k=3) == 1.0

    def test_miss(self):
        """No relevant in top k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc5"}
        assert calculate_hit_rate(retrieved, relevant, k=3) == 0.0


class TestNDCG:
    """Tests for NDCG@k calculation."""

    def test_perfect_ranking(self):
        """Docs retrieved in perfect relevance order."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 3, "doc2": 2, "doc3": 1}
        # Perfect ranking should give NDCG = 1.0
        assert calculate_ndcg_at_k(retrieved, relevance, k=3) == 1.0

    def test_reversed_ranking(self):
        """Docs in reverse relevance order."""
        retrieved = ["doc3", "doc2", "doc1"]
        relevance = {"doc1": 3, "doc2": 2, "doc3": 1}
        # Reversed order gives lower NDCG
        ndcg = calculate_ndcg_at_k(retrieved, relevance, k=3)
        assert 0 < ndcg < 1.0


class TestAggregateMetrics:
    """Tests for metrics aggregation."""

    def test_aggregate_identical(self):
        """Aggregate identical metrics."""
        m1 = RetrievalMetrics(
            precision_at_k=0.5, recall_at_k=0.5, hit_rate=1.0, mrr=0.5, ndcg_at_k=0.5
        )
        agg = aggregate_metrics([m1, m1])

        assert agg["precision_at_k"]["mean"] == 0.5
        assert agg["precision_at_k"]["std"] == 0.0

    def test_aggregate_different(self):
        """Aggregate different metrics."""
        m1 = RetrievalMetrics(0.4, 0.6, 1.0, 0.5, 0.5)
        m2 = RetrievalMetrics(0.6, 0.8, 1.0, 1.0, 0.7)
        agg = aggregate_metrics([m1, m2])

        assert agg["precision_at_k"]["mean"] == 0.5
        assert agg["precision_at_k"]["min"] == 0.4
        assert agg["precision_at_k"]["max"] == 0.6
