"""
Retrieval quality metrics: MRR@k, Recall@k, Precision@k, NDCG@k.

These measure the retrieval pipeline independently of the LLM generator —
following the RAG-engineer pattern of evaluating each stage separately.

Usage:
    from evaluation.retrieval_metrics import compute_retrieval_metrics

    results = compute_retrieval_metrics(retriever, test_set, k=5)
    print(results)  # {"mrr@5": 0.72, "recall@5": 0.81, ...}
"""

from __future__ import annotations

import math
from typing import Dict, List

from loguru import logger


# ---------------------------------------------------------------------------
# Core metric functions (pure, no I/O)
# ---------------------------------------------------------------------------


def _reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    """Return 1/rank of the first relevant doc, or 0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    hits = sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    return hits / k if k > 0 else 0.0


def _recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids)


def _ndcg_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain @ k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregation over a test set
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    retriever,
    test_set: List[Dict],
    k: int = 5,
    relevant_field: str = "relevant_doc_ids",
    question_field: str = "question",
) -> Dict[str, float]:
    """
    Compute MRR@k, Precision@k, Recall@k, and NDCG@k over a test set.

    Args:
        retriever: HybridRetriever instance (must support .retrieve(query, k))
        test_set: List of dicts with at least ``question_field`` and
                  ``relevant_field`` keys.
        k: Cut-off rank for all metrics.
        relevant_field: Key in each test item that holds a list of relevant
                        document IDs (strings).  If absent, falls back to
                        keyword-overlap relevance judgment (see below).
        question_field: Key for the question string.

    Returns:
        Dict with keys mrr@k, precision@k, recall@k, ndcg@k, n_queries.
    """
    mrr_scores, prec_scores, rec_scores, ndcg_scores = [], [], [], []
    skipped = 0

    for item in test_set:
        question = item.get(question_field, "")
        if not question:
            skipped += 1
            continue

        # Retrieve
        try:
            docs = retriever.retrieve(question, k=k, use_reranking=True)
        except Exception as e:
            logger.warning(f"Retrieval failed for '{question[:60]}': {e}")
            skipped += 1
            continue

        retrieved_ids = [d.doc_id for d in docs]

        # Determine relevance ground truth
        if relevant_field in item and item[relevant_field]:
            relevant_ids = set(item[relevant_field])
        else:
            # Fallback: keyword-overlap heuristic — a doc is "relevant" if it
            # contains at least 60% of the answer keywords from the test item.
            answer_kw = set(
                w.lower()
                for w in item.get("answer", item.get("keywords", "")).split()
                if len(w) >= 4
            )
            if not answer_kw:
                skipped += 1
                continue
            relevant_ids = {
                d.doc_id
                for d in docs
                if len(answer_kw & set(d.content.lower().split())) / len(answer_kw)
                >= 0.6
            }
            if not relevant_ids:
                # No doc passes threshold — use top-1 as pseudo-relevant
                relevant_ids = {retrieved_ids[0]} if retrieved_ids else set()

        mrr_scores.append(_reciprocal_rank(retrieved_ids, relevant_ids))
        prec_scores.append(_precision_at_k(retrieved_ids, relevant_ids, k))
        rec_scores.append(_recall_at_k(retrieved_ids, relevant_ids, k))
        ndcg_scores.append(_ndcg_at_k(retrieved_ids, relevant_ids, k))

    n = len(mrr_scores)
    if n == 0:
        logger.warning("No valid queries evaluated — check test set format.")
        return {
            "mrr@k": 0.0,
            "precision@k": 0.0,
            "recall@k": 0.0,
            "ndcg@k": 0.0,
            "n_queries": 0,
        }

    logger.info(f"Retrieval eval: {n} queries evaluated, {skipped} skipped (k={k})")

    return {
        f"mrr@{k}": round(sum(mrr_scores) / n, 4),
        f"precision@{k}": round(sum(prec_scores) / n, 4),
        f"recall@{k}": round(sum(rec_scores) / n, 4),
        f"ndcg@{k}": round(sum(ndcg_scores) / n, 4),
        "n_queries": n,
        "n_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Per-query-type breakdown
# ---------------------------------------------------------------------------


def compute_metrics_by_query_type(
    retriever,
    test_set: List[Dict],
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute retrieval metrics split by query type (drug / definition /
    symptom / comparison / default).

    Useful for diagnosing whether adaptive weights actually help per-type.
    """
    buckets: Dict[str, List[Dict]] = {}
    for item in test_set:
        q = item.get("question", "")
        qt = retriever._detect_query_type(q)
        buckets.setdefault(qt, []).append(item)

    results = {}
    for qt, items in buckets.items():
        metrics = compute_retrieval_metrics(retriever, items, k=k)
        metrics["n_queries"] = len(items)
        results[qt] = metrics
        logger.info(
            f"  [{qt:12s}] n={len(items):3d}  "
            f"MRR={metrics[f'mrr@{k}']:.3f}  "
            f"Recall={metrics[f'recall@{k}']:.3f}  "
            f"NDCG={metrics[f'ndcg@{k}']:.3f}"
        )

    return results
