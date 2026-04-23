"""
Retrieval evaluation: Recall@k, MRR, Context Precision.

Diagnoses whether poor answers come from bad retrieval or bad generation
by measuring retrieval quality independently, without running the LLM.

Usage:
    python evaluation/eval_retrieval.py [--n N] [--k 3 5 10]
    make eval-retrieval   # if you add this to the Makefile

Requires:
    - A running pipeline / retriever (loads from config/settings.py)
    - test_set_v2.json with `relevant_ids` field per test case
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Ensure project root on path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved docs."""
    if not relevant_ids:
        return 1.0  # vacuously true
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """1 / rank of the first relevant document (0 if none found)."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Fraction of retrieved docs that are actually relevant (precision)."""
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    return sum(1 for rid in retrieved_ids if rid in relevant_set) / len(retrieved_ids)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
        if rid in relevant_set
    )
    # Ideal DCG: all relevant docs in top positions
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-category breakdown helper
# ---------------------------------------------------------------------------

def _aggregate_by_category(results: List[Dict]) -> Dict[str, Dict]:
    """Group results by category and compute mean metrics."""
    by_cat: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    summary = {}
    for cat, items in by_cat.items():
        metric_keys = [k for k in items[0].keys() if k not in {"id", "query", "category"}]
        summary[cat] = {
            k: float(np.mean([i[k] for i in items if isinstance(i.get(k), (int, float))]))
            for k in metric_keys
        }
        summary[cat]["n"] = len(items)
    return summary


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(scores: List[float], n: int = 1000, ci: float = 0.95, seed: int = 42):
    rng = np.random.RandomState(seed)
    arr = np.array(scores, dtype=float)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    means = np.array(means)
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def load_retriever():
    """Load the hybrid retriever from the project config."""
    from config.settings import config
    from src.embeddings.medical_embedder import MedicalEmbedder
    from src.embeddings.vector_store import MedicalVectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever

    embedder = MedicalEmbedder(model_name=config.embedding.model_name)
    vector_store = MedicalVectorStore(
        persist_directory=config.knowledge_base.persist_dir,
        collection_name=config.knowledge_base.collection_name,
        embedder=embedder,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        dense_weight=config.retrieval.dense_weight,
        sparse_weight=config.retrieval.sparse_weight,
    )
    retriever.warm_up()
    return retriever


def run_retrieval_eval(
    test_set_path: str,
    k_values: List[int] = [3, 5, 10],
    n_cases: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run retrieval-only evaluation and return metrics dict."""

    test_data = json.loads(Path(test_set_path).read_text())
    cases = test_data.get("test_cases", test_data if isinstance(test_data, list) else [test_data])
    if n_cases:
        cases = cases[:n_cases]

    retriever = load_retriever()

    results = []
    for i, case in enumerate(cases):
        query = case.get("query", "")
        relevant_ids = case.get("relevant_ids", [])
        category = case.get("category", "unknown")

        if not query:
            continue

        try:
            docs = retriever.retrieve(query, k=max(k_values))
            retrieved_ids = [getattr(d, "doc_id", "") or getattr(d, "source", f"doc_{j}") for j, d in enumerate(docs)]
        except Exception as e:
            print(f"  [WARN] Retrieval failed for '{query[:50]}': {e}")
            retrieved_ids = []

        row = {"id": case.get("id", str(i)), "query": query[:60], "category": category}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
        row["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)
        row["context_precision"] = context_precision(retrieved_ids, relevant_ids)
        results.append(row)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(cases)}] {query[:50]!r}")

    if not results:
        print("No results to report.")
        return {}

    # Global metrics
    global_metrics = {}
    for k in k_values:
        scores = [r[f"recall@{k}"] for r in results]
        mean, lo, hi = bootstrap_ci(scores)
        global_metrics[f"recall@{k}"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}

        ndcg_scores = [r[f"ndcg@{k}"] for r in results]
        mean_n, lo_n, hi_n = bootstrap_ci(ndcg_scores)
        global_metrics[f"ndcg@{k}"] = {"mean": round(mean_n, 4), "ci95": [round(lo_n, 4), round(hi_n, 4)]}

    mrr_scores = [r["mrr"] for r in results]
    mean_mrr, lo_mrr, hi_mrr = bootstrap_ci(mrr_scores)
    global_metrics["mrr"] = {"mean": round(mean_mrr, 4), "ci95": [round(lo_mrr, 4), round(hi_mrr, 4)]}

    cp_scores = [r["context_precision"] for r in results]
    mean_cp, lo_cp, hi_cp = bootstrap_ci(cp_scores)
    global_metrics["context_precision"] = {"mean": round(mean_cp, 4), "ci95": [round(lo_cp, 4), round(hi_cp, 4)]}

    # Per-category breakdown
    category_metrics = _aggregate_by_category(results)

    output = {
        "n_cases": len(results),
        "k_values": k_values,
        "global": global_metrics,
        "by_category": category_metrics,
        "per_case": results,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL RESULTS  (n={len(results)} queries)")
    print("=" * 60)
    for metric, vals in global_metrics.items():
        m = vals["mean"]
        lo, hi = vals["ci95"]
        print(f"  {metric:<25} {m:.4f}  [95% CI {lo:.4f}-{hi:.4f}]")

    print("\nBy Category:")
    for cat, cmetrics in sorted(category_metrics.items()):
        n = cmetrics.get("n", "?")
        r5 = cmetrics.get("recall@5", 0)
        mrr = cmetrics.get("mrr", 0)
        print(f"  {cat:<15}  n={n:<4}  Recall@5={r5:.3f}  MRR={mrr:.3f}")
    print("=" * 60 + "\n")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation")
    parser.add_argument(
        "--test-set",
        default="evaluation/test_set_v2.json",
        help="Path to test set JSON",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit number of test cases")
    parser.add_argument(
        "--k", nargs="+", type=int, default=[3, 5, 10], help="k values for Recall@k and nDCG@k"
    )
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    run_retrieval_eval(
        test_set_path=args.test_set,
        k_values=args.k,
        n_cases=args.n,
        output_path=args.output,
        verbose=True,
    )
