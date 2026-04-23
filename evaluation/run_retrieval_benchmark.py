#!/usr/bin/env python3
"""
Retrieval benchmark on gold labels (plan B2).

Metrics: hit@k, MRR, precision@k, recall@k (when exact IDs provided), mean nDCG@k.
Optional bootstrap confidence intervals.

Usage:
  PYTHONPATH=. python evaluation/run_retrieval_benchmark.py --gold evaluation/data/gold_retrieval.jsonl --k 10
  PYTHONPATH=. python evaluation/run_retrieval_benchmark.py --smoke  # 5 queries only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _doc_matches_gold(doc, item: Dict[str, Any]) -> bool:
    ids: Set[str] = set(item.get("relevant_doc_ids") or [])
    if ids and getattr(doc, "doc_id", "") in ids:
        return True
    src = (getattr(doc, "source", "") or "").lower()
    for s in item.get("gold_source_substrings") or []:
        if s.lower() in src:
            return True
    content = (getattr(doc, "content", "") or "").lower()
    kws = [k.lower() for k in item.get("gold_content_keywords") or []]
    if len(kws) >= 2:
        hit = sum(1 for k in kws if k in content)
        return hit >= max(1, len(kws) // 2)
    if len(kws) == 1:
        return kws[0] in content
    return False


def _binary_relevance_vector(docs: List[Any], item: Dict[str, Any]) -> List[int]:
    return [1 if _doc_matches_gold(d, item) else 0 for d in docs]


def _mrr(rel: List[int]) -> float:
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def _precision_at_k(rel: List[int], k: int) -> float:
    top = rel[:k]
    return sum(top) / k if k else 0.0


def _recall_at_k(docs: List[Any], item: Dict[str, Any], k: int) -> Optional[float]:
    """Fraction of gold document IDs that appear in the top-k retrieval list.

    Returns None when the row has no exact gold IDs (soft labels only).
    """
    gold_ids = set(item.get("relevant_doc_ids") or [])
    if not gold_ids:
        return None
    retrieved = {getattr(d, "doc_id", "") or "" for d in docs[:k]}
    hits = len(gold_ids & retrieved)
    return hits / len(gold_ids)


def _ndcg_at_k(rel: List[int], k: int) -> float:
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sorted(rel[:k], reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _bootstrap_ci(values: List[float], n_boot: int = 400, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi


def _evaluate_rows(
    retriever,
    rows: List[Dict[str, Any]],
    k: int,
    bootstrap_n: int,
) -> Dict[str, Any]:
    hits, mrrs, precs, recalls, ndcgs = [], [], [], [], []
    by_risk: Dict[str, List[float]] = {}

    for item in rows:
        q = item.get("question", "")
        if not q:
            continue
        docs = retriever.retrieve(q, k=k, use_reranking=True)
        rel = _binary_relevance_vector(docs, item)
        hit = 1.0 if any(rel[:k]) else 0.0
        hits.append(hit)
        mrrs.append(_mrr(rel))
        precs.append(_precision_at_k(rel, k))
        rec = _recall_at_k(docs, item, k)
        if rec is not None:
            recalls.append(rec)
        ndcgs.append(_ndcg_at_k(rel, k))
        risk = item.get("risk_class", "unknown")
        by_risk.setdefault(risk, []).append(hit)

    n = len(hits)
    if n == 0:
        return {"n_queries": 0}

    return {
        f"hit@{k}": sum(hits) / n,
        f"mrr@{k}": sum(mrrs) / n,
        f"precision@{k}": sum(precs) / n,
        f"recall@{k}": (sum(recalls) / len(recalls)) if recalls else None,
        f"ndcg@{k}": sum(ndcgs) / n,
        "n_queries": n,
        "n_queries_with_exact_recall_labels": len(recalls),
        "by_risk": {rk: sum(v) / len(v) for rk, v in by_risk.items()},
        f"hit@{k}_ci95": _bootstrap_ci(hits, n_boot=bootstrap_n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=PROJECT_ROOT / "evaluation" / "data" / "gold_retrieval.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="Evaluate only first 5 rows")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--out", type=Path, default=None, help="Write JSON metrics here")
    ap.add_argument("--sweep-weights", action="store_true", help="Run static dense/sparse weight sweep")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Gold file not found: {args.gold}", file=sys.stderr)
        return 2

    rows = _load_gold(args.gold)
    if args.smoke:
        rows = rows[:5]

    from config.settings import config

    persist = Path(config.retrieval.persist_directory)
    if not persist.exists():
        print(f"KB path missing ({persist}); skipping live retrieval benchmark.", file=sys.stderr)
        print("VALIDATION_ONLY: gold file loaded OK, {} rows".format(len(rows)))
        return 0

    # Load retriever without importing FastAPI app
    from src.embeddings.embedding_models import MedicalEmbedder
    from src.embeddings.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever

    emb_name = getattr(config.embedding, "model_name", None) or "sentence-transformers/all-MiniLM-L6-v2"
    embedder = MedicalEmbedder(model_name=emb_name)
    vector_store = VectorStore(
        collection_name=config.retrieval.collection_name,
        persist_directory=str(config.retrieval.persist_directory),
    )
    rc = config.retrieval
    retriever = HybridRetriever(
        embedder,
        vector_store,
        reranker=None,
        min_score=float(getattr(rc, "min_retrieval_score", 0.0)),
        context_window_sentences=int(getattr(rc, "context_window_sentences", 0)),
        bm25_k1=float(getattr(rc, "bm25_k1", 1.2)),
        bm25_b=float(getattr(rc, "bm25_b", 0.5)),
        use_adaptive_fusion=bool(getattr(rc, "use_adaptive_fusion", True)),
        enable_mmr_diversity=bool(getattr(rc, "enable_mmr_diversity", False)),
        mmr_lambda=float(getattr(rc, "mmr_lambda", 0.5)),
    )

    if args.sweep_weights:
        sweep_pairs = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.8, 0.2)]
        sweep_results: Dict[str, Dict[str, Any]] = {}
        for dense_w, sparse_w in sweep_pairs:
            sweep_retriever = HybridRetriever(
                embedder,
                vector_store,
                reranker=None,
                min_score=float(getattr(rc, "min_retrieval_score", 0.0)),
                context_window_sentences=int(getattr(rc, "context_window_sentences", 0)),
                bm25_k1=float(getattr(rc, "bm25_k1", 1.2)),
                bm25_b=float(getattr(rc, "bm25_b", 0.5)),
                use_adaptive_fusion=False,
                dense_weight=float(dense_w),
                sparse_weight=float(sparse_w),
                enable_mmr_diversity=bool(getattr(rc, "enable_mmr_diversity", False)),
                mmr_lambda=float(getattr(rc, "mmr_lambda", 0.5)),
            )
            key = f"dense_{dense_w:.1f}_sparse_{sparse_w:.1f}"
            sweep_results[key] = _evaluate_rows(
                sweep_retriever, rows, args.k, args.bootstrap
            )

        summary = {"k": args.k, "weights": sweep_results}
        default_out = PROJECT_ROOT / "evaluation" / "results" / "weight_sweep.json"
        out_path = args.out or default_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    summary = _evaluate_rows(retriever, rows, args.k, args.bootstrap)
    n = int(summary.get("n_queries", 0))
    if n == 0:
        print("No rows evaluated.", file=sys.stderr)
        return 1

    from evaluation.manifest import build_manifest, write_manifest

    run_id = Path(args.out).stem if args.out else "latest"
    man = build_manifest(PROJECT_ROOT, extra={"benchmark": "retrieval", "k": args.k, "n": n})
    out_dir = PROJECT_ROOT / "evaluation" / "results" / run_id
    if args.out:
        out_path = args.out
    else:
        out_path = out_dir / "retrieval_benchmark.json"
    write_manifest(out_dir / "manifest.json", man)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
