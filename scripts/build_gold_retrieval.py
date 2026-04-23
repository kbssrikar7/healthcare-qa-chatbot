#!/usr/bin/env python3
"""
Build expanded retrieval gold files with stable chunk IDs.

Outputs:
  - evaluation/data/gold_retrieval_candidates.jsonl (manual review candidates)
  - evaluation/data/gold_retrieval_expanded.jsonl (auto-labeled baseline set)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _risk_from_category(category: str) -> str:
    c = (category or "").lower()
    if "drug" in c:
        return "drug"
    if "compare" in c:
        return "comparison"
    if "symptom" in c:
        return "symptom"
    if "clinical" in c or "research" in c:
        return "factual"
    return "definition"


def main() -> int:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from config.settings import config
    from src.embeddings.embedding_models import MedicalEmbedder
    from src.embeddings.vector_store import VectorStore
    from src.retrieval.hybrid_retriever import HybridRetriever

    test_set = _load_json(PROJECT_ROOT / "evaluation" / "test_set_v2.json")
    base_gold = _load_jsonl(PROJECT_ROOT / "evaluation" / "data" / "gold_retrieval.jsonl")

    embedder = MedicalEmbedder(model_name=config.embedding.model_name)
    store = VectorStore(
        collection_name=config.retrieval.collection_name,
        persist_directory=str(config.retrieval.persist_directory),
    )
    retriever = HybridRetriever(
        embedder,
        store,
        reranker=None,
        min_score=float(getattr(config.retrieval, "min_retrieval_score", 0.0)),
        context_window_sentences=int(getattr(config.retrieval, "context_window_sentences", 0)),
        bm25_k1=float(getattr(config.retrieval, "bm25_k1", 1.2)),
        bm25_b=float(getattr(config.retrieval, "bm25_b", 0.5)),
        use_adaptive_fusion=bool(getattr(config.retrieval, "use_adaptive_fusion", True)),
        enable_mmr_diversity=bool(getattr(config.retrieval, "enable_mmr_diversity", False)),
        mmr_lambda=float(getattr(config.retrieval, "mmr_lambda", 0.5)),
    )

    candidates: List[Dict[str, Any]] = []
    expanded: List[Dict[str, Any]] = []
    expanded.extend(base_gold)

    seen_questions = {row.get("question", "") for row in base_gold}
    tcases = test_set.get("test_cases", [])
    for i, case in enumerate(tcases):
        q = case.get("query", "").strip()
        if not q or q in seen_questions:
            continue

        docs = retriever.retrieve(q, k=20, use_reranking=True)
        kws = [k.lower() for k in case.get("expected_keywords", [])]
        src = (case.get("source") or "").lower()

        rel_ids: List[str] = []
        for d in docs:
            d_content = (d.content or "").lower()
            d_source = (d.source or "").lower()
            kw_hits = sum(1 for k in kws if k in d_content)
            source_match = bool(src and src.split("-")[0] in d_source)
            if kw_hits >= max(1, len(kws) // 3) or source_match:
                rel_ids.append(d.doc_id)
            if len(rel_ids) >= 3:
                break
        if not rel_ids and docs:
            rel_ids = [docs[0].doc_id]

        row = {
            "id": f"x{i+1:03d}",
            "question": q,
            "risk_class": _risk_from_category(case.get("category", "")),
            "gold_source_substrings": [case.get("source", "")] if case.get("source") else [],
            "gold_content_keywords": case.get("expected_keywords", [])[:6],
            "relevant_doc_ids": rel_ids,
        }
        expanded.append(row)

        candidates.append(
            {
                "id": row["id"],
                "question": q,
                "source_hint": case.get("source", ""),
                "keywords": case.get("expected_keywords", []),
                "top_docs": [
                    {
                        "doc_id": d.doc_id,
                        "source": d.source,
                        "score": float(d.score),
                        "content_preview": (d.content or "")[:220],
                    }
                    for d in docs[:20]
                ],
                "auto_relevant_doc_ids": rel_ids,
            }
        )

    _write_jsonl(PROJECT_ROOT / "evaluation" / "data" / "gold_retrieval_candidates.jsonl", candidates)
    _write_jsonl(PROJECT_ROOT / "evaluation" / "data" / "gold_retrieval_expanded.jsonl", expanded)
    print(
        f"Wrote {len(candidates)} candidates and {len(expanded)} expanded gold rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
