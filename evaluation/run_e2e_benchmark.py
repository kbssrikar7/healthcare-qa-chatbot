#!/usr/bin/env python3
"""
End-to-end benchmark harness (plan B3).

Modes:
  --offline   Score retrieval-hit proxy + optional keyword checks from gold file (no LLM).
  --api URL   Call POST /ask for each question and record latency + HTTP status.

Writes evaluation/results/<run_id>/e2e_benchmark.json and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=PROJECT_ROOT / "evaluation" / "data" / "gold_retrieval.jsonl")
    ap.add_argument("--api", type=str, default=os.getenv("E2E_API_URL", ""), help="Base URL e.g. http://127.0.0.1:8000")
    ap.add_argument("--offline", action="store_true", help="No HTTP; keyword/retrieval proxy only")
    ap.add_argument("--run-id", type=str, default="e2e_latest")
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Missing gold file: {args.gold}", file=sys.stderr)
        return 2

    rows = _load_gold(args.gold)[:50]
    results: List[Dict[str, Any]] = []

    if args.offline or not args.api:
        from config.settings import config

        persist = Path(config.retrieval.persist_directory)
        if not persist.exists():
            print("Offline mode: KB missing; writing empty summary.")
            summary = {"mode": "offline", "n": 0, "note": "no_kb"}
        else:
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
            hits = []
            for item in rows:
                q = item.get("question", "")
                if not q:
                    continue
                docs = retriever.retrieve(q, k=5, use_reranking=True)
                rel = any(
                    (s.lower() in (d.source or "").lower())
                    for d in docs
                    for s in (item.get("gold_source_substrings") or [])
                )
                kw = item.get("gold_content_keywords") or []
                rel = rel or any(
                    any(k.lower() in (d.content or "").lower() for k in kw) for d in docs
                )
                hits.append(1.0 if rel else 0.0)
                results.append({"id": item.get("id"), "retrieval_hit_proxy": bool(rel)})
            summary = {
                "mode": "offline",
                "n": len(hits),
                "mean_retrieval_hit_proxy": sum(hits) / len(hits) if hits else 0.0,
            }
    else:
        import requests
        from evaluation.medical_metrics import MedicalQAEvaluator

        base = args.api.rstrip("/")
        hdr = {}
        key = os.getenv("API_KEY", "")
        if key:
            hdr["X-API-Key"] = key
        med_eval = MedicalQAEvaluator()
        med_rows = []
        for item in rows:
            q = item.get("question", "")
            if not q:
                continue
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"{base}/ask",
                    json={"question": q, "num_sources": 5, "include_explanation": False},
                    headers=hdr,
                    timeout=120,
                )
                dt = (time.perf_counter() - t0) * 1000
                ok = r.status_code == 200
                payload = r.json() if ok else {}
                ans = (payload.get("answer") or "").lower()
                kws = [k.lower() for k in item.get("gold_content_keywords") or []]
                kw_hit = sum(1 for k in kws if k in ans) >= max(1, len(kws) // 2) if kws else None
                results.append(
                    {
                        "id": item.get("id"),
                        "status": r.status_code,
                        "latency_ms": dt,
                        "answer_keyword_overlap": kw_hit,
                    }
                )
                reference_answer = item.get("reference_answer")
                if ok and reference_answer:
                    try:
                        med = med_eval.evaluate(answer=payload.get("answer", ""), reference=reference_answer)
                        med_rows.append(med)
                        results[-1]["entity_accuracy"] = med.get("entity_accuracy", 0.0)
                        results[-1]["harm_score"] = med.get("harm_score", 0.0)
                    except Exception as med_err:
                        results[-1]["medical_eval_error"] = str(med_err)
            except Exception as e:
                results.append({"id": item.get("id"), "error": str(e)})
        summary = {"mode": "api", "n": len(results)}
        if med_rows:
            summary["mean_entity_accuracy"] = sum(
                float(m.get("entity_accuracy", 0.0)) for m in med_rows
            ) / len(med_rows)
            summary["mean_harm_score"] = sum(
                float(m.get("harm_score", 0.0)) for m in med_rows
            ) / len(med_rows)

    out_dir = PROJECT_ROOT / "evaluation" / "results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e_benchmark.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8")

    from evaluation.manifest import build_manifest, write_manifest

    write_manifest(out_dir / "manifest.json", build_manifest(PROJECT_ROOT, extra={"benchmark": "e2e", "summary": summary}))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
