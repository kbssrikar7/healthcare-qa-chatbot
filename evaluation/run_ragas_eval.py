#!/usr/bin/env python3
"""
Ragas evaluation runner for the healthcare QA/RAG pipeline.

This script evaluates the real pipeline on the existing evaluation test set and
produces a Ragas report using a non-LLM metric subset that works offline:
  - NonLLMContextPrecisionWithReference
  - NonLLMContextRecall
  - RougeScore
  - ExactMatch

Notes:
  - The current test set does not contain gold supporting passages. As a
    pragmatic fallback, this runner uses `reference_answer` as a single
    pseudo-reference context for retrieval overlap metrics.
  - This gives a useful relative signal, but it is not the same as evaluating
    against gold document chunks.

Usage:
  venv/bin/python evaluation/run_ragas_eval.py --n 5
  venv/bin/python evaluation/run_ragas_eval.py --variant langgraph --model tinyllama
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).parent.parent

# Prefer local caches to avoid long remote retries during evaluation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    ExactMatch,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    RougeScore,
)

from evaluation.run_paper_eval import load_pipeline, load_test_set


def extract_source_texts(response) -> list[str]:
    texts: list[str] = []
    for source in getattr(response, "sources", []) or []:
        if isinstance(source, dict):
            text = source.get("content") or source.get("text") or source.get("snippet")
        else:
            text = getattr(source, "content", None) or getattr(source, "text", None)
        if text:
            cleaned = str(text).strip()
            if cleaned:
                texts.append(cleaned)
    return texts


def build_samples(test_cases: list[dict], pipeline, num_sources: int) -> list[dict]:
    rows: list[dict] = []
    total = len(test_cases)
    print(f"Running pipeline on {total} evaluation cases...")

    for idx, case in enumerate(test_cases, start=1):
        question = case["query"]
        reference_answer = (case.get("reference_answer") or "").strip()

        try:
            response = pipeline.answer(
                question,
                num_documents=num_sources,
                include_explanation=False,
            )
            answer = (getattr(response, "answer", "") or "").strip()
            retrieved_contexts = extract_source_texts(response)
        except Exception as exc:
            print(f"  [{idx}/{total}] ERROR for {case.get('id', 'unknown')}: {exc}")
            answer = ""
            retrieved_contexts = []

        # Pragmatic fallback: the dataset has reference answers but not gold
        # support passages, so use the reference answer as pseudo reference
        # context for non-LLM retrieval overlap metrics. Keep this non-empty to
        # avoid NaN/empty-iterable failures in context-recall style metrics.
        reference_contexts = [reference_answer] if reference_answer else [question]

        rows.append(
            {
                "id": case.get("id"),
                "user_input": question,
                "response": answer,
                "reference": reference_answer,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
                "category": case.get("category"),
                "source_dataset": case.get("source"),
                "num_retrieved_contexts": len(retrieved_contexts),
            }
        )

        if idx % 5 == 0 or idx == total:
            print(f"  {idx}/{total} complete")

    return rows


def aggregate_scores(result_rows: list[dict], metric_names: list[str]) -> dict:
    summary = {}
    for metric_name in metric_names:
        values = []
        for row in result_rows:
            if metric_name not in row:
                continue
            value = row[metric_name]
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            values.append(float(value))
        if values:
            summary[metric_name] = {
                "mean": round(mean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the QA/RAG pipeline with Ragas")
    parser.add_argument("--test-set", default="evaluation/test_set_v2.json")
    parser.add_argument("--variant", default="standard", choices=["standard", "langchain", "langgraph"])
    parser.add_argument("--model", default="tinyllama")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--n", type=int, default=5, help="Number of evaluation samples to run")
    parser.add_argument("--num-sources", type=int, default=5)
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    test_path = PROJECT_ROOT / args.test_set
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        print(f"Test set not found: {test_path}")
        return 1

    test_cases = load_test_set(test_path)
    if args.n:
        test_cases = test_cases[: args.n]

    print(f"Loading pipeline: variant={args.variant} model={args.model}")
    t0 = time.perf_counter()
    pipeline = load_pipeline(args.variant, args.model, args.adapter_path)
    print(f"Pipeline loaded in {(time.perf_counter() - t0):.1f}s")

    raw_rows = build_samples(test_cases, pipeline, args.num_sources)
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "reference": row["reference"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference_contexts": row["reference_contexts"],
            }
            for row in raw_rows
        ]
    )

    metrics = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
        RougeScore(),
        ExactMatch(),
    ]

    print("Running Ragas evaluation...")
    result = evaluate(dataset, metrics=metrics)
    result_rows = result.to_pandas().to_dict(orient="records")

    metric_names = [
        "non_llm_context_precision_with_reference",
        "non_llm_context_recall",
        "rouge_score(mode=fmeasure)",
        "exact_match",
    ]

    combined_rows = []
    for raw, scored in zip(raw_rows, result_rows):
        merged = dict(raw)
        merged.update({name: scored.get(name) for name in metric_names})
        combined_rows.append(merged)

    summary = {
        "ragas_version": "0.4.3",
        "test_set": args.test_set,
        "variant": args.variant,
        "model": args.model,
        "n": len(combined_rows),
        "num_sources": args.num_sources,
        "metrics": aggregate_scores(combined_rows, metric_names),
        "notes": [
            "reference_contexts are approximated from reference_answer because the test set lacks gold supporting passages",
            "this runner uses non-LLM metrics only, so it does not require evaluator API keys",
        ],
    }

    stem = f"ragas_{args.variant}_{args.model}_{len(combined_rows)}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    json_path.write_text(json.dumps({"summary": summary, "rows": combined_rows}, indent=2))

    import csv

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "category",
                "source_dataset",
                "user_input",
                "response",
                "reference",
                "num_retrieved_contexts",
                *metric_names,
            ],
        )
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "source_dataset": row.get("source_dataset"),
                    "user_input": row.get("user_input"),
                    "response": row.get("response"),
                    "reference": row.get("reference"),
                    "num_retrieved_contexts": row.get("num_retrieved_contexts"),
                    **{name: row.get(name) for name in metric_names},
                }
            )

    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
