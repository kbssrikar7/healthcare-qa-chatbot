#!/usr/bin/env python3
"""
Pipeline profiler — runs a set of benchmark questions against the API and
prints per-query-type latency and quality breakdowns.

Usage:
  python scripts/profile_pipeline.py
  python scripts/profile_pipeline.py --base http://localhost:8000 --repeats 3
  python scripts/profile_pipeline.py --json-out results/profiling.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from collections import defaultdict
from typing import Any

BENCHMARK_QUESTIONS = [
    {"question": "What are early symptoms of Alzheimer's disease?", "type": "symptom"},
    {"question": "How does type 2 diabetes differ from type 1?", "type": "comparison"},
    {"question": "What causes chronic fatigue syndrome?", "type": "definition"},
    {"question": "What is the typical recovery time for a sprained ankle?", "type": "default"},
    {"question": "How is hypertension managed according to the latest clinical guidelines?", "type": "default"},
    {"question": "What are the side effects of metformin?", "type": "drug"},
    {"question": "What is the mechanism of action of ibuprofen?", "type": "drug"},
    {"question": "What are symptoms of iron deficiency anemia?", "type": "symptom"},
    {"question": "How does asthma differ from COPD?", "type": "comparison"},
    {"question": "What is hypothyroidism?", "type": "definition"},
]


def ask(base: str, question: str, headers: dict) -> dict[str, Any]:
    body = json.dumps({"question": question, "num_sources": 5}).encode()
    req = urllib.request.Request(
        base + "/ask", data=body, headers=headers, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode()
    wall_ms = (time.perf_counter() - t0) * 1000
    data = json.loads(raw)
    data["_wall_ms"] = wall_ms
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Profile the Healthcare QA pipeline")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--repeats", type=int, default=1, help="Runs per question")
    p.add_argument("--json-out", default="", help="Save full results to JSON file")
    args = p.parse_args()

    base = args.base.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("API_KEY", ""):
        headers["X-API-Key"] = key

    # Per query-type aggregation
    by_type: dict[str, list[dict]] = defaultdict(list)
    all_results = []

    print(f"\nProfiling {len(BENCHMARK_QUESTIONS)} questions × {args.repeats} repeat(s)\n")
    print(f"{'Question':<55} {'Type':<12} {'Wall ms':>8} {'Retrieval ms':>13} {'Gen ms':>8} {'Sources':>8} {'Answerable':>11}")
    print("-" * 120)

    for item in BENCHMARK_QUESTIONS:
        q, qtype = item["question"], item["type"]
        for _ in range(args.repeats):
            try:
                r = ask(base, q, headers)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            sl = r.get("stage_latencies") or {}
            retrieval_ms = sl.get("retrieval_ms", 0)
            gen_ms = sl.get("generation_ms", 0)
            wall_ms = r["_wall_ms"]
            n_sources = r.get("num_sources_returned", 0)
            conf = (r.get("confidence") or {}).get("score", 0)
            answerable = "yes" if r.get("answer", "").strip() and "don't have enough" not in r.get("answer", "") else "no"

            entry = {
                "question": q,
                "type": qtype,
                "wall_ms": round(wall_ms, 1),
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(gen_ms, 1),
                "num_sources": n_sources,
                "confidence": round(conf, 4),
                "answerable": answerable,
            }
            by_type[qtype].append(entry)
            all_results.append(entry)

            short_q = q[:52] + "..." if len(q) > 55 else q
            print(
                f"  {short_q:<55} {qtype:<12} {wall_ms:>8.0f} {retrieval_ms:>13.0f} {gen_ms:>8.0f} {n_sources:>8} {answerable:>11}"
            )

    # Per-type summary
    print("\n" + "=" * 80)
    print("PER QUERY-TYPE SUMMARY")
    print("=" * 80)
    print(f"{'Type':<14} {'N':>4} {'Avg Wall ms':>12} {'Avg Ret ms':>11} {'Avg Gen ms':>11} {'Ans Rate':>9}")
    print("-" * 70)
    for qtype, entries in sorted(by_type.items()):
        n = len(entries)
        avg_wall = statistics.mean(e["wall_ms"] for e in entries)
        avg_ret = statistics.mean(e["retrieval_ms"] for e in entries)
        avg_gen = statistics.mean(e["generation_ms"] for e in entries)
        ans_rate = sum(1 for e in entries if e["answerable"] == "yes") / n
        print(f"  {qtype:<14} {n:>4} {avg_wall:>12.0f} {avg_ret:>11.0f} {avg_gen:>11.0f} {ans_rate:>8.0%}")

    # Overall
    if all_results:
        print("-" * 70)
        print(
            f"  {'TOTAL':<14} {len(all_results):>4} "
            f"{statistics.mean(e['wall_ms'] for e in all_results):>12.0f} "
            f"{statistics.mean(e['retrieval_ms'] for e in all_results):>11.0f} "
            f"{statistics.mean(e['generation_ms'] for e in all_results):>11.0f} "
            f"{sum(1 for e in all_results if e['answerable'] == 'yes') / len(all_results):>8.0%}"
        )

    if args.json_out:
        out = {
            "questions": all_results,
            "by_type": {
                qt: {
                    "count": len(es),
                    "avg_wall_ms": round(statistics.mean(e["wall_ms"] for e in es), 1),
                    "avg_retrieval_ms": round(statistics.mean(e["retrieval_ms"] for e in es), 1),
                    "avg_generation_ms": round(statistics.mean(e["generation_ms"] for e in es), 1),
                    "answer_rate": round(sum(1 for e in es if e["answerable"] == "yes") / len(es), 3),
                }
                for qt, es in by_type.items()
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nFull results saved to {args.json_out}")


if __name__ == "__main__":
    main()
