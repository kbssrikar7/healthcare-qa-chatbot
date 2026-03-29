#!/usr/bin/env python3
"""
API-based evaluation runner.

Calls the running FastAPI server (localhost:8000) instead of loading the
pipeline in-process, avoiding memory conflicts with the running API.

Usage:
    python evaluation/run_api_eval.py                    # all modes, standard
    python evaluation/run_api_eval.py --mode metrics
    python evaluation/run_api_eval.py --mode latency
    python evaluation/run_api_eval.py --mode ablation
    python evaluation/run_api_eval.py --n 20             # quick run
"""
import os, sys, json, time, argparse, statistics, warnings
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ── API helper ───────────────────────────────────────────────────────────────
def ask(question: str, model: str = "tinyllama", use_langchain: bool = False,
        use_langgraph: bool = False, num_sources: int = 3) -> dict:
    import requests
    payload = {
        "question": question,
        "model_choice": model,
        "include_explanation": True,
        "num_sources": num_sources,
    }
    if use_langchain:
        payload["use_langchain"] = True
    if use_langgraph:
        payload["use_langgraph"] = True
    resp = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_api() -> bool:
    import requests
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Metric helpers ───────────────────────────────────────────────────────────
def keyword_coverage(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def rouge_l(hyp: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(ref, hyp)["rougeL"].fmeasure
    except Exception:
        return 0.0


def load_test_set(path: Path) -> list:
    with open(path) as f:
        return json.load(f)["test_cases"]


# ── Modes ────────────────────────────────────────────────────────────────────
def run_metrics(cases: list, variant: dict, n: int = None) -> dict:
    run_cases = cases[:n] if n else cases
    name = variant["name"]
    kw_scores, rl_scores, preds, refs = [], [], [], []
    answerable = 0
    print(f"\n  [{name}] {len(run_cases)} questions …")

    for i, case in enumerate(run_cases, 1):
        try:
            r = ask(case["query"],
                    model=variant.get("model", "tinyllama"),
                    use_langchain=variant.get("langchain", False),
                    use_langgraph=variant.get("langgraph", False))
            answer = r.get("answer", "")
            kw = case.get("expected_keywords", [])
            ref = case.get("reference_answer", "")
            kw_scores.append(keyword_coverage(answer, kw))
            if ref:
                rl_scores.append(rouge_l(answer, ref))
                preds.append(answer)
                refs.append(ref)
            if r.get("is_answerable", True):
                answerable += 1
        except Exception as e:
            print(f"    [{i}] ERROR: {e}")
        if i % 10 == 0:
            print(f"    {i}/{len(run_cases)} done")

    # BERTScore (batch)
    bs_f1_mean = 0.0
    if preds:
        try:
            import evaluate as _ev
            bsm = _ev.load("bertscore")
            bs = bsm.compute(predictions=preds, references=refs,
                             lang="en", rescale_with_baseline=True, device="cpu")
            bs_f1_mean = statistics.mean(bs["f1"])
        except Exception as e:
            print(f"    BERTScore error: {e}")

    n_total = len(run_cases)
    return {
        "variant": name,
        "n": n_total,
        "answerable_pct": round(answerable / n_total, 3) if n_total else 0,
        "keyword_coverage_mean": round(statistics.mean(kw_scores), 3) if kw_scores else 0,
        "rougeL_mean": round(statistics.mean(rl_scores), 3) if rl_scores else 0,
        "bertscore_f1_mean": round(bs_f1_mean, 3),
    }


def run_latency(cases: list, variant: dict, n: int = 20) -> dict:
    run_cases = cases[:n]
    name = variant["name"]
    times, stage_totals = [], defaultdict(list)
    print(f"\n  [{name}] latency ({n} questions) …")

    for case in run_cases:
        try:
            t0 = time.perf_counter()
            r = ask(case["query"],
                    model=variant.get("model", "tinyllama"),
                    use_langchain=variant.get("langchain", False),
                    use_langgraph=variant.get("langgraph", False))
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            for k, v in (r.get("stage_latencies") or {}).items():
                stage_totals[k].append(v)
        except Exception as e:
            print(f"    ERROR: {e}")

    result = {"variant": name}
    if times:
        result["total_ms_mean"] = round(statistics.mean(times), 1)
        result["total_ms_p90"] = round(sorted(times)[int(0.9 * len(times))], 1)
    for stage, vals in stage_totals.items():
        result[f"{stage}_mean"] = round(statistics.mean(vals), 1)
    return result


def run_calibration(cases: list, variant: dict, n: int = None) -> dict:
    """Compute ECE from API responses using keyword coverage as correctness proxy."""
    import numpy as np
    run_cases = cases[:n] if n else cases
    confidences, labels = [], []
    print(f"\n  Calibration: {len(run_cases)} questions …")

    for case in run_cases:
        try:
            r = ask(case["query"], model=variant.get("model", "tinyllama"))
            conf = r.get("confidence", {}).get("score", 0.5)
            kw = case.get("expected_keywords", [])
            label = 1 if keyword_coverage(r.get("answer", ""), kw) >= 0.5 else 0
            confidences.append(conf)
            labels.append(label)
        except Exception:
            pass

    if not confidences:
        return {"error": "no predictions"}

    conf_arr = np.array(confidences)
    label_arr = np.array(labels)

    # ECE (10 bins)
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if mask.sum() == 0:
            continue
        acc = label_arr[mask].mean()
        conf = conf_arr[mask].mean()
        count = mask.sum()
        ece += (count / len(conf_arr)) * abs(acc - conf)
        bin_accs.append(float(acc))
        bin_confs.append(float(conf))
        bin_counts.append(int(count))

    # Reliability diagram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        mids = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
        used = [m for m, cnt in zip(mids, bin_counts) if cnt > 0] if bin_counts else []
        if used:
            ax.bar(used, bin_accs, width=0.08, alpha=0.7, color="#4f46e5",
                   edgecolor="black", label="Model")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Fraction correct (KW coverage ≥ 0.5)")
        ax.set_title(f"Reliability Diagram  (ECE = {ece:.3f})")
        ax.legend()
        fig.tight_layout()
        out = PROJECT_ROOT / "evaluation" / "results" / "reliability_diagram.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Saved → {out}")
    except Exception as e:
        print(f"  Matplotlib error: {e}")

    return {
        "ece": round(float(ece), 4),
        "n": len(confidences),
        "mean_confidence": round(float(conf_arr.mean()), 4),
        "mean_label": round(float(label_arr.mean()), 4),
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


# ── Table printer ─────────────────────────────────────────────────────────────
def _print_table(rows: list, title: str):
    if not rows:
        return
    keys = [k for k in rows[0].keys()]
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    sep = "  ".join("-" * widths[k] for k in keys)
    print(f"\n{'=' * len(header)}")
    print(f"  {title}")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="API-based evaluation runner")
    parser.add_argument("--mode", choices=["metrics", "latency", "calibration", "all"],
                        default="all")
    parser.add_argument("--test-set", default="evaluation/test_set_v2.json")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N cases")
    parser.add_argument("--model", default="tinyllama",
                        choices=["tinyllama", "biomistral"],
                        help="Model to use for standard pipeline")
    args = parser.parse_args()

    if not check_api():
        print(f"API not reachable at {API_URL}")
        print("Start with:  python api/main.py &")
        sys.exit(1)

    test_path = PROJECT_ROOT / args.test_set
    cases = load_test_set(test_path)
    print(f"Loaded {len(cases)} test cases")

    out_dir = PROJECT_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Define variants
    VARIANTS = [
        {"name": f"standard_{args.model}", "model": args.model},
        {"name": "langchain", "model": args.model, "langchain": True},
        {"name": "langgraph", "model": args.model, "langgraph": True},
    ]

    # ── Metrics ──────────────────────────────────────────────────────────────
    if args.mode in ("metrics", "all"):
        print("\n[1/3] Quality metrics (ROUGE-L, BERTScore, KW-coverage) …")
        rows = []
        for v in VARIANTS:
            row = run_metrics(cases, v, n=args.n)
            rows.append(row)
        _print_table(rows, "Generation Quality Metrics")
        all_results["metrics"] = rows
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(rows, f, indent=2)

    # ── Latency ──────────────────────────────────────────────────────────────
    if args.mode in ("latency", "all"):
        print("\n[2/3] Per-pipeline latency …")
        rows = []
        for v in VARIANTS:
            row = run_latency(cases, v, n=args.n or 10)
            rows.append(row)
        _print_table(rows, "End-to-End Latency (ms)")
        all_results["latency"] = rows
        with open(out_dir / "latency.json", "w") as f:
            json.dump(rows, f, indent=2)

    # ── Calibration (standard pipeline only) ─────────────────────────────────
    if args.mode in ("calibration", "all"):
        print("\n[3/3] Confidence calibration (ECE) …")
        calib = run_calibration(cases, VARIANTS[0], n=args.n)
        print(f"  ECE = {calib.get('ece', 'N/A')}")
        all_results["calibration"] = calib
        with open(out_dir / "calibration.json", "w") as f:
            json.dump(calib, f, indent=2)

    summary_path = out_dir / "paper_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {summary_path}")


if __name__ == "__main__":
    main()
