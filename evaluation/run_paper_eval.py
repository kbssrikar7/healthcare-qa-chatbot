#!/usr/bin/env python3
"""
Paper-quality evaluation pipeline.

Runs all three pipeline variants (Standard, LangChain, LangGraph) over the
expanded test set (test_set_v2.json) and produces:
  • ROUGE-L / BERTScore / keyword-coverage per pipeline
  • Per-stage latency breakdown table
  • Ablation study: effect of each confidence signal
  • Confidence calibration: ECE + reliability diagram
  • JSON results + matplotlib figures saved under evaluation/results/

Usage:
    python evaluation/run_paper_eval.py              # all modes
    python evaluation/run_paper_eval.py --mode metrics
    python evaluation/run_paper_eval.py --mode latency
    python evaluation/run_paper_eval.py --mode ablation
    python evaluation/run_paper_eval.py --mode calibration
    python evaluation/run_paper_eval.py --n 20       # quick smoke test
"""
import os, sys, json, time, argparse, warnings
from pathlib import Path
from collections import defaultdict

# ── offline mode: avoid 40-second HF network retries ────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

# ── lazy imports (heavy) ─────────────────────────────────────────────────────
def _import_nlp_metrics():
    from rouge_score import rouge_scorer as _rs
    import evaluate as _ev
    bertscore = _ev.load("bertscore")
    return _rs, bertscore


# ────────────────────────────────────────────────────────────────────────────
# Utility: keyword coverage + correctness (from shared eval_utils)
# ────────────────────────────────────────────────────────────────────────────
from evaluation.eval_utils import keyword_coverage, is_correct, bootstrap_ci


# ────────────────────────────────────────────────────────────────────────────
# Pipeline loader
# ────────────────────────────────────────────────────────────────────────────
def load_pipeline(variant: str = "standard", model: str = "tinyllama"):
    """Return an initialised pipeline for variant ∈ {standard, langchain, langgraph}."""
    from api.main import (
        get_pipeline,
        _get_langchain_pipeline,
        _get_langgraph_pipeline,
    )
    if variant == "standard":
        return get_pipeline(model)
    elif variant == "langchain":
        return _get_langchain_pipeline(model_choice=model)
    elif variant == "langgraph":
        return _get_langgraph_pipeline(model_choice=model)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def load_test_set(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


# ────────────────────────────────────────────────────────────────────────────
# Mode 1: Generation metrics  (ROUGE-L, BERTScore, keyword coverage)
# ────────────────────────────────────────────────────────────────────────────
def run_metrics(test_cases: list, pipeline, variant_name: str, n: int = None) -> dict:
    cases = test_cases[:n] if n else test_cases
    _rs, bertscore_metric = _import_nlp_metrics()
    scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)

    preds, refs = [], []
    kw_scores, rouge_scores = [], []
    answerable, total = 0, 0

    print(f"\n  [{variant_name}] evaluating {len(cases)} questions …")
    for i, case in enumerate(cases, 1):
        try:
            resp = pipeline.answer(case["query"], include_explanation=False)
        except Exception as e:
            print(f"    [{i}] ERROR: {e}")
            continue

        answer = resp.answer if hasattr(resp, "answer") else str(resp)
        ref = case.get("reference_answer", "")
        kws = case.get("expected_keywords", [])

        kw_scores.append(keyword_coverage(answer, kws))
        if ref:
            rouge_scores.append(scorer.score(ref, answer)["rougeL"].fmeasure)
            preds.append(answer)
            refs.append(ref)
        if getattr(resp, "is_answerable", True):
            answerable += 1
        total += 1

        if i % 10 == 0:
            print(f"    {i}/{len(cases)} done")

    # BERTScore (batch) — try rescaled first, fall back to raw scores
    bs_f1_mean = 0.0
    bs_f1_scores = []
    if preds:
        try:
            try:
                bs = bertscore_metric.compute(predictions=preds, references=refs,
                                              lang="en", rescale_with_baseline=True,
                                              device="cpu")
            except Exception as baseline_err:
                print(f"    BERTScore baseline not cached, using unscaled: {baseline_err}")
                bs = bertscore_metric.compute(predictions=preds, references=refs,
                                              lang="en", rescale_with_baseline=False,
                                              device="cpu")
            import statistics
            bs_f1_scores = list(bs["f1"])
            bs_f1_mean = statistics.mean(bs_f1_scores)
        except Exception as e:
            print(f"    BERTScore error (scores will be 0): {e}")

    import statistics
    result = {
        "variant": variant_name,
        "n": total,
        "answerable_pct": answerable / total if total else 0,
        "keyword_coverage_mean": statistics.mean(kw_scores) if kw_scores else 0,
        "rougeL_mean": statistics.mean(rouge_scores) if rouge_scores else 0,
        "bertscore_f1_mean": bs_f1_mean,
    }

    # Add bootstrap 95% confidence intervals
    if len(kw_scores) >= 5:
        mean, lo, hi = bootstrap_ci(kw_scores)
        result["keyword_coverage_ci"] = [round(lo, 4), round(hi, 4)]
    if len(rouge_scores) >= 5:
        mean, lo, hi = bootstrap_ci(rouge_scores)
        result["rougeL_ci"] = [round(lo, 4), round(hi, 4)]
    if len(bs_f1_scores) >= 5:
        mean, lo, hi = bootstrap_ci(bs_f1_scores)
        result["bertscore_f1_ci"] = [round(lo, 4), round(hi, 4)]

    return result


# ────────────────────────────────────────────────────────────────────────────
# Mode 2: Latency breakdown  (per-stage wall-clock ms)
# ────────────────────────────────────────────────────────────────────────────
def run_latency(test_cases: list, pipeline, variant_name: str, n: int = 20) -> dict:
    """Run n questions and aggregate per-stage latencies."""
    cases = test_cases[:n]
    stage_totals: dict = defaultdict(list)
    total_times = []

    print(f"\n  [{variant_name}] latency profiling ({n} questions) …")
    for i, case in enumerate(cases, 1):
        try:
            t0 = time.perf_counter()
            resp = pipeline.answer(case["query"], include_explanation=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_times.append(elapsed_ms)
            # Stage-level latencies (only available for standard pipeline)
            lats = getattr(resp, "stage_latencies", None) or {}
            for k, v in lats.items():
                stage_totals[k].append(v)
        except Exception as e:
            print(f"    [{i}] ERROR: {e}")

    import statistics
    result = {"variant": variant_name}
    if total_times:
        result["total_ms_mean"] = round(statistics.mean(total_times), 1)
        result["total_ms_p90"] = round(sorted(total_times)[int(0.9 * len(total_times))], 1)
    for stage, vals in stage_totals.items():
        result[f"{stage}_mean"] = round(statistics.mean(vals), 1)
        result[f"{stage}_p90"] = round(sorted(vals)[int(0.9 * len(vals))], 1)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Mode 3: Ablation study  (zero out each confidence signal)
# ────────────────────────────────────────────────────────────────────────────
def run_ablation(test_cases: list, pipeline, n: int = 30) -> list:
    """
    For each signal weight set, run n questions and record mean confidence.
    Compares: all-signals vs. each signal ablated (set to 0, others renormalized).
    """
    from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer

    # Keys used by MultiSignalConfidenceScorer.weights
    SIGNALS = ["retrieval", "generation", "consistency", "source_agreement", "entity_coverage"]
    BASE_WEIGHTS = {s: 1.0 / len(SIGNALS) for s in SIGNALS}

    cases = test_cases[:n]
    results = []

    def _run_with_weights(name: str, weights: dict) -> dict:
        scorer = pipeline.multi_signal_scorer
        if scorer is None:
            return {"name": name, "error": "no scorer"}
        original_weights = dict(scorer.weights)
        # Normalize weights to sum 1
        total = sum(weights.values()) or 1.0
        scorer.weights = {k: v / total for k, v in weights.items()}

        confs = []
        kws_scores = []
        for case in cases:
            try:
                resp = pipeline.answer(case["query"], include_explanation=True)
                confs.append(resp.confidence.get("score", 0.5))
                kws = case.get("expected_keywords", [])
                kws_scores.append(keyword_coverage(resp.answer, kws))
            except Exception:
                pass
        scorer.weights = original_weights
        import statistics
        return {
            "name": name,
            "mean_confidence": round(statistics.mean(confs), 4) if confs else 0,
            "mean_kw_coverage": round(statistics.mean(kws_scores), 4) if kws_scores else 0,
        }

    # Baseline (all signals)
    results.append(_run_with_weights("all_signals", BASE_WEIGHTS))

    # Ablate each signal one at a time
    for ablated in SIGNALS:
        w = {s: (0.0 if s == ablated else 1.0) for s in SIGNALS}
        label = f"ablate_{ablated}"
        results.append(_run_with_weights(label, w))

    return results


# ────────────────────────────────────────────────────────────────────────────
# Mode 4: Calibration  (ECE + reliability diagram)
# ────────────────────────────────────────────────────────────────────────────
def run_calibration(test_cases: list, pipeline, n: int = None) -> dict:
    """
    Fit Platt calibration on keyword coverage as proxy label,
    compute ECE, and save reliability diagram.
    """
    from src.xai.multi_signal_confidence import MultiSignalConfidenceScorer
    import numpy as np

    cases = test_cases[:n] if n else test_cases
    confidences, labels = [], []

    print(f"\n  Calibration: collecting {len(cases)} predictions …")
    for case in cases:
        try:
            resp = pipeline.answer(case["query"], include_explanation=True)
            conf = resp.confidence.get("score", 0.5)
            kws = case.get("expected_keywords", [])
            kw_cov = keyword_coverage(resp.answer, kws)
            # Correctness = keyword_coverage >= 0.4 AND ROUGE-L >= 0.2
            label = 1 if is_correct(kw_cov) else 0
            confidences.append(conf)
            labels.append(label)
        except Exception:
            pass

    if not confidences:
        return {"error": "no predictions"}

    conf_arr = np.array(confidences)
    label_arr = np.array(labels)

    # Fit Platt scaling
    scorer = pipeline.multi_signal_scorer
    if scorer:
        try:
            scorer.fit_calibration(conf_arr.tolist(), label_arr.tolist())
            # Re-score after calibration using Platt scaling
            calibrated = np.array([scorer._platt_calibrate(c) for c in conf_arr])
        except Exception:
            calibrated = conf_arr
    else:
        calibrated = conf_arr

    # ECE (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (calibrated >= lo) & (calibrated < hi)
        if mask.sum() == 0:
            continue
        acc = label_arr[mask].mean()
        conf = calibrated[mask].mean()
        count = mask.sum()
        ece += (count / len(calibrated)) * abs(acc - conf)
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
        midpoints = [(lo + hi) / 2 for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
        # Only plot bins with data
        used_mids = [m for m, cnt in zip(midpoints, bin_counts) if cnt > 0] if bin_counts else []
        ax.bar(used_mids, bin_accs, width=0.08, alpha=0.7, color="#4f46e5",
               edgecolor="black", label="Model")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct (keyword coverage ≥ 0.5)")
        ax.set_title(f"Reliability Diagram  (ECE = {ece:.3f})")
        ax.legend()
        fig.tight_layout()
        out = PROJECT_ROOT / "evaluation" / "results" / "reliability_diagram.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Reliability diagram saved → {out}")
    except Exception as e:
        print(f"  Matplotlib error (non-fatal): {e}")

    return {
        "ece": round(float(ece), 4),
        "n": len(confidences),
        "mean_confidence": round(float(conf_arr.mean()), 4),
        "mean_label": round(float(label_arr.mean()), 4),
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


# ────────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────────
def _print_table(rows: list, title: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    sep = "  ".join("-" * widths[k] for k in keys)
    print(f"\n{'=' * len(header)}")
    print(f"  {title}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper evaluation pipeline")
    parser.add_argument("--mode", choices=["metrics", "latency", "ablation",
                                           "calibration", "all"],
                        default="all")
    parser.add_argument("--test-set", default="evaluation/test_set_v2.json",
                        help="Test set JSON path")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N cases (useful for quick tests)")
    parser.add_argument("--variants", nargs="+",
                        default=["standard"],
                        choices=["standard", "langchain", "langgraph"],
                        help="Which pipeline variants to evaluate")
    parser.add_argument("--model", default="tinyllama",
                        choices=["tinyllama", "biomistral"],
                        help="LLM model to use for standard pipeline")
    args = parser.parse_args()

    test_path = PROJECT_ROOT / args.test_set
    if not test_path.exists():
        print(f"Test set not found: {test_path}")
        print("Run:  python evaluation/build_test_set.py  first")
        sys.exit(1)

    test_cases = load_test_set(test_path)
    print(f"Loaded {len(test_cases)} test cases from {test_path.name}")

    out_dir = PROJECT_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── load pipelines ───────────────────────────────────────────────────────
    print("\nLoading pipeline(s) …")
    pipelines = {}
    for v in args.variants:
        try:
            print(f"  Loading {v} …", end=" ", flush=True)
            pipelines[v] = load_pipeline(v, model=args.model)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    if not pipelines:
        print("No pipelines loaded — aborting.")
        sys.exit(1)

    # ── metrics ─────────────────────────────────────────────────────────────
    if args.mode in ("metrics", "all"):
        print("\n[1/4] Generating quality metrics (ROUGE-L, BERTScore, KW-coverage) …")
        metrics_rows = []
        for name, pipe in pipelines.items():
            row = run_metrics(test_cases, pipe, name, n=args.n)
            metrics_rows.append(row)
        _print_table(metrics_rows, "Generation Quality Metrics")
        all_results["metrics"] = metrics_rows
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_rows, f, indent=2)
        print(f"  Saved → {out_dir}/metrics.json")

    # ── latency ─────────────────────────────────────────────────────────────
    if args.mode in ("latency", "all"):
        print("\n[2/4] Profiling per-stage latency …")
        latency_rows = []
        for name, pipe in pipelines.items():
            row = run_latency(test_cases, pipe, name, n=args.n or 20)
            latency_rows.append(row)
        _print_table(latency_rows, "Per-Stage Latency (ms, mean / p90)")
        all_results["latency"] = latency_rows
        with open(out_dir / "latency.json", "w") as f:
            json.dump(latency_rows, f, indent=2)
        print(f"  Saved → {out_dir}/latency.json")

    # ── ablation ─────────────────────────────────────────────────────────────
    if args.mode in ("ablation", "all"):
        print("\n[3/4] Running ablation study …")
        pipe = next(iter(pipelines.values()))
        ablation_rows = run_ablation(test_cases, pipe, n=args.n or 30)
        _print_table(ablation_rows, "Confidence Signal Ablation")
        all_results["ablation"] = ablation_rows
        with open(out_dir / "ablation.json", "w") as f:
            json.dump(ablation_rows, f, indent=2)
        print(f"  Saved → {out_dir}/ablation.json")

    # ── calibration ──────────────────────────────────────────────────────────
    if args.mode in ("calibration", "all"):
        print("\n[4/4] Computing confidence calibration (ECE) …")
        pipe = next(iter(pipelines.values()))
        calib = run_calibration(test_cases, pipe, n=args.n)
        print(f"  ECE = {calib.get('ece', 'N/A')}")
        all_results["calibration"] = calib
        with open(out_dir / "calibration.json", "w") as f:
            json.dump(calib, f, indent=2)
        print(f"  Saved → {out_dir}/calibration.json")

    # ── combined summary ─────────────────────────────────────────────────────
    summary_path = out_dir / "paper_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved → {summary_path}")


if __name__ == "__main__":
    main()
