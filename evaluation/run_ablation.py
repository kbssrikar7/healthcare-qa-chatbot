#!/usr/bin/env python3
"""
Confidence Signal Ablation Study.

Runs N questions through the pipeline ONCE, captures the individual
5-signal breakdown, then re-combines with each ablation variant's weights
offline (no extra LLM calls needed).

Produces:
  • ablation.json   — mean confidence + ECE per variant
  • ablation_plot.png — bar chart for the paper

Usage:
    python evaluation/run_ablation.py --n 15
    python evaluation/run_ablation.py --n 15 --out-dir evaluation/results
"""
import os, sys, json, argparse, statistics
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SIGNALS = ["retrieval", "generation", "consistency", "source_agreement", "entity_coverage"]

ABLATION_VARIANTS = {
    "all_signals":        {s: 1.0 for s in SIGNALS},
    "ablate_retrieval":   {s: (0.0 if s == "retrieval"       else 1.0) for s in SIGNALS},
    "ablate_generation":  {s: (0.0 if s == "generation"      else 1.0) for s in SIGNALS},
    "ablate_consistency": {s: (0.0 if s == "consistency"     else 1.0) for s in SIGNALS},
    "ablate_source_agr":  {s: (0.0 if s == "source_agreement" else 1.0) for s in SIGNALS},
    "ablate_entity_cov":  {s: (0.0 if s == "entity_coverage" else 1.0) for s in SIGNALS},
}

SIGNAL_MAP = {
    "retrieval":       "retrieval_confidence",
    "generation":      "generation_confidence",
    "consistency":     "consistency_score",
    "source_agreement":"source_agreement",
    "entity_coverage": "medical_entity_coverage",
}


def recompute_conf(breakdown: dict, weights: dict) -> float:
    import numpy as np
    total = sum(weights.values()) or 1.0
    w = {k: v / total for k, v in weights.items()}
    raw = sum(w.get(sig, 0) * breakdown.get(SIGNAL_MAP[sig], 0.0) for sig in SIGNALS)
    return float(np.clip(raw, 0.0, 1.0))


def compute_ece(confidences: list, labels: list, n_bins: int = 10) -> float:
    import numpy as np
    if not confidences:
        return float("nan")
    ca = np.array(confidences)
    la = np.array(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ca >= lo) & (ca < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(ca)) * abs(la[mask].mean() - ca[mask].mean())
    return float(ece)


def keyword_coverage(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15,
                        help="Number of questions to run (15 ≈ 25 min on CPU)")
    parser.add_argument("--test-set", default="evaluation/test_set_v2.json")
    parser.add_argument("--out-dir", default="evaluation/results")
    args = parser.parse_args()

    # Load test cases
    ts_path = PROJECT_ROOT / args.test_set
    with open(ts_path) as f:
        all_cases = json.load(f)["test_cases"]
    cases = all_cases[:args.n]
    print(f"Ablation study: {len(cases)} questions, {len(ABLATION_VARIANTS)} variants")
    print("Loading pipeline (this takes ~280s cold) …")

    from api.main import get_pipeline
    pipeline = get_pipeline("tinyllama")
    print("Pipeline loaded.\n")

    # Step 1: Run each question ONCE, collect signal breakdown + labels
    records = []  # {breakdown, answer, kw_coverage, nli_label}
    for i, case in enumerate(cases, 1):
        try:
            resp = pipeline.answer(case["query"], include_explanation=True)
            answer = resp.answer if hasattr(resp, "answer") else str(resp)
            bd = getattr(resp, "confidence_breakdown", None) or {}

            # Correctness labels
            kws = case.get("expected_keywords", [])
            kw_cov = keyword_coverage(answer, kws)
            kw_label = 1 if kw_cov >= 0.5 else 0

            records.append({
                "query": case["query"],
                "answer": answer,
                "breakdown": bd,
                "kw_coverage": kw_cov,
                "kw_label": kw_label,
            })
            print(f"  [{i}/{len(cases)}] conf_raw={bd.get('retrieval_confidence', '?'):.2f} "
                  f"kw_cov={kw_cov:.2f}  '{case['query'][:50]}'")
        except Exception as e:
            print(f"  [{i}/{len(cases)}] ERROR: {e}")

    if not records:
        print("No records collected — aborting.")
        sys.exit(1)

    print(f"\nCollected {len(records)} records. Re-weighting offline …\n")

    # Step 2: Re-combine signals for each ablation variant
    ablation_results = []
    for variant_name, weights in ABLATION_VARIANTS.items():
        confs = [recompute_conf(r["breakdown"], weights) for r in records]
        labels = [r["kw_label"] for r in records]
        kw_covs = [r["kw_coverage"] for r in records]
        ece = compute_ece(confs, labels)
        ablation_results.append({
            "variant": variant_name,
            "n": len(records),
            "mean_confidence": round(statistics.mean(confs), 4),
            "mean_kw_coverage": round(statistics.mean(kw_covs), 4),
            "ece": round(ece, 4),
        })

    # Print table
    print(f"{'Variant':<25} {'Mean Conf':>10} {'Mean KW-Cov':>12} {'ECE':>8}")
    print("-" * 58)
    for row in ablation_results:
        print(f"{row['variant']:<25} {row['mean_confidence']:>10.4f} "
              f"{row['mean_kw_coverage']:>12.4f} {row['ece']:>8.4f}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved → {out_dir}/ablation.json")

    # Step 3: Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels_x = [r["variant"].replace("ablate_", "-").replace("_", " ") for r in ablation_results]
        x = np.arange(len(labels_x))
        confs_vals = [r["mean_confidence"] for r in ablation_results]
        ece_vals = [r["ece"] for r in ablation_results]
        kw_vals = [r["mean_kw_coverage"] for r in ablation_results]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for ax, vals, title, color in zip(
            axes,
            [confs_vals, ece_vals, kw_vals],
            ["Mean Confidence", "ECE (lower = better)", "Mean KW Coverage"],
            ["#4f46e5", "#e53e3e", "#38a169"],
        ):
            bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_x, rotation=30, ha="right", fontsize=9)
            ax.set_title(title, fontsize=12)
            ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
            # Highlight baseline (first bar)
            bars[0].set_edgecolor("gold")
            bars[0].set_linewidth(2)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(f"Confidence Signal Ablation Study  (n={len(records)} questions)",
                     fontsize=13)
        fig.tight_layout()
        out_path = out_dir / "ablation_plot.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved → {out_path}")
    except Exception as e:
        print(f"Plot error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
