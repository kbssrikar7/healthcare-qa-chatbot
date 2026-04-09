#!/usr/bin/env python3
"""
Generate paper-quality figures from evaluation results.

Reads JSON results from evaluation/results/ and produces publication-ready
matplotlib figures saved as both PNG (300 DPI) and PDF (vector).

Usage:
    python evaluation/generate_paper_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "primary": "#4f46e5",
    "secondary": "#0ea5e9",
    "accent": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "gray": "#6b7280",
}


def _save(fig, name: str):
    """Save figure as PNG and PDF."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


# ── Fig 2: Metric comparison bar chart ──────────────────────────────────────
def fig_metric_comparison():
    """Bar chart comparing models on ROUGE-L, BERTScore, keyword coverage."""
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        print("  Skipping metric comparison (metrics.json not found)")
        return

    with open(metrics_path) as f:
        rows = json.load(f)

    # Handle both list-of-dicts and single dict
    if isinstance(rows, dict):
        rows = [rows]

    labels = [r.get("variant", r.get("name", "?")) for r in rows]
    kw = [r.get("keyword_coverage_mean", 0) for r in rows]
    rl = [r.get("rougeL_mean", 0) for r in rows]
    bs = [r.get("bertscore_f1_mean", 0) for r in rows]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, kw, width, label="Keyword Coverage", color=COLORS["primary"])
    ax.bar(x, rl, width, label="ROUGE-L", color=COLORS["secondary"])
    ax.bar(x + width, bs, width, label="BERTScore F1", color=COLORS["accent"])

    # Add CI error bars if available
    for i, r in enumerate(rows):
        if "keyword_coverage_ci_95" in r or "keyword_coverage_ci" in r:
            ci = r.get("keyword_coverage_ci_95", r.get("keyword_coverage_ci"))
            ax.errorbar(
                x[i] - width,
                kw[i],
                yerr=[[kw[i] - ci[0]], [ci[1] - kw[i]]],
                fmt="none",
                color="black",
                capsize=3,
            )
        if "rougeL_ci_95" in r or "rougeL_ci" in r:
            ci = r.get("rougeL_ci_95", r.get("rougeL_ci"))
            ax.errorbar(
                x[i],
                rl[i],
                yerr=[[rl[i] - ci[0]], [ci[1] - rl[i]]],
                fmt="none",
                color="black",
                capsize=3,
            )
        if "bertscore_ci_95" in r or "bertscore_f1_ci" in r:
            ci = r.get("bertscore_ci_95", r.get("bertscore_f1_ci"))
            ax.errorbar(
                x[i] + width,
                bs[i],
                yerr=[[bs[i] - ci[0]], [ci[1] - bs[i]]],
                fmt="none",
                color="black",
                capsize=3,
            )

    ax.set_xlabel("Pipeline Variant")
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality Metrics by Pipeline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    _save(fig, "fig2_metric_comparison")


# ── Fig 3: Reliability diagram ──────────────────────────────────────────────
def fig_reliability_diagram():
    """Side-by-side reliability diagrams (raw vs calibrated)."""
    cal_path = RESULTS_DIR / "calibration.json"
    if not cal_path.exists():
        print("  Skipping reliability diagram (calibration.json not found)")
        return

    with open(cal_path) as f:
        cal = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, suffix, title in [
        (axes[0], "_raw", "Before Platt Scaling"),
        (axes[1], "_calibrated", "After Platt Scaling"),
    ]:
        bcs = cal.get(f"bin_confidences{suffix}", cal.get("bin_confidences", []))
        bas = cal.get(f"bin_accuracies{suffix}", cal.get("bin_accuracies", []))
        bns = cal.get(f"bin_counts{suffix}", cal.get("bin_counts", []))
        ece = cal.get(f"ece{suffix}", cal.get("ece", 0))

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
        if bcs:
            ax.bar(
                bcs,
                bas,
                width=0.08,
                alpha=0.75,
                color=COLORS["primary"],
                edgecolor="black",
                label="Model",
            )
            for x, y, cnt in zip(bcs, bas, bns):
                ax.text(x, y + 0.02, str(cnt), ha="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction Correct")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(fontsize=9)

    fig.suptitle("Confidence Calibration — Healthcare QA Chatbot", fontsize=13, y=1.02)
    _save(fig, "fig3_reliability_diagram")


# ── Fig 4: Ablation study ───────────────────────────────────────────────────
def fig_ablation():
    """Grouped bar chart showing ablation of each confidence signal."""
    abl_path = RESULTS_DIR / "ablation.json"
    if not abl_path.exists():
        print("  Skipping ablation (ablation.json not found)")
        return

    with open(abl_path) as f:
        rows = json.load(f)

    if not rows or "error" in rows[0]:
        print("  Skipping ablation (no valid data)")
        return

    names = [r.get("name", r.get("variant", "?")) for r in rows]
    confs = [r.get("mean_confidence", 0) for r in rows]
    kws = [r.get("mean_kw_coverage", 0) for r in rows]
    conf_errs = []
    kw_errs = []
    for row, conf, kw in zip(rows, confs, kws):
        conf_ci = row.get("confidence_ci_95", [conf, conf])
        kw_ci = row.get("kw_coverage_ci_95", [kw, kw])
        conf_errs.append([(conf - conf_ci[0]), (conf_ci[1] - conf)])
        kw_errs.append([(kw - kw_ci[0]), (kw_ci[1] - kw)])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        confs,
        width,
        label="Mean Confidence",
        color=COLORS["primary"],
        yerr=np.array(conf_errs).T,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        kws,
        width,
        label="Keyword Coverage",
        color=COLORS["accent"],
        yerr=np.array(kw_errs).T,
        capsize=3,
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Confidence Signal Ablation Study")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save(fig, "fig4_ablation")


# ── Fig 5: Latency breakdown ────────────────────────────────────────────────
def fig_latency():
    """Stacked bar chart of per-stage latency."""
    lat_path = RESULTS_DIR / "latency.json"
    if not lat_path.exists():
        print("  Skipping latency (latency.json not found)")
        return

    with open(lat_path) as f:
        rows = json.load(f)

    if isinstance(rows, dict):
        rows = [rows]

    # Gather all stage keys
    stage_keys = set()
    for r in rows:
        for k in r:
            if k.endswith("_mean") and k != "total_ms_mean":
                stage_keys.add(k.replace("_mean", ""))
    stage_keys = sorted(stage_keys)

    if not stage_keys:
        # Fallback: just show total
        variants = [r.get("variant", "?") for r in rows]
        totals = [r.get("total_ms_mean", 0) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(variants, totals, color=COLORS["primary"])
        ax.set_xlabel("Latency (ms)")
        ax.set_title("End-to-End Latency")
        _save(fig, "fig5_latency")
        return

    variants = [r.get("variant", "?") for r in rows]
    colors = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(variants))
    for i, stage in enumerate(stage_keys):
        vals = [r.get(f"{stage}_mean", 0) for r in rows]
        ax.barh(variants, vals, left=bottom, label=stage, color=colors[i % len(colors)])
        bottom += np.array(vals)

    ax.set_xlabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Breakdown")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "fig5_latency")


# ── Fig 6: Confidence distribution ──────────────────────────────────────────
def fig_confidence_distribution():
    """Histogram of confidence scores from trajectory data."""
    traj_path = PROJECT_ROOT / "data" / "feedback" / "response_trajectories.jsonl"
    if not traj_path.exists():
        print("  Skipping confidence distribution (no trajectory data)")
        return

    confs = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                c = row.get("outcome", {}).get("confidence_score")
                if c is not None:
                    confs.append(float(c))

    if not confs:
        print("  No confidence scores in trajectories")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confs, bins=20, color=COLORS["primary"], edgecolor="black", alpha=0.8)
    ax.axvline(
        np.mean(confs),
        color=COLORS["danger"],
        linestyle="--",
        label=f"Mean = {np.mean(confs):.3f}",
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of AI Confidence Scores")
    ax.legend()
    _save(fig, "fig6_confidence_distribution")


def main():
    print("Generating paper-quality figures...")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Output dir:  {FIGURES_DIR}")

    fig_metric_comparison()
    fig_reliability_diagram()
    fig_ablation()
    fig_latency()
    fig_confidence_distribution()

    print("\nDone. Figures saved to evaluation/results/figures/")


if __name__ == "__main__":
    main()
