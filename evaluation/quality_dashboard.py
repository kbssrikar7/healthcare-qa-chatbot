"""
Offline quality dashboard — reads evaluation result JSON files and produces
a markdown report with per-category breakdowns and run-over-run comparison.

Usage:
    python evaluation/quality_dashboard.py
    python evaluation/quality_dashboard.py --results-dir evaluation/results --out report.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _fmt(val, pct=False):
    if val is None:
        return "—"
    if pct:
        return f"{val * 100:.1f}%"
    return f"{val:.4f}"


def build_report(results_dir="evaluation/results", out_path=None):
    results_dir = Path(results_dir)
    lines = [
        f"# Healthcare QA — Quality Dashboard",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
    ]

    # ── Core metrics ──────────────────────────────────────────────────────
    metrics = _load_json(results_dir / "metrics.json")
    if metrics:
        lines += ["## Core Evaluation Metrics\n"]
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    # ── Model comparison (paper eval summary) ─────────────────────────────
    paper = _load_json(results_dir / "paper_eval_summary.json")
    if paper:
        lines += ["## Model Comparison\n"]
        models = paper.get("models", {})
        if models:
            metric_keys = sorted({k for m in models.values() for k in m if isinstance(m[k], (int, float))})
            header = "| Model | " + " | ".join(metric_keys) + " |"
            sep = "|---|" + "---|" * len(metric_keys)
            lines += [header, sep]
            for model_name, m in models.items():
                row = f"| {model_name} | " + " | ".join(_fmt(m.get(k)) for k in metric_keys) + " |"
                lines.append(row)
            lines.append("")

    # ── KB A/B ────────────────────────────────────────────────────────────
    ab = _load_json(results_dir / "kb_v1_vs_v2.json")
    if ab:
        lines += ["## Knowledge Base A/B (v1 vs v2)\n"]
        decision = ab.get("decision", "?")
        delta = ab.get("delta", 0)
        lines += [
            f"- **Decision:** {decision}",
            f"- **Keyword coverage delta:** +{delta:.4f}",
            f"- v1 kw: {_fmt(ab.get('kw_v1'))} | v2 kw: {_fmt(ab.get('kw_v2'))}",
            "",
        ]

    # ── Calibration ───────────────────────────────────────────────────────
    calib = _load_json(results_dir / "calibration.json")
    if calib:
        lines += ["## Confidence Calibration\n"]
        lines += [
            f"- **ECE:** {_fmt(calib.get('ece'))}",
            f"- **n:** {calib.get('n', '?')}",
            f"- **Mean confidence:** {_fmt(calib.get('mean_confidence'))}",
            f"- **Mean accuracy:** {_fmt(calib.get('mean_label'))}",
        ]
        if "platt_a" in calib:
            lines.append(f"- **Platt a/b:** {calib['platt_a']:.3f} / {calib['platt_b']:.3f}")
        lines.append("")

    # ── Retrieval eval (if present) ───────────────────────────────────────
    ret = _load_json(results_dir / "retrieval_eval.json")
    if ret:
        lines += ["## Retrieval Quality\n"]
        s = ret.get("summary", {})
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in s.items():
            if k != "n":
                lines.append(f"| {k} | {_fmt(v)} |")
        lines.append(f"| questions evaluated | {s.get('n', '?')} |")
        lines.append("")

    # ── Ablation ──────────────────────────────────────────────────────────
    ablation = _load_json(results_dir / "ablation.json")
    if ablation:
        lines += ["## Ablation Study\n"]
        rows = ablation if isinstance(ablation, list) else ablation.get("results", [])
        if rows:
            keys = [k for k in rows[0] if k != "variant"]
            lines += [
                "| Variant | " + " | ".join(keys) + " |",
                "|---|" + "---|" * len(keys),
            ]
            for r in rows:
                row = f"| {r.get('variant','?')} | " + " | ".join(_fmt(r.get(k)) for k in keys) + " |"
                lines.append(row)
            lines.append("")

    report = "\n".join(lines)

    if out_path:
        Path(out_path).write_text(report)
        print(f"Dashboard written to {out_path}")
    else:
        print(report)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="evaluation/results")
    parser.add_argument("--out", default=None, help="Output markdown path (default: stdout)")
    args = parser.parse_args()
    build_report(args.results_dir, args.out)
