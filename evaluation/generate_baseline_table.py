#!/usr/bin/env python3
"""
Generate LaTeX and Markdown comparison tables from metrics JSON files.

Usage
-----
    python evaluation/generate_baseline_table.py
    python evaluation/generate_baseline_table.py --format markdown
    python evaluation/generate_baseline_table.py --format latex --ci
    python evaluation/generate_baseline_table.py --output evaluation/paper_table.tex

Outputs
-------
- LaTeX \\booktabs table (default) or Markdown table
- Reads all metrics_*.json from evaluation/results/
- Ordered: No-RAG → Dense-only → No-XAI → QLoRA → TinyLlama (full) → BioMistral
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results"

# Display order + human-readable labels
VARIANT_ORDER = [
    ("no_rag",              "No RAG (LLM only)"),
    ("dense_only",          "Dense-only (no BM25)"),
    ("no_xai",              "Full RAG, no XAI"),
    ("qlora_tinyllama",     "Full RAG + QLoRA"),
    ("standard_tinyllama",  "Full RAG + TinyLlama \\textbf{(ours)}"),
    ("standard_biomistral", "Full RAG + BioMistral-7B"),
]

METRICS = [
    ("keyword_coverage_mean", "Kw.Cov.↑",  ".3f"),
    ("rougeL_mean",           "ROUGE-L↑",  ".3f"),
    ("bertscore_f1_mean",     "BERTScore↑", ".3f"),
    ("latency_mean_ms",       "Latency (s)↑", None),  # custom formatter
]

# CI keys follow pattern metric_key + "_ci_lo" / "_ci_hi"
CI_SUFFIXES = ("_ci_lo", "_ci_hi")


def _load_all() -> dict:
    """Return dict {variant_key: metrics_dict}."""
    data = {}
    for path in RESULTS_DIR.glob("metrics_*.json"):
        try:
            d = json.loads(path.read_text())
            variant = d.get("variant", path.stem.replace("metrics_", ""))
            data[variant] = d
        except Exception as e:
            print(f"  WARNING: Could not parse {path.name}: {e}", file=sys.stderr)
    return data


def _fmt(val, fmt: str | None) -> str:
    if val is None:
        return "—"
    if fmt is None:
        # latency: ms → seconds
        return f"{float(val)/1000:.1f}"
    return format(float(val), fmt)


def _ci_str(d: dict, key: str, fmt: str | None) -> str:
    lo = d.get(key + "_ci_lo")
    hi = d.get(key + "_ci_hi")
    if lo is None or hi is None:
        return ""
    lo_s = _fmt(lo, fmt)
    hi_s = _fmt(hi, fmt)
    return f"[{lo_s},{hi_s}]"


def build_latex(data: dict, with_ci: bool = False) -> str:
    n_cols = len(METRICS) + 1  # +1 for variant name
    col_spec = "l" + "r" * len(METRICS)

    header_row = " & ".join(
        ["\\textbf{System}"] + [f"\\textbf{{{lbl}}}" for _, lbl, _ in METRICS]
    )

    rows = []
    for variant_key, label in VARIANT_ORDER:
        d = data.get(variant_key)
        if d is None:
            continue

        cells = [label]
        for metric_key, _, fmt in METRICS:
            val = d.get(metric_key)
            cell = _fmt(val, fmt)
            if with_ci:
                ci = _ci_str(d, metric_key, fmt)
                if ci:
                    cell = f"{cell} {{\\tiny {ci}}}"
            cells.append(cell)

        # Bold the best value in each metric column (mark TinyLlama row)
        if "ours" in label:
            # Row-level bold for the recommended system
            row_str = " & ".join(f"\\textbf{{{c}}}" if i > 0 else c for i, c in enumerate(cells))
        else:
            row_str = " & ".join(cells)

        rows.append(f"    {row_str} \\\\")

    # Add a mid-rule before BioMistral (last entry)
    if len(rows) >= 2:
        rows.insert(-1, "    \\midrule")

    n_sample = data.get("standard_tinyllama", {}).get("n", "97")
    caption = (
        f"Comparison of system variants on {n_sample}-question evaluation set "
        f"(mean ± 95\\% bootstrap CI). "
        f"↑ = higher is better. Latency measured on CPU (Intel i7, no GPU)."
    )

    table = "\n".join([
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:system_comparison}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header_row} \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    return table


def build_markdown(data: dict, with_ci: bool = False) -> str:
    headers = ["System"] + [lbl for _, lbl, _ in METRICS]
    lines   = ["| " + " | ".join(headers) + " |",
               "| " + " | ".join(["---"] * len(headers)) + " |"]

    for variant_key, label in VARIANT_ORDER:
        d = data.get(variant_key)
        if d is None:
            continue
        cells = [label]
        for metric_key, _, fmt in METRICS:
            val = d.get(metric_key)
            cell = _fmt(val, fmt)
            if with_ci:
                ci = _ci_str(d, metric_key, fmt)
                if ci:
                    cell = f"{cell} {ci}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline comparison table")
    parser.add_argument("--format", choices=["latex", "markdown"], default="latex")
    parser.add_argument("--ci",     action="store_true",
                        help="Include 95%% bootstrap confidence intervals")
    parser.add_argument("--output", default=None,
                        help="Write to file (default: stdout)")
    args = parser.parse_args()

    data = _load_all()
    if not data:
        print(f"ERROR: No metrics JSON files found in {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} variant(s): {sorted(data.keys())}", file=sys.stderr)

    if args.format == "latex":
        output = build_latex(data, with_ci=args.ci)
    else:
        output = build_markdown(data, with_ci=args.ci)

    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
