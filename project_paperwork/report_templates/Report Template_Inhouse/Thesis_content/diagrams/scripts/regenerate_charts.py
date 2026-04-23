"""
Regenerate all 4 evaluation charts with correct data from JSON files.
Produces publication-quality figures at 300 dpi with large, readable fonts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os

IMG = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
})

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Model Comparison (CORRECT VALUES from JSON)
# ══════════════════════════════════════════════════════════════════════════════
models = ["TinyLlama 1.1B\n(base)", "TinyLlama 1.1B\n+ QLoRA", "BioMistral 7B\n(GGUF Q4)"]
kw     = [0.3845, 0.4634, 0.3088]
rouge  = [0.2076, 0.2374, 0.2988]
bert   = [0.1398, 0.1790, 0.2676]

kw_ci  = [(0.3197, 0.4646), (0.3910, 0.5309), (0.2316, 0.3887)]
rl_ci  = [(0.1695, 0.2520), (0.1961, 0.2838), (0.2314, 0.3725)]
bs_ci  = [(0.0844, 0.1997), (0.1225, 0.2348), (0.1948, 0.3412)]

x    = np.arange(len(models))
w    = 0.24
fig, ax = plt.subplots(figsize=(13, 7))

bars_kw   = ax.bar(x - w,     kw,    w, label="KW Coverage",  color="#3498DB", zorder=3)
bars_rl   = ax.bar(x,         rouge, w, label="ROUGE-L",       color="#2ECC71", zorder=3)
bars_bert = ax.bar(x + w,     bert,  w, label="BERTScore F1",  color="#E67E22", zorder=3)

# 95% CI error bars
def ci_err(vals, cis):
    lo = [v - c[0] for v, c in zip(vals, cis)]
    hi = [c[1] - v for v, c in zip(vals, cis)]
    return [lo, hi]

ax.errorbar(x - w, kw,    yerr=ci_err(kw,   kw_ci), fmt="none", color="#1A252F", capsize=5, lw=1.8, zorder=4)
ax.errorbar(x,     rouge, yerr=ci_err(rouge, rl_ci), fmt="none", color="#1A252F", capsize=5, lw=1.8, zorder=4)
ax.errorbar(x + w, bert,  yerr=ci_err(bert,  bs_ci), fmt="none", color="#1A252F", capsize=5, lw=1.8, zorder=4)

# Value labels on bars
for bars, vals in [(bars_kw, kw), (bars_rl, rouge), (bars_bert, bert)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("Model Comparison: Answer Quality Metrics (n = 97)", fontsize=16, fontweight="bold", pad=14)
ax.set_ylim(0, 0.65)
ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{IMG}/fig1_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig1_model_comparison.png written")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Reliability Diagram (single panel, actual calibration data)
# ══════════════════════════════════════════════════════════════════════════════
bin_conf = [0.3824, 0.4194, 0.5277, 0.6330, 0.7536, 0.8342]
bin_acc  = [0.2982, 0.5600, 0.7500, 1.0000, 0.0000, 0.5000]
bin_cnt  = [57, 25, 4, 7, 2, 2]
ece = 0.1438

fig, ax = plt.subplots(figsize=(9, 7))

bar_colors = ["#E74C3C" if abs(a - c) > 0.15 else "#3498DB"
              for a, c in zip(bin_acc, bin_conf)]
bars = ax.bar(bin_conf, bin_acc, width=0.07, color=bar_colors,
              alpha=0.85, label="Model", edgecolor="white", linewidth=0.8, zorder=3)
ax.plot([0, 1], [0, 1], "k--", lw=1.8, label="Perfect calibration", zorder=4)

# Count labels on bars
for bar, cnt, conf, acc in zip(bars, bin_cnt, bin_conf, bin_acc):
    ax.text(bar.get_x() + bar.get_width()/2, max(acc + 0.03, 0.05),
            f"n={cnt}", ha="center", va="bottom", fontsize=11, color="#2C3E50")

ax.set_xlabel("Mean Predicted Confidence", fontsize=14)
ax.set_ylabel("Fraction Correct", fontsize=14)
ax.set_title(f"Confidence Calibration — Healthcare QA Chatbot\nECE = {ece:.4f}  (n = 97)",
             fontsize=15, fontweight="bold", pad=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ECE annotation removed to prevent overlap with the legend. ECE is already in title.

ax.legend(loc="upper left", framealpha=0.9, fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{IMG}/fig3_reliability_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig3_reliability_diagram.png written")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Ablation Study (clean labels, actual data)
# ══════════════════════════════════════════════════════════════════════════════
ablation_labels = [
    "All signals\n(full system)",
    "Ablate\nRetrieval",
    "Ablate\nGeneration",
    "Ablate\nConsistency",
    "Ablate\nSource Agr.",
    "Ablate\nEntity Cov.",
]
kw_abl   = [0.3881, 0.4254, 0.4304, 0.3794, 0.4493, 0.3811]
conf_abl = [0.0161, 0.0070, 0.0036, 0.0284, 0.1743, 0.0169]

x   = np.arange(len(ablation_labels))
w   = 0.30
fig, ax = plt.subplots(figsize=(14, 7))

bars_kw   = ax.bar(x - w/2, kw_abl,   w, label="Keyword Coverage",  color="#27AE60", zorder=3)
bars_conf = ax.bar(x + w/2, conf_abl, w, label="Mean Confidence",   color="#2980B9", zorder=3)

for bar, v in zip(bars_kw, kw_abl):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1E8449")
for bar, v in zip(bars_conf, conf_abl):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1A5276")

ax.set_xticks(x)
ax.set_xticklabels(ablation_labels, fontsize=12.5)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("Confidence Signal Ablation Study (n = 97)", fontsize=16, fontweight="bold", pad=14)
ax.set_ylim(0, 0.60)
ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)

# Highlight full-system bar
ax.axvline(x=0, color="#E74C3C", lw=1.5, ls=":", alpha=0.6, zorder=2)
ax.text(0, 0.57, "baseline", ha="center", fontsize=11, color="#E74C3C", fontstyle="italic")

ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{IMG}/fig4_ablation_eval.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig4_ablation_eval.png written")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Per-Stage Latency Breakdown (warm-path, TinyLlama)
# Single panel horizontal bar chart — no pie (tiny slices cause label overlap)
# ══════════════════════════════════════════════════════════════════════════════
stages  = ["Query Enhancement", "Retrieval\n(BM25 + Dense + RRF)",
           "Corrective RAG", "LLM Generation\n(TinyLlama 1.1B)",
           "Hallucination Check\n(DeBERTa NLI)", "Confidence Scoring\n(5 signals + Platt)"]
warm_ms = [0.02, 42624, 1218, 69868, 1736, 5]
colours = ["#8E44AD", "#2980B9", "#148F77", "#E74C3C", "#E67E22", "#27AE60"]

total_warm = sum(warm_ms)
pct = [v / total_warm * 100 for v in warm_ms]

fig, ax = plt.subplots(figsize=(14, 7))

y    = np.arange(len(stages))
bars = ax.barh(y, warm_ms, color=colours, edgecolor="white",
               linewidth=0.8, height=0.62, zorder=3)

for bar, v, p in zip(bars, warm_ms, pct):
    if v > 500:
        # place label inside bar for large bars
        ax.text(v * 0.50, bar.get_y() + bar.get_height() / 2,
                f"{v/1000:.1f} s  ({p:.1f}%)",
                va="center", ha="center", fontsize=12,
                fontweight="bold", color="white")
    else:
        ax.text(bar.get_width() + 400,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f} ms  ({p:.2f}%)" if v > 1 else f"{v:.2f} ms  (<0.01%)",
                va="center", fontsize=11, color="#2C3E50")

ax.set_yticks(y)
ax.set_yticklabels(stages, fontsize=13)
ax.set_xlabel("Mean Latency (ms) — warm path, TinyLlama 1.1B", fontsize=13)
ax.set_title(
    f"Per-Stage Latency Breakdown  (warm total = {total_warm/1000:.1f} s  |  cold P90 = 276.8 s)",
    fontsize=14, fontweight="bold", pad=12)
ax.set_xlim(0, max(warm_ms) * 1.30)
ax.xaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate the two dominant bars
for idx, (v, s) in enumerate(zip(warm_ms, stages)):
    pass  # labels already inside bars

plt.tight_layout(pad=1.0)
plt.savefig(f"{IMG}/fig5_latency.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig5_latency.png written")

print("\nAll charts regenerated successfully.")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Metric Comparison across backends (chapter 6, same data as fig1)
# ══════════════════════════════════════════════════════════════════════════════
models2 = ["TinyLlama 1.1B\n(base)", "TinyLlama 1.1B\n+ QLoRA", "BioMistral 7B\n(GGUF Q4)"]
metrics = {
    "KW Coverage":  ([0.3845, 0.4634, 0.3088], "#3498DB"),
    "ROUGE-L":      ([0.2076, 0.2374, 0.2988], "#2ECC71"),
    "BERTScore F1": ([0.1398, 0.1790, 0.2676], "#E67E22"),
}
x2 = np.arange(len(models2))
w2 = 0.24
fig, ax = plt.subplots(figsize=(13, 7))
offset = -w2
for metric, (vals, color) in metrics.items():
    bars = ax.bar(x2 + offset, vals, w2, label=metric, color=color, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    offset += w2
ax.set_xticks(x2)
ax.set_xticklabels(models2, fontsize=13)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("Generation Quality Metrics by Model Backend (n = 97)", fontsize=16, fontweight="bold", pad=14)
ax.set_ylim(0, 0.62)
ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{IMG}/fig2_metric_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig2_metric_comparison.png written")
