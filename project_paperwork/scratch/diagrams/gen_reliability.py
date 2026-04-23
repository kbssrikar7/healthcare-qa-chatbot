"""
Generate a clean, publication-quality reliability diagram.
Shows only the Platt-calibrated data (after scaling), single chart, no overlaps.
Data from the evaluation report: n=97, ECE=0.14 using keyword-coverage proxy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------- Actual calibration data from the evaluation ----------
# Confidence bins (mean predicted confidence per bin) and fraction correct
# Source: evaluation/results/ and paper section 5.5
# Bins correspond to confidence intervals, correctness = kw >= 0.4
bin_centers      = [0.40, 0.50, 0.60, 0.80]
fraction_correct = [0.30, 0.56, 1.00, 0.50]
bin_counts       = [57,   25,   7,    2   ]
# Use narrow bars so adjacent bins don't overlap

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(6, 5))

colors = []
for fc, bc in zip(fraction_correct, bin_centers):
    gap = abs(fc - bc)
    colors.append('#e05c5c' if gap > 0.15 else '#4a90d9')

bars = ax.bar(
    bin_centers,
    fraction_correct,
    width=0.060,
    color=colors,
    edgecolor='white',
    linewidth=1.2,
    alpha=0.88,
    label='Model (calibrated)',
    zorder=3
)

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', zorder=4)

# Annotate bars with sample counts
for bar, n in zip(bars, bin_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.025,
        f'n={n}',
        ha='center', va='bottom',
        fontsize=9, color='#333333'
    )

# Axes and labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.12)
ax.set_xlabel('Mean Predicted Confidence', fontsize=11)
ax.set_ylabel('Fraction Correct  (KW ≥ 0.4)', fontsize=11)
ax.set_title(
    'Reliability Diagram — Platt-Calibrated Confidence\n'
    'ECE = 0.14  (n = 97)',
    fontsize=11, fontweight='bold', pad=10
)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(labelsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
output = '/home/kbs/Documents/final_project/paper/reliability_diagram_calibrated.png'
plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {output}")
