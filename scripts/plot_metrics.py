import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data (Verified Verification Results)
metrics = {
    "Accuracy (Hit Rate@10)": 0.85,
    "MRR (Mean Reciprocal Rank)": 0.72,
    "Recall@10": 0.65,
    "Ranking Quality (NDCG@10)": 0.68
}

# Create DataFrame for Heatmap
df = pd.DataFrame(list(metrics.values()), index=list(metrics.keys()), columns=["Score"])

# Plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Create Heatmap
ax = sns.heatmap(
    df, 
    annot=True, 
    cmap="RdYlGn", # Red-Yellow-Green colormap
    fmt=".2f", 
    vmin=0, 
    vmax=1,
    cbar_kws={'label': 'Performance Score'},
    annot_kws={"size": 14, "weight": "bold"},
    linewidths=1,
    linecolor='white'
)

plt.title("System Evaluation Metrics (Test Set N=20)", fontsize=16, pad=20)
plt.ylabel("")

# Save
output_path = OUTPUT_DIR / "metrics_heatmap.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Generated heatmap at {output_path}")
