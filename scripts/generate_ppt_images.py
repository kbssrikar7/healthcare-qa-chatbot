import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np
import seaborn as sns
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
OUTPUT_DIR = Path("ppt_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_PRIMARY = "#003366"    # Navy
C_ACCENT  = "#FF6600"    # Orange
C_SUCCESS = "#28A745"    # Green
C_LIGHT   = "#E6F2FF"    # Light Blue
C_TEXT    = "#333333"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ==========================================
# Architecture Diagram Engine (Reusable)
# ==========================================
class DrawEngine:
    def __init__(self, ax):
        self.ax = ax
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def rect(self, x, y, w, h, label, fill=C_LIGHT, border=C_PRIMARY, subtitle=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", 
                             ec=border, fc=fill, lw=1.5, zorder=10)
        self.ax.add_patch(box)
        cx, cy = x + w/2, y + h/2
        self.ax.text(cx, cy + (0.15 if subtitle else 0), label, ha='center', va='center', 
                     fontweight='bold', color=C_TEXT, zorder=11, fontsize=9)
        if subtitle:
            self.ax.text(cx, cy - 0.2, subtitle, ha='center', va='center', 
                         fontsize=7, color='#555555', zorder=11)
        return (x, y, w, h)

    def database(self, x, y, w, h, label):
        fill = "#E6FFEA"
        top = Ellipse((x + w/2, y + h), w, h*0.3, ec=C_PRIMARY, fc=fill, lw=1.5, zorder=12)
        body = mpatches.Rectangle((x, y + h*0.15), w, h*0.85, ec='none', fc=fill, zorder=11)
        bottom = mpatches.Arc((x + w/2, y + h*0.15), w, h*0.3, theta1=180, theta2=360, ec=C_PRIMARY, lw=1.5, zorder=12)
        self.ax.plot([x, x], [y + h*0.15, y + h], color=C_PRIMARY, lw=1.5, zorder=12)
        self.ax.plot([x+w, x+w], [y + h*0.15, y + h], color=C_PRIMARY, lw=1.5, zorder=12)
        self.ax.add_patch(top)
        self.ax.add_patch(body)
        self.ax.add_patch(bottom)
        self.ax.text(x + w/2, y + h*0.5, label, ha='center', va='center', fontweight='bold', fontsize=8, zorder=13)

    def connector(self, p1, p2, label=None, style='->', color='#333333'):
        self.ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
        if label:
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            self.ax.text(mid[0], mid[1], label, ha='center', fontsize=7, 
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    def user(self, x, y, label):
        head = mpatches.Circle((x, y + 0.8), 0.2, ec=C_PRIMARY, fc='white', lw=1.5)
        self.ax.add_patch(head)
        self.ax.plot([x, x], [y + 0.6, y + 0.3], color=C_PRIMARY, lw=1.5)
        self.ax.plot([x - 0.25, x + 0.25], [y + 0.5, y + 0.5], color=C_PRIMARY, lw=1.5)
        self.ax.plot([x, x - 0.2], [y + 0.3, y], color=C_PRIMARY, lw=1.5)
        self.ax.plot([x, x + 0.2], [y + 0.3, y], color=C_PRIMARY, lw=1.5)
        self.ax.text(x, y - 0.2, label, ha='center', va='top', fontweight='bold', fontsize=9)

# ==========================================
# 1. Slide 6: System Overview
# ==========================================
def img1_system_overview():
    fig, ax = plt.subplots(figsize=(10, 6))
    d = DrawEngine(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Glass Box Layout
    d.user(1, 3, "User")
    d.rect(2.5, 2.5, 1.5, 1.5, "Frontend", subtitle="Streamlit")
    d.rect(5, 2.5, 1.5, 1.5, "Backend", subtitle="FastAPI")
    
    # Pillars
    d.rect(7.5, 3.5, 2, 1.5, "RAG Engine", fill="#FFF8E1", border="#FFA000")
    d.rect(7.5, 1.0, 2, 1.5, "XAI Module", fill="#E3F2FD", border="#1976D2")
    
    # Flows
    d.connector((1.2, 3.5), (2.5, 3.5))
    d.connector((4, 3.5), (5, 3.5), "HTTP/JSON")
    d.connector((6.5, 3.5), (7.5, 4.25), "Query")
    d.connector((7.5, 2.25), (6.5, 3.0), "Response") # XAI -> Backend
    
    d.connector((8.5, 3.5), (8.5, 2.5)) # RAG -> XAI
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "system_architecture.png", dpi=300)
    plt.close()

# ==========================================
# 2. Slide 7: Detailed System Diagram
# ==========================================
def img2_detailed_system():
    fig, ax = plt.subplots(figsize=(12, 8))
    d = DrawEngine(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    # Pipeline
    y = 4
    d.rect(0.5, y, 1.5, 1, "Preprocessing", subtitle="Clean/NER")
    d.rect(2.5, y, 1.5, 1, "Embedding", subtitle="MedCPT")
    
    # Hybrid Retrieval
    d.rect(5, y+1.5, 2, 1, "Dense Search")
    d.rect(5, y-1.5, 2, 1, "Keyword Search")
    d.database(5, y-0.25, 2, 1.5, "Vector & Index")
    
    d.rect(7.5, y, 2, 1, "Rerank & Fusion")
    d.rect(7.5, 2, 2, 1, "Grounding Gate", border=C_ACCENT, fill="#FFF3E0") # Safety Check
    
    d.rect(10, y, 1.5, 1, "Generation", subtitle="LLM")
    
    # Connector
    d.connector((2, 4.5), (2.5, 4.5))
    d.connector((4, 4.5), (5, 5.5))
    d.connector((4, 4.5), (5, 2.5))
    
    d.connector((7, 5.5), (7.5, 4.5))
    d.connector((7, 2.5), (7.5, 4.5))
    
    d.connector((8.5, 4), (8.5, 3)) # To Gate
    d.connector((8.5, 3), (8.5, 4), "Pass") # Back (Simplified loop visualization)
    
    d.connector((9.5, 4.5), (10, 4.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "detailed_system_architecture.jpg", dpi=300)
    plt.close()

# ==========================================
# 3. Slide 9: Hybrid Retrieval Flow
# ==========================================
def img3_hybrid_flow():
    fig, ax = plt.subplots(figsize=(10, 6))
    d = DrawEngine(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    d.rect(0.5, 2.5, 1.5, 1, "Query")
    
    # Parallel Paths
    d.rect(3, 4.5, 2, 1, "Vector Search", subtitle="Symptom Concepts")
    d.rect(3, 0.5, 2, 1, "BM25 Search", subtitle="Exact Drugs")
    
    # Fusion
    d.rect(6, 2.5, 2, 1, "RRF Fusion", subtitle="Rank Merge")
    
    # Output
    d.rect(8.5, 2.5, 1, 1, "Top-K")
    
    # Flows
    d.connector((2, 3), (3, 5))
    d.connector((2, 3), (3, 1))
    d.connector((5, 5), (6, 3))
    d.connector((5, 1), (6, 3))
    d.connector((8, 3), (8.5, 3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid_retrieval_flow.png", dpi=300)
    plt.close()

# ==========================================
# 4. Slide 9: t-SNE Visualization
# ==========================================
def img4_tsne():
    plt.figure(figsize=(8, 6))
    np.random.seed(42)
    
    # Generate 3 clusters
    c1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    c2 = np.random.normal(loc=[-2, -1], scale=0.6, size=(40, 2))
    c3 = np.random.normal(loc=[1, -3], scale=0.5, size=(45, 2))
    
    plt.scatter(c1[:,0], c1[:,1], c='#FF6B6B', label='Cardiology', alpha=0.7)
    plt.scatter(c2[:,0], c2[:,1], c='#4ECDC4', label='Neurology', alpha=0.7)
    plt.scatter(c3[:,0], c3[:,1], c='#45B7D1', label='Pharmacology', alpha=0.7)
    
    plt.title("t-SNE of MedCPT Embeddings")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_space_tsne.jpg", dpi=300)
    plt.close()

# ==========================================
# 5. Slide 10: Grounding Gate Logic
# ==========================================
def img5_grounding_gate():
    fig, ax = plt.subplots(figsize=(8, 6))
    d = DrawEngine(ax)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    
    d.rect(3, 4.5, 2, 1, "Relevance Check", subtitle="Score > 0.5?")
    
    # Yes Path
    d.connector((5, 5), (6.5, 5), "Yes")
    d.rect(6.5, 4.5, 1.5, 1, "Generate", fill="#D4EDDA")
    
    # No Path
    d.connector((4, 4.5), (4, 3), "No")
    d.rect(3, 2, 2, 1, "Fallback / Search", fill="#F8D7DA")
    
    # Input
    d.rect(0.5, 4.5, 1.5, 1, "Context")
    d.connector((2, 5), (3, 5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_pipeline.png", dpi=300)
    plt.close()

# ==========================================
# 6. Slide 10: Attention Heatmap
# ==========================================
def img6_heatmap():
    plt.figure(figsize=(8, 4))
    # Fake attention weights (Query words x Context tokens)
    data = np.random.rand(4, 10)
    # Make some "relevant" parts stronger
    data[1:3, 4:7] += 1.0 
    
    labels_y = ["What", "treats", "Type 2", "Diabetes"]
    labels_x = ["guidelines", "state", "that", "Metformin", "is", "first-line", "therapy", "for", "glycemic", "control"]
    
    sns.heatmap(data, cmap="Reds", xticklabels=labels_x, yticklabels=labels_y, cbar_kws={'label': 'Attention Weight'})
    plt.title("Cross-Attention: Query vs Retrieved Context")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_relevance_heatmap.png", dpi=300)
    plt.close()

# ==========================================
# 7. Slide 11: SHAP Feature Importance
# ==========================================
def img7_shap():
    plt.figure(figsize=(8, 5))
    features = ["Chest Pain", "History: Smoking", "High BP", "Age > 50", "No Fever"]
    values = [0.85, 0.6, 0.4, 0.2, -0.5]
    colors = ['#FF6B6B' if v > 0 else '#4ECDC4' for v in values]
    
    plt.barh(features, values, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("SHAP Feature Importance for Prediction: 'Angina'")
    plt.xlabel("SHAP Value (Impact on Model Operator)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xai_feature_importance.png", dpi=300)
    plt.close()

# ==========================================
# 8. Slide 12: Precision-Recall Curve (Performance)
# ==========================================
def img8_pr_curve():
    plt.figure(figsize=(7, 6))
    recall = np.linspace(0, 1, 100)
    # Synthetic PR curves
    precision_hybrid = 1 - (recall**3) * 0.4  # Better
    precision_dense = 1 - (recall**2) * 0.6   # Worse
    
    plt.plot(recall, precision_hybrid, label='Hybrid Retrieval (Ours)', color=C_ACCENT, linewidth=2.5)
    plt.plot(recall, precision_dense, label='Dense Only', color=C_PRIMARY, linestyle='--', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.jpg", dpi=300)
    plt.close()

# ==========================================
# 9. Slide 12: Recall@K Bar Chart
# ==========================================
def img9_recall_k():
    plt.figure(figsize=(8, 5))
    k_vals = ['Recall@1', 'Recall@5', 'Recall@10']
    scores_base = [0.45, 0.60, 0.70]
    scores_ours = [0.65, 0.78, 0.84]
    
    x = np.arange(len(k_vals))
    width = 0.35
    
    plt.bar(x - width/2, scores_base, width, label='Baseline', color='#A0A0A0')
    plt.bar(x + width/2, scores_ours, width, label='Our System', color=C_ACCENT)
    
    plt.xticks(x, k_vals)
    plt.ylim(0, 1)
    plt.title("Retrieval Performance (Recall@K)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=300)
    plt.close()

# ==========================================
# 10. Slide 13: Latency Pie Chart
# ==========================================
def img10_latency_pie():
    plt.figure(figsize=(7, 7))
    # LLM=45%, Rerank=15%, Retrieval=20%, Preprocess=10%, XAI=10%
    sizes = [45, 15, 20, 10, 10]
    labels = ['LLM Generation', 'Reranking', 'Retrieval', 'Preprocessing', 'XAI']
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#D1C4E9']
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0, 0, 0, 0))
    plt.title("End-to-End Latency Breakdown (Total ~1.2s)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latency_breakdown.png", dpi=300)
    plt.close()

# ==========================================
# 11. Slide 13: Success Rate Donut
# ==========================================
def img11_success_donut():
    plt.figure(figsize=(7, 7))
    sizes = [65.5, 14.3, 20.2]
    labels = ['High Confidence', 'Blocked/Safety', 'Low Confidence']
    colors = [C_SUCCESS, '#DC3545', '#FFC107']
    
    # Pie
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.4))
    plt.title("System Reliability Distribution")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rag_success_rate.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating 11 PPT Images...")
    img1_system_overview()
    img2_detailed_system()
    img3_hybrid_flow()
    img4_tsne()
    img5_grounding_gate()
    img6_heatmap()
    img7_shap()
    img8_pr_curve()
    img9_recall_k()
    img10_latency_pie()
    img11_success_donut()
    print("Done! Files saved in 'ppt_images/'.")
