import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import numpy as np
from pathlib import Path

# ==========================================
# Configuration & Style
# ==========================================
OUTPUT_DIR = Path("images/publication_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enterprise Color Palette
C_PRIMARY = "#003366"    # Navy Blue (Borders/Main)
C_FILL_SVC = "#E6F2FF"   # Light Blue (Services)
C_FILL_DB  = "#E6FFEA"   # Light Green (Databases)
C_FILL_EXT = "#F0F0F0"   # Light Grey (External/Users)
C_ACCENT   = "#FF6600"   # Orange (Highlights)
C_TEXT     = "#000000"
C_ARROW    = "#333333"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1,
})

class DrawEngine:
    def __init__(self, ax):
        self.ax = ax
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def rect(self, x, y, w, h, label, fill=C_FILL_SVC, border=C_PRIMARY, subtitle=None):
        """Standard Service Rectangle"""
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", 
                             ec=border, fc=fill, lw=1.5, zorder=10)
        self.ax.add_patch(box)
        
        cx, cy = x + w/2, y + h/2
        self.ax.text(cx, cy + (0.15 if subtitle else 0), label, ha='center', va='center', 
                     fontweight='bold', color=C_TEXT, zorder=11)
        if subtitle:
            self.ax.text(cx, cy - 0.15, subtitle, ha='center', va='center', 
                         fontsize=8, color='#555555', zorder=11)
        return (x, y, w, h)

    def database(self, x, y, w, h, label):
        """Cylinder Shape for DB"""
        # Ellipse top
        top = mpatches.Ellipse((x + w/2, y + h), w, h*0.3, ec=C_PRIMARY, fc=C_FILL_DB, lw=1.5, zorder=12)
        # Rectangle body
        body = mpatches.Rectangle((x, y + h*0.15), w, h*0.85, ec='none', fc=C_FILL_DB, zorder=11)
        # Bottom curve
        bottom = mpatches.Arc((x + w/2, y + h*0.15), w, h*0.3, theta1=180, theta2=360, ec=C_PRIMARY, lw=1.5, zorder=12)
        # Side lines
        self.ax.plot([x, x], [y + h*0.15, y + h], color=C_PRIMARY, lw=1.5, zorder=12)
        self.ax.plot([x+w, x+w], [y + h*0.15, y + h], color=C_PRIMARY, lw=1.5, zorder=12)
        
        self.ax.add_patch(top)
        self.ax.add_patch(body)
        self.ax.add_patch(bottom)
        
        self.ax.text(x + w/2, y + h*0.5, label, ha='center', va='center', 
                     fontweight='bold', fontsize=8, zorder=13)

    def actor(self, x, y, label):
        """Stick figure user"""
        # Head
        head = mpatches.Circle((x, y + 0.8), 0.2, ec=C_PRIMARY, fc='white', lw=1.5)
        self.ax.add_patch(head)
        # Body
        self.ax.plot([x, x], [y + 0.6, y + 0.3], color=C_PRIMARY, lw=1.5)
        # Arms
        self.ax.plot([x - 0.25, x + 0.25], [y + 0.5, y + 0.5], color=C_PRIMARY, lw=1.5)
        # Legs
        self.ax.plot([x, x - 0.2], [y + 0.3, y], color=C_PRIMARY, lw=1.5)
        self.ax.plot([x, x + 0.2], [y + 0.3, y], color=C_PRIMARY, lw=1.5)
        
        self.ax.text(x, y - 0.2, label, ha='center', va='top', fontweight='bold')

    def connector(self, p1, p2, label=None, style='->'):
        """Orthogonal or straight arrow"""
        # Simple straight line for now, or elbow if needed
        # We'll use annotate for correct arrow heads
        self.ax.annotate("", xy=p2, xytext=p1, 
                         arrowprops=dict(arrowstyle=style, color=C_ARROW, lw=1.5))
        if label:
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            self.ax.text(mid[0], mid[1] + 0.1, label, ha='center', fontsize=8, 
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    def title(self, label):
        self.ax.text(0.5, 0.95, label, transform=self.ax.transAxes, 
                     ha='center', fontsize=16, fontweight='bold', color=C_PRIMARY)


# ==========================================
# Diagram Functions
# ==========================================

def slide6_system_overview():
    fig, ax = plt.subplots(figsize=(12, 7))
    d = DrawEngine(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    d.title("High-Level System Architecture")
    
    # Components
    d.actor(1, 4, "Patient")
    
    d.rect(2.5, 3.5, 2, 1.5, "Frontend UI", subtitle="Streamlit")
    d.rect(5.5, 3.5, 2, 1.5, "Orchestrator", subtitle="FastAPI")
    
    # RAG Container
    rag_box = FancyBboxPatch((8, 1), 3.5, 6, boxstyle="round,pad=0.2", 
                             ec=C_PRIMARY, fc="#F5F5F5", linestyle="--")
    ax.add_patch(rag_box)
    ax.text(9.75, 6.7, "RAG Engine", ha='center', fontweight='bold', color='#555555')
    
    d.rect(8.5, 5, 2.5, 1, "Retrieval", subtitle="Hybrid (BM25+Dense)")
    d.rect(8.5, 3.5, 2.5, 1, "Reranker", subtitle="Cross-Encoder")
    d.rect(8.5, 2, 2.5, 1, "Generative Model", subtitle="BioMistral-7B")
    
    # Database
    d.database(8.75, 0, 2, 1.2, "Vector DB\n(Chroma)")
    
    # Flows
    d.connector((1.3, 4.5), (2.5, 4.5)) # User -> UI
    d.connector((4.5, 4.25), (5.5, 4.25), "JSON") # UI -> API
    d.connector((7.5, 4.25), (8.5, 5.5), "Query") # API -> Retrieval
    
    # Internal RAG flows
    d.connector((9.75, 5), (9.75, 4.5))
    d.connector((9.75, 3.5), (9.75, 3))
    
    # Return path
    d.connector((9.75, 2), (7.5, 3.8), "Response") # Gen -> API
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide6_system_overview.png", dpi=300, facecolor='white')
    plt.close()

def slide7_detailed_system():
    fig, ax = plt.subplots(figsize=(14, 9))  # Wider
    d = DrawEngine(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    
    d.title("Detailed Healthcare RAG Pipeline Component View")
    
    # Layout Grid
    y_main = 5
    
    # 1. Input Processing
    d.rect(0.5, y_main, 2, 1, "Query Processing", subtitle="Clean/NER")
    
    # 2. Embedding
    d.rect(3, y_main, 2, 1, "Embedding Model", subtitle="MedCPT")
    
    # 3. Retrieval
    d.rect(5.5, y_main+1.5, 2, 1, "Dense Retrieval")
    d.rect(5.5, y_main-1.5, 2, 1, "Sparse Retrieval", subtitle="BM25")
    
    # DB
    d.database(5.5, y_main-0.25, 2, 1.5, "ChromaDB")
    
    # 4. Fusion
    d.rect(8, y_main, 2, 1, "Hybrid Fusion\n& Reranking")
    
    # 5. Generation
    d.rect(10.5, y_main, 2.5, 1, "LLM Generation", subtitle="BioMistral (QLoRA)")
    
    # 6. XAI
    d.rect(10.5, 2, 2.5, 1.5, "XAI Module", subtitle="SHAP/Citations")
    
    # 7. Output processing
    d.rect(10.5, 0.5, 2.5, 1, "Output Formatter")
    
    # Connectors
    d.connector((2.5, 5.5), (3, 5.5))
    d.connector((5, 5.5), (5.5, 5.5)) # To Middle? No
    
    # Arrows (Manual precise)
    ax.annotate("", xy=(5.5, 6), xytext=(4, 6), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    # Embedding -> Dense
    d.connector((5, 5.5), (5.5, 6.5)) # Emb -> Dense (Approx)
    d.connector((5, 5.5), (5.5, 4))   # Emb -> Sparse
    
    d.connector((7.5, 6.5), (8, 6))   # Dense -> Fusion
    d.connector((7.5, 4), (8, 5))     # Sparse -> Fusion
    
    d.connector((10, 5.5), (10.5, 5.5)) # Fusion -> Gen
    
    d.connector((11.75, 5), (11.75, 3.5)) # Gen -> XAI
    d.connector((11.75, 2), (11.75, 1.5)) # XAI -> Format
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide7_detailed_system.png", dpi=300, facecolor='white')
    plt.close()

def slide9_hybrid_retrieval():
    fig, ax = plt.subplots(figsize=(10, 6))
    d = DrawEngine(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    d.title("Hybrid Retrieval Architecture")
    
    # Input
    d.rect(0.5, 2.5, 1.5, 1, "Query")
    
    # Split
    d.connector((2, 3), (3, 4.5))
    d.connector((2, 3), (3, 1.5))
    
    # Path 1: Dense
    d.rect(3, 4, 2, 1, "Dense Enc", subtitle="MedCPT")
    d.database(5.5, 3.8, 1.5, 1.2, "Vector DB")
    
    # Path 2: Sparse
    d.rect(3, 1, 2, 1, "Keyword Ext")
    d.database(5.5, 0.8, 1.5, 1.2, "Inv. Index")
    
    # Fusion
    d.rect(8, 2, 1.5, 2, "Hybrid\nFusion", subtitle="Reciprocal Rank")
    
    # Connectors
    d.connector((5, 4.5), (5.5, 4.5))
    d.connector((5, 1.5), (5.5, 1.5))
    
    d.connector((7, 4.5), (8, 3.5))
    d.connector((7, 1.5), (8, 2.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide9_hybrid_retrieval.png", dpi=300, facecolor='white')
    plt.close()

def slide10_corrective_rag():
    fig, ax = plt.subplots(figsize=(10, 7))
    d = DrawEngine(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    d.title("Corrective RAG (CRAG) Logic")
    
    # Flow
    d.rect(1, 6, 2, 1, "Retrieved Docs")
    
    # Decision Diamond
    # Draw logic check box
    d.rect(4, 5.5, 2, 2, "Grounding\nEvaluator", subtitle="Relevance Check")
    
    # Paths
    d.rect(7, 6.5, 2, 1, "Relevant", fill="#D4EDDA", border="#28A745")
    d.rect(7, 4.5, 2, 1, "Irrelevant", fill="#F8D7DA", border="#DC3545")
    
    d.rect(7, 2, 2, 1.5, "Knowledge\nRefinement", subtitle="Web Search / Filter")
    
    d.rect(4, 0.5, 2, 1, "Final Context")
    
    # Connectors
    d.connector((3, 6.5), (4, 6.5))
    d.connector((6, 7), (7, 7))
    d.connector((6, 5), (7, 5))
    
    d.connector((9, 7), (9, 3.5), style="-")
    d.connector((9, 5), (9, 3.5))
    d.connector((9, 3.5), (9, 2)) # Down to refinement
    
    d.connector((8, 2), (6, 1))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide10_corrective_rag.png", dpi=300, facecolor='white')
    plt.close()

def slide11_xai_module_linear():
    fig, ax = plt.subplots(figsize=(12, 6))
    d = DrawEngine(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    
    d.title("XAI Module Processing Pipeline")
    
    # 1. Input
    d.rect(0.5, 2.5, 2, 1, "Raw LLM Output")
    
    # 2. Parallel Analysis Processors
    d.rect(3.5, 4.5, 2.5, 1, "Feature Importance", subtitle="SHAP / LIME")
    d.rect(3.5, 2.5, 2.5, 1, "Confidence Scorer", subtitle="Logits Analysis")
    d.rect(3.5, 0.5, 2.5, 1, "Source Attribution", subtitle="Citation Matching")
    
    # 3. Aggregation
    d.rect(7, 2, 2, 2, "Explanation\nAggregator")
    
    # 4. Output
    d.rect(10, 2.25, 1.5, 1.5, "Explainable\nResponse", fill="#FFF3CD", border=C_ACCENT)
    
    # Flow
    # Split
    d.connector((2.5, 3), (3.5, 5))
    d.connector((2.5, 3), (3.5, 3))
    d.connector((2.5, 3), (3.5, 1))
    
    # Join
    d.connector((6, 5), (7, 3.5))
    d.connector((6, 3), (7, 3))
    d.connector((6, 1), (7, 2.5))
    
    d.connector((9, 3), (10, 3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "slide11_xai_module.png", dpi=300, facecolor='white')
    plt.close()
    
# Charts
def generate_charts():
    # Accuracy
    plt.figure(figsize=(8, 5))
    models = ['Baseline', 'Naive RAG', 'Hybrid', 'Our System']
    acc = [0.62, 0.74, 0.81, 0.89]
    plt.bar(models, acc, color=[C_PRIMARY]*3 + [C_ACCENT])
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1)
    plt.savefig(OUTPUT_DIR / "slide12_accuracy_results.png", dpi=300)
    plt.close()
    
    # Latency Scatter
    plt.figure(figsize=(8, 5))
    lat = [200, 600, 900, 1200]
    acc = [0.62, 0.74, 0.81, 0.89]
    plt.scatter(lat, acc, s=100, c='gray')
    plt.scatter([1200], [0.89], s=150, c=C_ACCENT, label='Our System')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Accuracy')
    plt.title('Performance Trade-off')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_DIR / "slide13_latency_results.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    slide6_system_overview()
    slide7_detailed_system()
    slide9_hybrid_retrieval()
    slide10_corrective_rag()
    slide11_xai_module_linear() # New linear layout
    generate_charts()
