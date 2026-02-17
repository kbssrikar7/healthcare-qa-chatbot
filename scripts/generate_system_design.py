import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_box(ax, x, y, w, h, text, color='#E0E0E0', edge='black', fontweight='normal'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, zorder=4, wrap=True, fontweight=fontweight)
    return x, y, w, h

def draw_arrow(ax, x1, y1, x2, y2, text=None, color='black'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color), zorder=5)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2, text, ha='center', va='bottom', fontsize=8, color='#333333',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), zorder=6)

def generate_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, "Advanced Medical RAG System Design", ha='center', fontsize=18, weight='bold')

    # External User
    draw_box(ax, 0.5, 7.5, 2, 1, "Medical Professional\n(User)", color='#FFE4B5', fontweight='bold')
    
    # Frontend
    draw_box(ax, 3.5, 7.5, 2.5, 1, "Frontend\n(Streamlit)", color='#ADD8E6')
    draw_arrow(ax, 2.5, 8, 3.5, 8, "Query / UI")

    # API
    draw_box(ax, 7, 7.5, 2.5, 1, "Backend API\n(FastAPI)", color='#98FB98')
    draw_arrow(ax, 6, 8, 7, 8, "REST/JSON")

    # Pipeline Boundary (Large Box)
    pipeline_rect = patches.Rectangle((3, 1), 10, 5.5, linewidth=2, edgecolor='#555555', facecolor='#FAFAFA', linestyle='--', zorder=1)
    ax.add_patch(pipeline_rect)
    ax.text(8, 6.2, "Healthcare RAG Pipeline (Core Orchestrator)", ha='center', fontsize=12, weight='bold', color='#444444')

    # Retrieval Stage
    draw_box(ax, 3.5, 4.5, 2.5, 1, "Hybrid Retriever\n(Dense + Sparse)", color='#FFD700')
    draw_arrow(ax, 8.25, 7.5, 4.75, 5.5, "1. Retrieve")
    
    # Data Sources
    draw_box(ax, 3.5, 1.5, 1.1, 0.8, "ChromaDB\n(Dense)", color='#F0E68C')
    draw_box(ax, 4.9, 1.5, 1.1, 0.8, "BM25\n(Sparse)", color='#F0E68C')
    draw_arrow(ax, 4.05, 4.5, 4.05, 2.3)
    draw_arrow(ax, 5.45, 4.5, 5.45, 2.3)

    # Reranker & Grounding
    draw_box(ax, 7, 4.5, 2.5, 1, "Reranker &\nGrounding Gate", color='#FF6347')
    draw_arrow(ax, 6, 5, 7, 5, "2. Refine")

    # Generation
    draw_box(ax, 10, 4.5, 2.5, 1, "Medical LLM\n(BioMistral/TinyLlama)", color='#DDA0DD')
    draw_arrow(ax, 9.5, 5, 10, 5, "3. Generate")

    # XAI Module
    draw_box(ax, 10, 2.5, 2.5, 1, "XAI Module\n(Explainability)", color='#87CEFA')
    draw_arrow(ax, 11.25, 4.5, 11.25, 3.5, "4. Explain")
    
    # Final Response
    draw_arrow(ax, 10, 3, 8.25, 7.5, "5. Response + XAI", color='blue')

    # Legend / Info
    info_text = "Key Features:\n• Hybrid RRF Retrieval\n• Grounding Check (Anti-Hallucination)\n• Source Attribution\n• Confidence Scoring"
    ax.text(0.5, 0.5, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    output_path = "docs/architecture/system_design.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    generate_diagram()
