
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_images_dir():
    if not os.path.exists("images"):
        os.makedirs("images")

def draw_box(ax, x, y, w, h, text, color='#E0E0E0', edge='black'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=edge, facecolor=color, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, zorder=2, wrap=True)
    return x+w/2, y+h/2, x+w/2, y, x+w/2, y+h, x, y+h/2, x+w, y+h/2

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.05, text, ha='center', fontsize=8, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.8, "System Architecture: Traceable Healthcare Chatbot", ha='center', fontsize=14, weight='bold')
    
    # Nodes
    # Client Side
    cb_x, cb_y = 1, 4
    draw_box(ax, cb_x, cb_y, 2, 1, "Client\n(Streamlit UI)", color='#ADD8E6')
    
    # Server Side Container
    rect = patches.Rectangle((3.5, 0.5), 6, 4.8, linewidth=1, edgecolor='gray', facecolor='#F5F5F5', linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(6.5, 5.1, "Backend System (FastAPI)", ha='center', fontsize=10, style='italic')
    
    # API Layer
    api_x, api_y = 4, 4
    draw_box(ax, api_x, api_y, 2, 1, "API Gateway\n(FastAPI)", color='#90EE90')
    
    # RAG Engine
    rag_x, rag_y = 4, 2
    draw_box(ax, rag_x, rag_y, 2, 1, "Corrective RAG\nEngine", color='#FFB6C1')
    
    # Vector DB
    vdb_x, vdb_y = 7, 2
    draw_box(ax, vdb_x, vdb_y, 2, 1, "Vector DB\n(ChromaDB)", color='#FFD700')
    
    # LLM
    llm_x, llm_y = 4, 0.6
    draw_box(ax, llm_x, llm_y, 2, 0.8, "LLM Service\n(HuggingFace/Ollama)", color='#FFA07A')
    
    # XAI
    xai_x, xai_y = 7, 4
    draw_box(ax, xai_x, xai_y, 2, 1, "XAI Module\n(SHAP/LIME)", color='#D8BFD8')

    # Connections
    # Client -> API
    draw_arrow(ax, 3, 4.5, 4, 4.5, "HTTP/JSON")
    
    # API -> RAG
    draw_arrow(ax, 5, 4, 5, 3, "Query")
    
    # RAG -> VectorDB (Retrieve)
    draw_arrow(ax, 6, 2.5, 7, 2.5, "Retrieve\nContext")
    
    # RAG -> LLM (Generate)
    draw_arrow(ax, 5, 2, 5, 1.4, "Prompt")
    
    # API -> XAI
    draw_arrow(ax, 6, 4.5, 7, 4.5, "Explain")
    
    plt.tight_layout()
    plt.savefig("images/system_architecture.png", dpi=300)
    plt.close()
    print("Generated system_architecture.png")

def generate_rag_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    ax.text(6, 4.8, "Corrective RAG Pipeline Flow", ha='center', fontsize=14, weight='bold')

    # Steps
    y_pos = 2.5
    w, h = 1.5, 1
    
    # 1. User Query
    x1 = 0.5
    draw_box(ax, x1, y_pos, w, h, "User Query", color='#E6E6FA')
    
    # 2. Retrieval
    x2 = 2.5
    draw_box(ax, x2, y_pos, w, h, "Retrieve Top-K\n(Vector Search)", color='#87CEFA')
    
    # 3. Grade / Correct
    x3 = 4.5
    draw_box(ax, x3, y_pos, w, h, "Grade Documents\n(Relevance Check)", color='#98FB98')
    
    # Decision Point (Diamond ideally, but box for now)
    # If relevant -> Proceed
    # If ambiguous -> Refine
    
    # 4. Context Compression
    x4 = 6.5
    draw_box(ax, x4, y_pos, w, h, "Context\nCompression\n& Reorder", color='#DDA0DD')
    
    # 5. Generation
    x5 = 8.5
    draw_box(ax, x5, y_pos, w, h, "LLM Generation", color='#F08080')
    
    # 6. Response
    x6 = 10.5
    draw_box(ax, x6, y_pos, 1, h, "Answer", color='#E6E6FA')
    
    # Arrows
    draw_arrow(ax, x1+w, y_pos+h/2, x2, y_pos+h/2)
    draw_arrow(ax, x2+w, y_pos+h/2, x3, y_pos+h/2)
    draw_arrow(ax, x3+w, y_pos+h/2, x4, y_pos+h/2, "Filtered")
    draw_arrow(ax, x4+w, y_pos+h/2, x5, y_pos+h/2)
    draw_arrow(ax, x5+w, y_pos+h/2, x6, y_pos+h/2)
    
    # Corrective Loop
    # From Grade back to Retrieve (simplified visual)
    ax.annotate("Refine Query", xy=(x2+w/2, y_pos+h), xytext=(x3+w/2, y_pos+h),
                arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=0.3", color='red'),
                color='red', fontsize=8, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("images/rag_pipeline.png", dpi=300)
    plt.close()
    print("Generated rag_pipeline.png")

def generate_sequence_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, "Query Processing Sequence", ha='center', fontsize=14, weight='bold')
    
    # Actors/Lifelines
    actors = ["User", "UI", "Orchestrator", "Retriever", "LLM"]
    x_positions = [1, 3, 5, 7, 9]
    
    for actor, x in zip(actors, x_positions):
        draw_box(ax, x-0.5, 8.5, 1, 0.5, actor)
        ax.plot([x, x], [0.5, 8.5], linestyle='--', color='gray')
    
    # Interactions
    y = 8
    step = 0.8
    
    # 1. User -> UI
    draw_arrow(ax, 1, y, 3, y, "Enters Query")
    y -= step
    
    # 2. UI -> Orchestrator
    draw_arrow(ax, 3, y, 5, y, "POST /chat")
    y -= step
    
    # 3. Orch -> Retriever
    draw_arrow(ax, 5, y, 7, y, "Get Context")
    y -= step
    
    # 4. Retriever -> Orch
    draw_arrow(ax, 7, y, 5, y, "Documents")
    ax.text(6, y + 0.1, "(Compressed)", fontsize=8, color='gray', ha='center')
    y -= step
    
    # 5. Orch -> LLM
    draw_arrow(ax, 5, y, 9, y, "Prompt(Query+Ctx)")
    y -= step
    
    # 6. LLM -> Orch
    draw_arrow(ax, 9, y, 5, y, "Response")
    y -= step
    
    # 7. Orch -> UI
    draw_arrow(ax, 5, y, 3, y, "JSON (Ans+Sources)")
    y -= step
    
    # 8. UI -> User
    draw_arrow(ax, 3, y, 1, y, "Display Answer")
    
    plt.tight_layout()
    plt.savefig("images/sequence_diagram.png", dpi=300)
    plt.close()
    print("Generated sequence_diagram.png")
    
if __name__ == "__main__":
    create_images_dir()
    try:
        generate_system_architecture()
        generate_rag_pipeline()
        generate_sequence_diagram()
        print("All diagrams generated successfully.")
    except ImportError:
        print("Error: matplotlib is not installed. Please install it with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating diagrams: {e}")
