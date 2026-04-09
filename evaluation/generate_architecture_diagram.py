#!/usr/bin/env python3
"""
Generate system architecture diagram for the Healthcare QA Chatbot paper.

Produces:
  1. Mermaid diagram (markdown for README)
  2. Matplotlib figure  (PNG/PDF for paper)

Usage:
    python evaluation/generate_architecture_diagram.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Mermaid Diagram ──────────────────────────────────────────────────────

MERMAID_DIAGRAM = r"""
```mermaid
flowchart TD
    subgraph Input["🔒 Input Layer"]
        A["User Question"] --> B["Input Safety Check<br/>(Emergency Detection, Content Filter)"]
        A --> B2["Query Sanitization<br/>(Prompt Injection Guard)"]
    end

    subgraph Retrieval["🔍 Hybrid Retrieval"]
        B --> C["Query Enhancement<br/>(Medical Term Expansion)"]
        C --> D1["Dense Retrieval<br/>(MedCPT / MiniLM<br/>via ChromaDB)"]
        C --> D2["Sparse Retrieval<br/>(BM25 with Medical<br/>Tokenization)"]
        D1 --> E["RRF Fusion<br/>(Reciprocal Rank Fusion)"]
        D2 --> E
        E --> F["Cross-Encoder Reranking<br/>(Optional)"]
    end

    subgraph QualityGate["✅ Quality Gate"]
        F --> G["Corrective RAG<br/>(Document Grading)"]
        G --> H["Grounding Gate<br/>(Adaptive Threshold)"]
        H -->|Insufficient| MCP["MCP Web Search<br/>(Fallback)"]
        H -->|Sufficient| I["Context Compression<br/>(Lost-in-Middle Mitigation)"]
        MCP --> I
    end

    subgraph Generation["🧠 LLM Generation"]
        I --> J["Prompt Construction<br/>(RAG Template)"]
        J --> K["LLM Inference<br/>(TinyLlama 1.1B /<br/>BioMistral 7B GGUF)"]
        K --> K2["Response Cleaning<br/>(Artifact Removal)"]
    end

    subgraph XAI["📊 XAI & Safety Layer"]
        K2 --> L["Hallucination Detection<br/>(DeBERTa NLI +<br/>Rule-Based)"]
        K2 --> M["Multi-Signal Confidence<br/>(5 Signals: Retrieval,<br/>Generation, Consistency,<br/>Source Agreement,<br/>Entity Coverage)"]
        K2 --> N["Source Attribution<br/>(Sentence-Level<br/>Evidence Mapping)"]
        K2 --> O["Output Safety Check<br/>(Diagnosis Prevention,<br/>Drug Interactions,<br/>Pediatric Warnings)"]
    end

    subgraph Output["📤 Output"]
        L --> P["Explainable Response<br/>(Answer + Confidence +<br/>Sources + Rationale +<br/>Disclaimer)"]
        M --> P
        N --> P
        O --> P
    end

    style Input fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style Retrieval fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style QualityGate fill:#d1fae5,stroke:#10b981,stroke-width:2px
    style Generation fill:#e0e7ff,stroke:#6366f1,stroke-width:2px
    style XAI fill:#fce7f3,stroke:#ec4899,stroke-width:2px
    style Output fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
```
""".strip()


# ── 2. Matplotlib Figure ────────────────────────────────────────────────────


def generate_matplotlib_figure():
    """Generate a paper-quality architecture diagram using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping figure generation")
        return

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Stage definitions: (x, y, width, height, label, color)
    stages = [
        # Row 1 — Input
        (
            0.5,
            6.8,
            3.0,
            0.9,
            "Input Safety\n(Emergency, Content Filter,\nPrompt Injection Guard)",
            "#fef3c7",
        ),
        # Row 2 — Retrieval
        (0.5, 5.2, 2.0, 0.9, "Query Enhancement\n(Medical Term Expansion)", "#dbeafe"),
        (
            3.0,
            5.2,
            2.0,
            0.9,
            "Dense Retrieval\n(MedCPT/MiniLM\nvia ChromaDB)",
            "#dbeafe",
        ),
        (
            5.5,
            5.2,
            2.0,
            0.9,
            "Sparse Retrieval\n(BM25 + Medical\nTokenization)",
            "#dbeafe",
        ),
        (8.0, 5.2, 2.0, 0.9, "RRF Fusion\n+ Cross-Encoder\nReranking", "#dbeafe"),
        # Row 3 — Quality Gate
        (0.5, 3.6, 2.5, 0.9, "Corrective RAG\n(Document Grading)", "#d1fae5"),
        (3.5, 3.6, 2.5, 0.9, "Grounding Gate\n(Adaptive Threshold)", "#d1fae5"),
        (
            6.5,
            3.6,
            2.5,
            0.9,
            "Context Compression\n(Lost-in-Middle\nMitigation)",
            "#d1fae5",
        ),
        # Row 4 — Generation
        (0.5, 2.0, 3.0, 0.9, "Prompt Construction\n(RAG Template)", "#e0e7ff"),
        (
            4.0,
            2.0,
            3.0,
            0.9,
            "LLM Inference\n(TinyLlama 1.1B /\nBioMistral 7B GGUF)",
            "#e0e7ff",
        ),
        # Row 5 — XAI
        (0.3, 0.4, 2.5, 0.9, "Hallucination\nDetection\n(DeBERTa NLI)", "#fce7f3"),
        (
            3.2,
            0.4,
            2.8,
            0.9,
            "Multi-Signal\nConfidence Scoring\n(5 Signals)",
            "#fce7f3",
        ),
        (
            6.4,
            0.4,
            2.5,
            0.9,
            "Source Attribution\n(Sentence-Level\nEvidence)",
            "#fce7f3",
        ),
        (
            9.3,
            0.4,
            2.5,
            0.9,
            "Output Safety\n(Drug Interactions,\nPediatric Warnings)",
            "#fce7f3",
        ),
    ]

    for x, y, w, h, label, color in stages:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#374151",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="medium",
            linespacing=1.35,
        )

    # Arrows between rows
    arrow_props = dict(arrowstyle="->", color="#6b7280", lw=1.5)
    # Input → Retrieval
    ax.annotate("", xy=(2.0, 5.95), xytext=(2.0, 6.65), arrowprops=arrow_props)
    # Retrieval → Quality Gate
    ax.annotate("", xy=(5.0, 4.35), xytext=(5.0, 5.05), arrowprops=arrow_props)
    # Quality Gate → Generation
    ax.annotate("", xy=(3.5, 2.75), xytext=(3.5, 3.45), arrowprops=arrow_props)
    # Generation → XAI
    ax.annotate("", xy=(5.5, 1.15), xytext=(5.5, 1.85), arrowprops=arrow_props)

    # Row labels
    labels = [
        (13.0, 7.15, "INPUT", "#f59e0b"),
        (13.0, 5.55, "RETRIEVAL", "#3b82f6"),
        (13.0, 3.95, "QUALITY\nGATE", "#10b981"),
        (13.0, 2.35, "GENERATION", "#6366f1"),
        (13.0, 0.75, "XAI &\nSAFETY", "#ec4899"),
    ]
    for x, y, label, color in labels:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
        )

    # Title
    ax.text(
        7,
        7.7,
        "Healthcare QA Chatbot — System Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
    )

    fig.tight_layout()

    png_path = OUT_DIR / "architecture_diagram.png"
    pdf_path = OUT_DIR / "architecture_diagram.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Architecture PNG → {png_path}")
    print(f"  Architecture PDF → {pdf_path}")


# ── 3. Save Mermaid ─────────────────────────────────────────────────────────


def save_mermaid():
    mermaid_path = OUT_DIR / "architecture_diagram.md"
    mermaid_path.write_text(MERMAID_DIAGRAM)
    print(f"  Mermaid diagram  → {mermaid_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating system architecture diagrams...")
    save_mermaid()
    generate_matplotlib_figure()
    print("Done.")
