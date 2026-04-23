import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/data_ingestion_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/data_ingestion_test.mmd"

mermaid_code = """
graph TB
    subgraph "Source Datasets"
        ChatDoc[ChatDoctor<br/>HealthCareMagic]
        PubMed[PubMedQA]
        MedMCQA[MedMCQA]
    end

    subgraph "Pre-processing Layer"
        Format[Format Normalisation]
        Clean[Text Cleaning]
        Chunk[RecursiveSentenceChunker<br/>512-token target]
    end

    subgraph "Embedding Layer"
        Encoder[Language Model<br/>all-MiniLM-L6-v2]
    end

    subgraph "Storage Layer"
        Chroma[(ChromaDB<br/>Vector Store)]
        BM25[(BM25 Index<br/>Pickle Cache)]
    end

    subgraph "Indexing Layer"
        Builder[BM25 Index Build<br/>rank-bm25]
    end

    ChatDoc --> Format
    PubMed --> Format
    MedMCQA --> Format

    Format --> Clean
    Clean --> Chunk

    Chunk -->|Plain text| Encoder
    Encoder -->|Upsert vectors| Chroma

    Chunk -->|Raw tokens| Builder
    Builder -->|Serialize| BM25

    style ChatDoc fill:#f5f5f5
    style PubMed fill:#f5f5f5
    style MedMCQA fill:#f5f5f5
    style Format fill:#4ecdc4
    style Clean fill:#4ecdc4
    style Chunk fill:#4ecdc4
    style Encoder fill:#ffce54
    style Builder fill:#ffce54
    style Chroma fill:#a0d468
    style BM25 fill:#a0d468
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
