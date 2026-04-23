import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/hybrid_retrieval_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/hybrid_retrieval_test.mmd"

mermaid_code = """
graph TB
    Query[User Query]

    subgraph "Encoding Layer"
        QENC[Query Encoder<br/>all-MiniLM-L6-v2]
        BM25T[BM25 Tokeniser<br/>whitespace + lowercase]
    end

    subgraph "Storage Layer"
        CHROMA[(ChromaDB Vector Store<br/>505,584 docs)]
        BM25I[(BM25 Index<br/>pickle.gz cache)]
    end

    subgraph "Retrieval Results Layer"
        DR[Dense Results<br/>top-k = 10]
        SR[Sparse Results<br/>top-k = 10]
    end

    subgraph "Fusion & Processing Layer"
        RRF[Reciprocal Rank Fusion<br/>RRF]
        SD[Source Diversity Filter]
        DK[Dynamic k Selection<br/>factual=5, research=10]
    end

    CTX[Top Documents<br/>to Generator]

    Query --> QENC
    Query --> BM25T

    QENC --> CHROMA
    BM25T --> BM25I

    CHROMA --> DR
    BM25I --> SR

    DR --> RRF
    SR --> RRF

    RRF --> SD
    SD --> DK
    DK --> CTX

    style Query fill:#f5f5f5
    style CTX fill:#f5f5f5
    style QENC fill:#4ecdc4
    style BM25T fill:#4ecdc4
    style CHROMA fill:#a0d468
    style BM25I fill:#a0d468
    style DR fill:#ffce54
    style SR fill:#ffce54
    style RRF fill:#fc6e51
    style SD fill:#fc6e51
    style DK fill:#fc6e51
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
