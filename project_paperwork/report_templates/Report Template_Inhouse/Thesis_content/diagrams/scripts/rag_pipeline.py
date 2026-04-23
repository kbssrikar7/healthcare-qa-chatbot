import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/rag_pipeline_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/rag_pipeline_test.mmd"

mermaid_code = """
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
graph TB
    Query[User Query]
    
    subgraph "Query Processing Layer"
        S1[1. Query Enhancement<br/>TinyLlama expander]
        S2[2. Hybrid Retrieval<br/>BM25 + Dense + RRF]
        S3[3. Corrective RAG<br/>relevance check]
        S4{4. Grounding Gate<br/>adaptive threshold}
    end

    subgraph "Fallback Engine"
        Abstain[Extractive Answer<br/>source-backed fallback]
    end

    subgraph "Generation & Validation Layer"
        S5[5. Context Compression<br/>token-limit trim]
        S6[6. LLM Generation<br/>TinyLlama / BioMistral]
        S7[7. Factual Consistency<br/>DeBERTa NLI]
        S8[8. Confidence Scoring<br/>5 signals + Platt]
    end

    Resp[Response + Confidence]
    Cache[(Response Cache<br/>TTL-based)]

    Query --> S1
    Query -.->|cache hit| Cache
    
    S1 --> S2
    S2 --> S3
    S3 --> S4

    S4 -->|insufficient re-retrieve| S2
    S4 -->|sufficient| S5
    S4 -->|fails abstain| Abstain

    S5 --> S6
    S6 --> S7
    S7 --> S8
    S8 --> Resp
    Abstain --> Resp

    Resp -.->|store| Cache

    style Query fill:#f5f5f5
    style Resp fill:#f5f5f5
    style Cache fill:#ac92ec
    style S1 fill:#4ecdc4
    style S2 fill:#ffce54
    style S3 fill:#fc6e51
    style S4 fill:#ed5565
    style S5 fill:#4ecdc4
    style S6 fill:#ffce54
    style S7 fill:#fc6e51
    style S8 fill:#ed5565
    style Abstain fill:#da4453
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
