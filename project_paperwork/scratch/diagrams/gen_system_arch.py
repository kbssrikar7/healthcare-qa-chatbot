import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images_gemini/system_architecture_modern.png"
output_mmd = "/home/kbs/Documents/final_project/scratch/diagrams/system_architecture.mmd"

mermaid_code = """
graph TB
    subgraph "Frontend Layer"
        User([User])
        NextJS[Next.js Assistant UI<br/>React / TypeScript]
    end

    subgraph "API Gateway"
        FastAPI[FastAPI Orchestrator<br/>/ask /health]
    end

    subgraph "Safety & Guardrails"
        Safety[SafetyGuardrails<br/>Emergency Detection]
        Filter[Medical Stop-word<br/>Filtering]
    end

    subgraph "8-Stage RAG Pipeline"
        S1[1. Query Enhancement]
        S2[2. Hybrid Retrieval]
        S3[3. Corrective RAG]
        S4[4. Grounding Gate]
        S5[5. Context Compression]
        S6[6. LLM Generation]
        S7[7. Factual Consistency]
        S8[8. Confidence Scoring]
    end

    subgraph "Storage Layer (Localhost)"
        Chroma[(ChromaDB<br/>Dense Vectors)]
        BM25[(BM25 Index<br/>Sparse Persistence)]
    end

    User --> NextJS
    NextJS -->|JSON/POST| FastAPI
    
    FastAPI --> Safety
    Safety --> S1
    
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 -->|Sufficient Context| S5
    S5 --> S6
    S6 --> S7
    S7 --> S8
    
    S8 -->|Response + XAI| FastAPI
    FastAPI -->|Streaming JSON| NextJS
    
    S2 -.-> Chroma
    S2 -.-> BM25
    
    %% Styling
    style User fill:#f9f9f9,stroke:#333
    style NextJS fill:#4ecdc4,stroke:#333,color:#000
    style FastAPI fill:#ffce54,stroke:#333,color:#000
    style Safety fill:#fc6e51,stroke:#333,color:#fff
    style Filter fill:#fc6e51,stroke:#333,color:#fff
    style S1,S2,S3,S4,S5,S6,S7,S8 fill:#a0d468,stroke:#333,color:#000
    style Chroma,BM25 fill:#5d9cec,stroke:#333,color:#fff
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("/home/kbs/Documents/final_project/scratch/diagrams/puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([
    mmdc_path, 
    "-i", output_mmd, 
    "-o", output_png, 
    "-t", "default", 
    "-b", "white", 
    "-s", "3", 
    "-p", "/home/kbs/Documents/final_project/scratch/diagrams/puppeteer-config.json"
])

print(f"Generated: {output_png}")
