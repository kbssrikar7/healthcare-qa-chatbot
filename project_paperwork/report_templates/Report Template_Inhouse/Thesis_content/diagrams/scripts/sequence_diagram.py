import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/sequence_diagram_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/sequence_diagram_test.mmd"

mermaid_code = """
sequenceDiagram
    actor User
    participant API as FastAPI
    participant Cache as Cache Manager
    participant Safety as Safety Guardrails
    participant Retriever as Hybrid Retriever
    participant LLM as LLM Backend
    participant XAI as XAI Module

    User->>API: POST /ask { query, session_id }
    API->>Cache: lookup(query)
    
    alt Cache hit
        Cache-->>API: cached response
        API-->>User: 200 OK (fast path)
    else Cache miss
        API->>Safety: check_safety(query)
        
        alt Emergency
            Safety-->>API: redirect message
            API-->>User: safety redirect
        else Safe
            Safety-->>API: None (safe)
            
            API->>Retriever: retrieve(query, k=10)
            Note over Retriever: BM25 + Dense<br/>-> RRF fusion
            Retriever-->>API: top-k documents
            
            API->>LLM: generate(context, query)
            LLM-->>API: answer + token scores
            
            API->>XAI: score(query, answer, docs)
            Note over XAI: 5 signals + Platt<br/>DeBERTa NLI check
            XAI-->>API: confidence + hallucination flags
            
            API->>Cache: store(query, response)
            API-->>User: answer + sources + confidence + latencies
        end
    end
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
