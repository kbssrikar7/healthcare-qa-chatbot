import subprocess
import os
import json

mermaid_code = """
graph TD

  %% Styles
  classDef source fill:#EBF5FB,stroke:#2980B9,stroke-width:2px,color:#1A5276,font-size:24px,font-weight:bold,padding:15px
  classDef stage fill:#EAF4EC,stroke:#27AE60,stroke-width:2px,color:#1E8449,font-size:24px,font-weight:bold,padding:15px
  classDef embed fill:#FEF9E7,stroke:#F39C12,stroke-width:2px,color:#784212,font-size:24px,font-weight:bold,padding:15px
  classDef storage fill:#FDF2E9,stroke:#E67E22,stroke-width:2px,color:#784212,font-size:24px,font-weight:bold,padding:15px
  classDef index fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px,color:#922B21,font-size:24px,font-weight:bold,padding:15px

  subgraph Sources ["Source Datasets"]
    direction LR
    A((ChatDoctor<br>HealthCareMagic)):::source
    B((PubMedQA)):::source
    C((MedMCQA)):::source
  end

  subgraph PreProcess ["Pre-processing (src/data_pipeline/)"]
    direction TB
    D[Format Normalisation]:::stage
    E[Text Cleaning]:::stage
    F[RecursiveSentenceChunker<br>512-token target]:::stage
  end

  subgraph Embedding ["Embedding"]
    direction TB
    G[all-MiniLM-L6-v2<br>SentenceTransformer]:::embed
  end

  subgraph Storage ["Knowledge Base"]
    direction LR
    H[(ChromaDB<br>Vector Store)]:::storage
    I[(BM25 Index<br>Pickle Cache)]:::storage
  end

  subgraph Indexing ["Index Construction"]
    direction TB
    J[BM25 Index Build<br>rank-bm25]:::index
  end

  %% Links
  A & B & C --> D
  D --> E --> F
  F -- plain text chunks --> G
  G -- upsert vectors & metadata --> H
  F -. raw tokens .-> J
  J -- serialize --> I
"""

output_path = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/data_ingestion_mermaid_test"

with open(output_path + ".mmd", "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_path + ".mmd", "-o", output_path + ".png", "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])
print("Generated mermaid diagram.")
