import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images_gemini/deployment_architecture_modern.png"
output_mmd = "/home/kbs/Documents/final_project/scratch/diagrams/deployment_architecture.mmd"

mermaid_code = """
graph LR
    subgraph "Development & CI"
        Push[Git Push to Main]
        GA[GitHub Actions CI<br/>Lint + Pytest]
        Build[Docker Build<br/>Multistage]
    end

    subgraph "Docker Compose (Localhost Runtime)"
        subgraph "Application Layer"
            NextJS["<b>Next.js Frontend</b><br/>Node.js Container<br/>Port 3000"]
            FastAPI["<b>FastAPI Backend</b><br/>Python Container<br/>Port 8000"]
        end
        
        subgraph "Data Layer"
            Vol[("<b>Docker Volume</b><br/>ChromaDB Vectors<br/>BM25 Pickle Index")]
        end
        
        NextJS -->|HTTP API Proxy| FastAPI
        FastAPI -->|Persistent Storage| Vol
    end

    subgraph "User Interface"
        Browser(("User Browser<br/>localhost:3000"))
    end

    Push --> GA
    GA --> Build
    Build -->|Deploy| NextJS
    Build -->|Deploy| FastAPI
    Browser -->|Access UI| NextJS

    %% Styling
    style Push fill:#f5f5f5,stroke:#333
    style GA fill:#5d9cec,stroke:#333,color:#fff
    style Build fill:#5d9cec,stroke:#333,color:#fff
    style NextJS fill:#4ecdc4,stroke:#333,color:#000
    style FastAPI fill:#ffce54,stroke:#333,color:#000
    style Vol fill:#ac92ec,stroke:#333,color:#fff
    style Browser fill:#a0d468,stroke:#333,color:#000
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

# Reuse existing config
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
