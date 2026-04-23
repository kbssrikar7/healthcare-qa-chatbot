import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/deployment_architecture_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/deployment_architecture_test.mmd"

mermaid_code = """
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
graph LR
    subgraph "Developer Workstation"
        Code["Source Code<br/>local repo"]
        Push["git push<br/>origin main"]
        Code --> Push
    end

    subgraph "GitHub Actions CI"
        Lint["Flake8 Lint + pytest<br/>250 tests"]
        Build["Docker Build<br/>& Layer Caching"]
        Lint --> Build
    end

    subgraph "Production Server (Docker Compose)"
        FastAPI["FastAPI Backend<br/>:8000 /ask /health"]
        Streamlit["Streamlit Frontend<br/>:8501 Primary UI"]
        Vol[("Docker Volume<br/>ChromaDB + BM25")]
        
        Streamlit -.->|HTTP API calls| FastAPI
        FastAPI -->|rw| Vol
    end

    subgraph "End User"
        Browser(("Web Browser<br/>localhost:8501"))
    end

    Push -->|Webhook Trigger| Lint
    Build -->|docker-compose up| FastAPI
    Browser -->|HTTP| Streamlit

    style Code fill:#f5f5f5
    style Push fill:#5d9cec
    style Lint fill:#ffce54
    style Build fill:#fc6e51
    style FastAPI fill:#4ecdc4
    style Streamlit fill:#4ecdc4
    style Vol fill:#ffce54
    style Browser fill:#ac92ec
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
