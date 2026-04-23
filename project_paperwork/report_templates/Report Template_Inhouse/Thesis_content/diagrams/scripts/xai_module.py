import subprocess
import os
import json

output_png = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/images/xai_module_final.png"
output_mmd = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/xai_module_test.mmd"

mermaid_code = """
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
graph TB
    subgraph "Confidence Scoring Path"
        S1["Retrieval Quality<br/>w=0.25 — RRF mean"]
        S2["Generation Confidence<br/>w=0.25 — logprob proxy"]
        S3["Self-Consistency<br/>w=0.20 — ROUGE sim"]
        S4["Source Agreement<br/>w=0.20 — BM25 overlap"]
        S5["Entity Coverage<br/>w=0.10 — med terms"]
        
        WC[Weighted Combination]
        Platt[Platt Scaling<br/>σ at + b]
        Conf((Calibrated Confidence<br/>∈ 0, 1))
        
        S1 --> WC
        S2 --> WC
        S3 --> WC
        S4 --> WC
        S5 --> WC
        WC --> Platt
        Platt --> Conf
    end

    subgraph "Hallucination Detection Path"
        Ans["Generated Answer"]
        Ctx["Retrieved Context"]

        NLI["DeBERTa NLI Hallucination Detector<br/>val: entailment / neutral / contradiction"]
        Flag(("Hallucination Flag<br/>if contradiction > 0.6"))

        Ans --> NLI
        Ctx --> NLI
        NLI --> Flag
    end

    style S1 fill:#a0d468
    style S2 fill:#a0d468
    style S3 fill:#a0d468
    style S4 fill:#a0d468
    style S5 fill:#a0d468
    style WC fill:#4ecdc4
    style Platt fill:#ffce54
    style Conf fill:#fc6e51
    
    style Ans fill:#5d9cec
    style Ctx fill:#5d9cec
    style NLI fill:#4ecdc4
    style Flag fill:#ed5565
"""

with open(output_mmd, "w") as f:
    f.write(mermaid_code)

puppeteer_config = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
with open("puppeteer-config.json", "w") as f:
    json.dump(puppeteer_config, f)

mmdc_path = "/home/kbs/Documents/final_project/node_modules/.bin/mmdc"
subprocess.run([mmdc_path, "-i", output_mmd, "-o", output_png, "-t", "default", "-b", "white", "-s", "3", "-p", "puppeteer-config.json"])

print(f"Written: {output_png}")
