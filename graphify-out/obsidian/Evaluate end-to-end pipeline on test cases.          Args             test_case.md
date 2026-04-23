---
source_file: "/home/kbs/Documents/final_project/evaluation/run_evaluation.py"
type: "rationale"
community: "Community 2"
location: "L425"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Evaluate end-to-end pipeline on test cases.          Args:             test_case

## Connections
- [[.evaluate_pipeline()]] - `rationale_for` [EXTRACTED]
- [[HealthcareQAPipeline]] - `uses` [INFERRED]
- [[HybridRetriever]] - `uses` [INFERRED]
- [[MedicalEmbedder]] - `uses` [INFERRED]
- [[MedicalLLM]] - `uses` [INFERRED]
- [[MedicalPromptManager]] - `uses` [INFERRED]
- [[VectorStore]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2