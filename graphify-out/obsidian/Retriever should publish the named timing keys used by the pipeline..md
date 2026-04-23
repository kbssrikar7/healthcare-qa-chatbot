---
source_file: "/home/kbs/Documents/final_project/tests/test_local_rollout.py"
type: "rationale"
community: "Community 2"
location: "L60"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Retriever should publish the named timing keys used by the pipeline.

## Connections
- [[HybridRetriever]] - `uses` [INFERRED]
- [[MedicalEmbedder]] - `uses` [INFERRED]
- [[VectorStore]] - `uses` [INFERRED]
- [[test_retriever_exposes_deterministic_stage_timings()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_2