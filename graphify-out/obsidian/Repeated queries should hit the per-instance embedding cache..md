---
source_file: "/home/kbs/Documents/final_project/tests/test_local_rollout.py"
type: "rationale"
community: "Community 2"
location: "L18"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Repeated queries should hit the per-instance embedding cache.

## Connections
- [[HybridRetriever]] - `uses` [INFERRED]
- [[MedicalEmbedder]] - `uses` [INFERRED]
- [[VectorStore]] - `uses` [INFERRED]
- [[test_query_embedding_cache_is_reused()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_2