---
source_file: "/home/kbs/Documents/final_project/src/pipeline/qa_pipeline.py"
type: "rationale"
community: "Community 5"
location: "L800"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_5
---

# Adaptive grounding gate: doc is relevant if             score >= max(absolute_sc

## Connections
- [[._check_answerability()_1]] - `rationale_for` [EXTRACTED]
- [[CacheManager]] - `uses` [INFERRED]
- [[HallucinationDetector]] - `uses` [INFERRED]
- [[MultiSignalConfidenceScorer]] - `uses` [INFERRED]
- [[RationaleGenerator]] - `uses` [INFERRED]
- [[RetrievedDocument]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_5