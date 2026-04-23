---
source_file: "/home/kbs/Documents/final_project/src/pipeline/qa_pipeline.py"
type: "rationale"
community: "Community 2"
location: "L239"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Answer a medical question with explanations.          Full pipeline: enhance → r

## Connections
- [[.answer()_2]] - `rationale_for` [EXTRACTED]
- [[CacheManager]] - `uses` [INFERRED]
- [[HallucinationDetector]] - `uses` [INFERRED]
- [[MultiSignalConfidenceScorer]] - `uses` [INFERRED]
- [[RationaleGenerator]] - `uses` [INFERRED]
- [[RetrievedDocument]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2