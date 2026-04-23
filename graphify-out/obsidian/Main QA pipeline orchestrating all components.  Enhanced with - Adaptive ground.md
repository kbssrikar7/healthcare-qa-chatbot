---
source_file: "/home/kbs/Documents/final_project/src/pipeline/qa_pipeline.py"
type: "rationale"
community: "Community 2"
location: "L1"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Main QA pipeline orchestrating all components.  Enhanced with: - Adaptive ground

## Connections
- [[CacheManager]] - `uses` [INFERRED]
- [[HallucinationDetector]] - `uses` [INFERRED]
- [[MultiSignalConfidenceScorer]] - `uses` [INFERRED]
- [[RationaleGenerator]] - `uses` [INFERRED]
- [[RetrievedDocument]] - `uses` [INFERRED]
- [[qa_pipeline.py]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_2