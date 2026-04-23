---
source_file: "/home/kbs/Documents/final_project/src/pipeline/qa_pipeline.py"
type: "rationale"
community: "Community 2"
location: "L69"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Main pipeline orchestrating retrieval, generation, and XAI.      Stages (each sk

## Connections
- [[CacheManager]] - `uses` [INFERRED]
- [[HallucinationDetector]] - `uses` [INFERRED]
- [[HealthcareQAPipeline]] - `rationale_for` [EXTRACTED]
- [[MultiSignalConfidenceScorer]] - `uses` [INFERRED]
- [[RationaleGenerator]] - `uses` [INFERRED]
- [[RetrievedDocument]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2