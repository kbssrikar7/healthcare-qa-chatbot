---
source_file: "/home/kbs/Documents/final_project/tests/test_medication_grounding.py"
type: "code"
community: "Community 33"
location: "L44"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Community_33
---

# _mk_pipeline()

## Connections
- [[HealthcareQAPipeline]] - `calls` [INFERRED]
- [[_DummyLLM]] - `calls` [EXTRACTED]
- [[_DummyPromptManager]] - `calls` [EXTRACTED]
- [[_DummyRetriever]] - `calls` [EXTRACTED]
- [[test_medication_entity_verdict_rejects_unrelated_retrieval_context()]] - `calls` [EXTRACTED]
- [[test_medication_entity_verdict_rejects_wrong_drug_answer()]] - `calls` [EXTRACTED]
- [[test_medication_grounding.py]] - `contains` [EXTRACTED]
- [[test_prepare_retrieval_queries_keeps_brand_and_generic_for_dolo()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Community_33