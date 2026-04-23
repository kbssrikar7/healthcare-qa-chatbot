---
source_file: "/home/kbs/Documents/final_project/tests/test_safety.py"
type: "rationale"
community: "Community 2"
location: "L182"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# You have options' should not trigger diagnosis warning.

## Connections
- [[.test_no_false_positive_on_common_phrases()]] - `rationale_for` [EXTRACTED]
- [[ContentFilter]] - `uses` [INFERRED]
- [[EmergencyDetector]] - `uses` [INFERRED]
- [[MedicalGuardrails]] - `uses` [INFERRED]
- [[SafetyLevel]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2