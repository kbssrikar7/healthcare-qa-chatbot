---
source_file: "/home/kbs/Documents/final_project/tests/test_safety.py"
type: "rationale"
community: "Community 2"
location: "L187"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# You have some choices' should not trigger diagnosis warning.

## Connections
- [[.test_no_false_positive_you_have_some()]] - `rationale_for` [EXTRACTED]
- [[ContentFilter]] - `uses` [INFERRED]
- [[EmergencyDetector]] - `uses` [INFERRED]
- [[MedicalGuardrails]] - `uses` [INFERRED]
- [[SafetyLevel]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2