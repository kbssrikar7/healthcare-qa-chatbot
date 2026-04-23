---
source_file: "/home/kbs/Documents/final_project/tests/test_safety.py"
type: "rationale"
community: "Community 2"
location: "L168"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Cyrillic lookalikes should be normalized and still caught if dangerous.

## Connections
- [[.test_unicode_bypass_input()]] - `rationale_for` [EXTRACTED]
- [[ContentFilter]] - `uses` [INFERRED]
- [[EmergencyDetector]] - `uses` [INFERRED]
- [[MedicalGuardrails]] - `uses` [INFERRED]
- [[SafetyLevel]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2