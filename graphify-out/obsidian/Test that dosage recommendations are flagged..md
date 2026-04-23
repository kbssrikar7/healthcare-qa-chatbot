---
source_file: "/home/kbs/Documents/final_project/tests/test_safety.py"
type: "rationale"
community: "Community 2"
location: "L66"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_2
---

# Test that dosage recommendations are flagged.

## Connections
- [[.test_dangerous_dosage_blocked()]] - `rationale_for` [EXTRACTED]
- [[ContentFilter]] - `uses` [INFERRED]
- [[EmergencyDetector]] - `uses` [INFERRED]
- [[MedicalGuardrails]] - `uses` [INFERRED]
- [[SafetyLevel]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_2