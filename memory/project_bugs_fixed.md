---
name: Healthcare QA bugs fixed
description: Bugs found and fixed in the healthcare QA chatbot codebase
type: project
---
Bugs fixed during initial review (2026-03-29):

1. **Feedback buttons always showed warning** (streamlit_app.py) - `st.warning()` was outside `else` block, so it ran even on success. Fixed to `else: st.warning(...)`.

2. **load_in_4bit=True on CPU** (api/main.py _get_langchain_pipeline/_get_langgraph_pipeline) - Passed `load_in_4bit=adapter_path.exists()` which is True when adapter exists, but 4-bit quantization only works on GPU. Fixed to `adapter_path.exists() and torch.cuda.is_available()`.

3. **psutil missing from requirements.txt** - Used in /health endpoint but not listed. Added `psutil>=5.9.0`.

4. **HuggingFace Hub causing ~40s startup delay** - Without offline mode, transformers tries 5 retries to ping HF Hub even though all models are cached. Added `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` as `os.environ.setdefault` in api/main.py. Startup now takes ~3s instead of ~40s.

**Why:** These were discovered during first run of the project.
**How to apply:** Be aware these patterns are fixed; don't revert them.
