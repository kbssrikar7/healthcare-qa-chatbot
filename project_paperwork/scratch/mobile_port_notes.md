# Mobile (Android) On-Device Port — Notes

Last updated: 2026-08-26

## STATUS: STAGES 0-2 COMPLETE — full on-device RAG working (retrieval + generation)

Scope: Android-first on-device replication of ExplainRAG/MediQuery, using Google AI
Edge Gallery as the reference app. Mobile-only base model swap — the desktop system
and paper (`paper.tex`) are unaffected and stay on TinyLlama/BioMistral. A curated,
derived knowledge-base subset is used for mobile, not the full 505,584-doc corpus.
See the full tradeoff analysis and staged architecture in the planning session record
at `/home/kbs/.claude/plans/yeah-while-doing-so-playful-cocke.md`.

---

## Decision Log

- **2026-08-25** — Chose **mobile-only model swap** over a whole-project swap. Reason:
  TinyLlama has no supported conversion path onto Google's on-device toolchain
  (MediaPipe LLM Inference API / LiteRT-LM support Gemma-family, Qwen2.5, Phi-4 Mini,
  DeepSeek-R1-Distill — not TinyLlama). A whole-project swap would also invalidate the
  paper's QLoRA adapter (`models/fine_tuned/medical_adapter/adapter_config.json` is
  TinyLlama-architecture-specific) and Platt calibration
  (`src/xai/multi_signal_confidence.py:105-106`, fitted on TinyLlama outputs only),
  requiring a full re-evaluation (`make eval`, `evaluation/run_ablation.py`,
  bootstrap CI, `evaluation/generate_baseline_table.py`) and reconciliation of every
  number already in `paper.tex` — a multi-day effort with no upside for the mobile
  goal.
- **2026-08-25** — Chose **Android-first**, iOS deferred until the Android
  architecture is validated. Reason: the reference app and the recommended runtime
  (LiteRT-LM) are Android-first; targeting both platforms in v1 doubles
  conversion/build surface for no validated architecture to fall back on.
- **2026-08-25** — This session scoped to **plan + notes scaffold only**. Actual
  mobile development (Android project scaffolding, model conversion, KB-subset
  extraction script, etc.) is deferred to a later session.
- **2026-08-25** — Test device connected via adb: Samsung Galaxy S9+ (SM-G965F),
  Android 10 (API 29), ~5.7GB RAM, arm64-v8a. Verified `com.google.ai.edge.litertlm:
  litertlm-android:0.11.0`'s own AndroidManifest declares `minSdkVersion 23` (checked
  by downloading the AAR from Google's Maven and inspecting it directly) — the Gallery
  app's own minSdk=31 is driven by unrelated features (ML Kit GenAI Prompt, newer
  notification APIs, etc.), not by the core inference library. This device is
  compatible with on-device LiteRT-LM inference.
- **2026-08-25** — Scaffolded a minimal from-scratch Android app at `android/` (repo
  root, sibling to `api/` and `frontend-react/`) rather than stripping down the full
  Gallery app — package `com.mediquery.mobile`, minSdk 26 / compileSdk 34 / targetSdk
  34 (compileSdk pinned to 34 because that's the platform already installed locally at
  `~/Android/Sdk/platforms/android-34`; can be raised later if needed). Toolchain
  versions (AGP 8.13.0, Kotlin 2.2.0, Compose BOM 2026.02.00) copied directly from
  Gallery's own `libs.versions.toml` for known compatibility with `litertlm-android`.
  Gradle wrapper jar/scripts copied from the Gallery repo (generic bootstrap code,
  Apache-2.0).
- **2026-08-25** — Verified all LiteRT-LM Kotlin API calls (`Engine`, `EngineConfig`,
  `Conversation`, `ConversationConfig`, `Contents.of(String)`, `MessageCallback`,
  `Backend.CPU()`, `SamplerConfig`) directly against the library's compiled bytecode
  (`javap` on the AAR's `classes.jar`, downloaded from `dl.google.com`) rather than
  guessing from Gallery's source alone — confirmed exact constructor signatures before
  writing `MainActivity.kt`.
- **2026-08-25** — **Blocker (needs your action): the Gemma3-1B-IT model repo on
  Hugging Face (`litert-community/Gemma3-1B-IT`) is gated** — Google's Gemma license
  must be accepted via a logged-in HF account before the `.task` file can be
  downloaded. This can't be done on your behalf (credential/account action). To
  unblock: (1) log into/create a Hugging Face account, (2) visit
  https://huggingface.co/litert-community/Gemma3-1B-IT and accept the license, (3)
  either download `Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task` (~529MB) yourself
  via browser and tell me the local file path, or generate a read-only HF access token
  yourself and run the download in your own shell (`! huggingface-cli download ...`)
  rather than pasting the token into chat.
- **2026-08-26** — First on-device run verified. Built with `./gradlew :app:assembleDebug`
  (two build fixes needed: `android.useAndroidX=true` in `gradle.properties`, missing
  by default; and `compileSdk`/`targetSdk` raised from 34 to 36 — the Compose BOM
  2026.02.00 dependency tree requires compileSdk 35+, only android-34 was installed
  locally, installed android-36 via `sdkmanager`). Installed via `adb install -r` and
  launched via `adb shell am start` on the Galaxy S9+ test device — confirmed via
  screenshot: UI renders correctly (title, status text, Load model button, query
  field, Send button), no crash in logcat, and the "model file not found" error path
  (tapping Load model with no `.task` file pushed yet) displays exactly the expected
  message. **Stage 0's toolchain is now proven end-to-end on the real device** —
  Compose + `litertlm-android` linking + packaging + install + launch all work.
  Remaining for Stage 0: push a real `.task` file and confirm actual inference runs
  (blocked on the gated-download item above) and confirm the `message.toString()`
  streaming callback is incremental vs. cumulative (code currently assumes
  incremental — `answer += token` — easy one-line fix if wrong, not yet verified
  since no model has run yet).
- **2026-08-26** — **Stage 0 fully validated with real on-device inference.** User
  downloaded `Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task` via their phone's own
  browser (HF account + license acceptance done on-device) — it landed in
  `/sdcard/Download/` on the phone directly, so no desktop `adb push` was even needed;
  copied on-device via `adb shell cp` into `/data/local/tmp/gemma3-1b-it.task`.
  Tapped "Load model" — took ~2 minutes to reach "Model ready." (TFLite/XNNPACK
  delegate loaded 2230/2298 nodes, `Gemma3DataProcessor` created, `Backend.CPU()`
  registered — NPU accelerator load failed as expected on this chipset, GPU wasn't
  attempted since the code explicitly requests CPU). Asked "What are the symptoms of
  Type 2 diabetes?" — got a complete, coherent, well-structured markdown-formatted
  answer (bolded headers, bullet list: increased thirst, increased urination,
  weight gain/difficulty losing weight, fatigue) in **~3 minutes** end-to-end
  (CPU-only, no GPU delegation attempted). No crash, no OOM, on a device with only
  ~5.7GB RAM against the model's ~2GB peak-memory spec.
  **Confirmed `message.toString()` in `MessageCallback.onMessage` is incremental**
  (not cumulative) — the answer rendered as one clean, non-duplicated block, which
  resolves the open question from the first entry above; the `answer += token` code
  as written is correct.
  **Known rough edges to fix before this goes further, not blockers:** (1) markdown
  syntax (`**bold**`, `*` bullets) renders as literal characters — the UI has no
  markdown renderer yet, purely cosmetic; (2) ~3 minutes CPU-only generation is slow
  — worth trying `Backend.GPU()` next given this device's Mali GPU was detected
  (`libGLES_mali.so` loaded) but not used, before concluding CPU-only is the realistic
  number for this hardware class.
  **Bottom line: the Stage 0 spike goal — prove the LiteRT-LM/Gemma toolchain runs
  end-to-end on real (old) hardware — is achieved.** Next real milestone is Stage 1
  (KB subset) + Stage 2 (retrieval rebuild), i.e. actually wiring retrieval into this
  app, not just standalone chat.
- **2026-08-26** — Fixed markdown rendering: added `com.halilibo.compose-richtext`
  (`richtext-commonmark` + `richtext-ui-material3`, both `1.0.0-alpha02` — same library
  Gallery itself uses, see `Android/src/app/.../ui/common/MarkdownText.kt` in the
  Gallery clone) and a `MarkdownText` composable, replacing the plain `Text` for the
  answer. Verified building cleanly before moving on to Stage 1/2.
- **2026-08-26** — **Stage 1 (KB subset) built, with one real bug caught before it
  did damage.** First attempt at "one hop of neighbor chunks" (chunk_id±1 within the
  same `source`) assumed `source` metadata identifies one original document — it
  doesn't. `source` is a **dataset-level** label (e.g. "MedQuAD"), reused across
  thousands of original documents, and `chunk_id` only resets to 0 within one
  original document. So a `(source, chunk_id)` lookup matched thousands of unrelated
  documents that happened to share a chunk position, and the "subset" build was
  silently writing out ~490,000 records (basically the whole 505k KB) before being
  killed mid-run. Root cause: no stable per-document ID exists in the current KB
  metadata schema, so true same-document neighbor expansion isn't safely possible
  without deeper data engineering (worth flagging to the desktop team as a metadata
  gap if this becomes recurring need — a stable `doc_id` field, distinct from
  `source`, would fix it). **Fix applied: dropped neighbor expansion entirely**,
  shipped the direct top-15-per-question set only. Final result:
  `evaluation/build_mobile_kb_subset.py` → **1,444 unique chunks, 6.6MB**
  (`evaluation/mobile_kb_subset.jsonl`), built from real `vs.search()` calls (the
  same dense-retrieval code path the desktop system uses) against the live 97-question
  eval set and the full 505,584-doc `data/knowledge_base_v2` ChromaDB.
- **2026-08-26** — **Stage 2 (on-device retrieval) built and verified working,
  end-to-end, with real RAG-grounded generation.** Departed from the original plan's
  "SQLite + sqlite-vec + FTS5" recommendation — see Retrieval Architecture section
  below for why — in favor of pure in-memory brute-force retrieval, appropriate at
  this KB's actual size (1,444 chunks, not 500k+). Built:
  - `retrieval/WordPieceTokenizer.kt` — hand-written BERT WordPiece tokenizer (no
    external tokenizer library), verified against the real HF tokenizer config
    (`do_lower_case: true`, standard vocab.txt) rather than assumed.
  - `retrieval/OnnxEmbedder.kt` — runs `sentence-transformers/all-MiniLM-L6-v2`'s
    **official arm64 int8-quantized ONNX export** (`onnx/model_qint8_arm64.onnx` from
    the model's own HF repo — non-gated, ~22MB, bundled as an app asset, not pushed)
    via `com.microsoft.onnxruntime:onnxruntime-android:1.29.0` (minSdk 24, verified by
    downloading and inspecting the AAR manifest same as litertlm earlier). Graph
    inputs/outputs (`input_ids`/`attention_mask`/`token_type_ids` → `last_hidden_state`
    [1, 256, 384]) were inspected directly via the `onnx` Python package before writing
    any Kotlin, not assumed. Pooling is masked mean over tokens + L2 normalize,
    confirmed against the model's own `1_Pooling/config.json`
    (`pooling_mode_mean_tokens: true`) and `src/embeddings/embedding_models.py`
    (`normalize_embeddings=True` is the desktop default) — this is a faithful match to
    desktop embedding behavior, not an approximation.
  - `retrieval/HybridRetriever.kt` — ports `src/retrieval/hybrid_retriever.py`'s query
    classification regexes, adaptive weight table, `reciprocal_rank_fusion()` formula,
    and — most fiddly — `rank_bm25.BM25Okapi`'s *exact* scoring formula including its
    negative-IDF epsilon-clipping behavior (`epsilon=0.25` default), not a simplified
    BM25 variant. Dense side is brute-force cosine similarity over the 1,444
    in-memory embeddings (trivially fast at this size — no ANN index needed).
  - `retrieval/KnowledgeBase.kt` — loads the JSONL subset, pushed via
    `adb push mobile_kb_subset.jsonl /data/local/tmp/mobile_kb.jsonl` (same
    adb-push-to-`/data/local/tmp` convention as the LLM model file).
  - `MainActivity.kt` updated: loads embedder+KB alongside the LLM, retrieves top-3
    chunks per query, builds a RAG prompt ("Answer using ONLY the context..."), and
    displays a "Sources (N): ..." line with real retrieved dataset names.
  - **First real bug found by actually running it, not by review:** initial run with
    `maxNumTokens = 1024` (the model_allowlist.json default, which is a *generation*
    length default, not the real context window) failed with "Input token ids are too
    long... 1472 >= 1024" once RAG context was included. Fixed by raising
    `maxNumTokens` to 2048, matching the model file's own name
    (`..._ekv2048` = 2048-token KV cache).
  - **Second real quality issue found by actually running it:** a hard 400-character
    truncation per retrieved passage (needed to keep prompt size sane) produced
    visibly garbled, hallucination-prone output — the model was given dangling
    mid-sentence fragments as "context" and partly hallucinated a second unasked
    question. Fixed with `truncateAtSentence()` (truncate at the last sentence-ending
    punctuation within the char budget, falling back to a word boundary), and bumped
    the budget to 600 chars/passage now that the context window is 2048. Retested:
    clean, coherent, non-garbled output.
  - **Verified end-to-end on-device with a real query** ("What are the symptoms of
    Type 2 diabetes?"): retrieval correctly found 3 real sources (MedicalMeadow/
    meadow_mediqa, ChatDoctor, HealthCareMagic — visible in the app's "Sources" line),
    and generation produced a coherent, well-structured, markdown-rendered answer
    grounded in that retrieved context. Answer content quality note (honest, not
    swept under the rug): the listed symptoms (foot pain/tingling, muscle spasms,
    fatty deposits, blurred vision, increased urination, night sweats) are a mix of
    plausible and slightly off-textbook — likely because the 1,444-chunk mobile
    subset's top hits for this query lean on patient-forum-style sources
    (ChatDoctor/HealthCareMagic Q&A) rather than the cleaner MedQuAD-style textbook
    entries the desktop's full 505k corpus would also surface. This is exactly the
    "mobile-subset vs. desktop-on-same-subset" comparison the plan called for — worth
    running that comparison formally (Results section below) before drawing
    conclusions about retrieval quality loss.
  - **Bottom line: Stage 1 and Stage 2 goals are both achieved.** This is now a real,
    working, fully-offline, on-device RAG system — not a demo of the LLM alone. Next
    real milestones: (1) the formal desktop-vs-mobile retrieval/latency/confidence
    comparison in the Results section below, (2) trying `Backend.GPU()` for
    generation speed (still untested — everything so far has run CPU-only), (3)
    deciding whether/how to port the XAI confidence layer (see that section below,
    still unstarted).
- **2026-08-26 — Tried `Backend.GPU()` for the LLM engine. Massive win, changes the
  viability picture for this hardware class.** One-line change
  (`Backend.CPU()` → `Backend.GPU()` in `EngineConfig`), same model file, same RAG
  pipeline, same question re-run for direct comparison. Confirmed via logcat this is
  a real GPU path, not a silent no-op fallback: `LITERT_CL` (OpenCL) delegate compiled
  1330-1373 node subgraphs (`Replacing N out of N node(s) with delegate (LITERT_CL)`),
  using this device's Mali GPU (`gpu_environment.cc: Created OpenCL device`).
  - **Model load: ~29s** (07:46:02 tap → ready by 07:46:31), down from ~2 minutes on
    CPU.
  - **Generation: ~18s** (07:46:59 send → complete by 07:47:17, confirmed via
    Send-button state), down from ~3-6 minutes on CPU for the same question with the
    same retrieved context — **roughly a 10-20x speedup**.
  - Output quality: same 3 sources retrieved (MedicalMeadow/meadow_mediqa,
    ChatDoctor, HealthCareMagic — retrieval is CPU-side and unaffected by this
    change), clean coherent bullet-point answer, no degradation observed from the
    CPU run.
  - **This matters beyond "it's faster":** ~18s end-to-end on a 2018 mid-range phone
    with no NPU is in the range of being genuinely *usable*, not just a proof of
    concept — CPU-only's 3-6 minutes was a technical success but not something a real
    user would tolerate. Worth leading with GPU as the default backend recommendation
    from here on, not CPU, and worth testing on a couple of newer/different-GPU
    devices later to see how this scales (this is one data point on one Mali GPU).
  - Not yet tested (at time of writing): GPU backend under thermal throttling over a
    longer session, and whether `Backend.GPU()` for the *embedder* (ONNX Runtime has
    its own separate GPU/NNAPI execution providers, untouched here — `OnnxEmbedder`
    still runs CPU-only) would meaningfully speed up retrieval, though retrieval
    latency was already not the bottleneck (embedding+BM25+RRF over 1,444 chunks is
    sub-second-scale regardless).
- **2026-08-26 — Ran 3 consecutive GPU-backed queries of different types (symptom,
  drug, comparison) to check thermal/consistency behavior — no throttling, no quality
  degradation, got faster with each run.** Methodology: fresh app relaunch + model
  reload before each question (found that clearing the text field via adb DEL
  keyevents was unreliable — relaunching is simpler and, since GPU load is fast, cheap
  enough to just do per-question). Checked `dumpsys thermalservice` (`TYPE_CPU`,
  `TYPE_BATTERY`) before/after.
  - **Q1** "What are the symptoms of Type 2 diabetes?" (symptom type, cold model
    load): ~29s load + ~18s generation (this is the run reported above).
  - **Q2** "What is metformin used for?" (drug type, warm reload — page cache from
    Q1's model load): **~13s load** + generation **done within the first 3s poll
    interval** (i.e. well under 15s, could not resolve more precisely with this
    polling method). Answer: *"Metformin can be used for managing type-2 diabetes
    mellitus by lowering blood glucose levels and improving insulin sensitivity."* —
    accurate, correctly grounded. Sources: MedicalMeadow/meadow_medqa, ChatDoctor,
    MedMCQA/Surgery.
  - **Q3** "What is the difference between Type 1 and Type 2 diabetes?" (comparison
    type, warm reload): **~3s load** + generation again resolved within the first
    poll (well under 13s). Answer: *"Type 1 diabetes involves an autoimmune
    destruction of beta cells mediated by T cells and humoral mediators. Type 2
    diabetes involves an insulin resistance that can be caused by genetic factors or
    other factors that do not involve an autoimmune destruction."* — accurate,
    textbook-quality, correctly grounded (retrieved MedMCQA/Medicine, an exam-style
    source, alongside MedicalMeadow and ChatDoctor).
  - **Model-load time trend across the session: ~29s → ~13s → ~3s.** Not GPU-specific
    — almost certainly OS page-cache warmup on the 529MB `.task` file across repeated
    loads within the same session, not a GPU characteristic. Worth knowing when citing
    load-time numbers: "cold" vs "warm" load matters a lot and wasn't controlled for
    across these three runs.
  - **Thermal:** CPU temp 38.8°C (baseline, before any query) → 40.9°C (after Q1+Q2)
    → 42.5°C (after Q3). Battery temp 36.7°C → 37.9°C → 38.1°C. Android's
    `mStatus` field (throttling indicator) stayed `0` (none) throughout — no
    throttling triggered across this session.
  - **Caveat: device was on USB power throughout (battery `level=100`,
    `status=5`/charging/full)** — this does not test real unplugged, battery-only
    thermal or performance behavior, which could differ (charging generally keeps a
    phone's power delivery less constrained than running on battery alone). A future
    test unplugged, and/or a longer session (10+ consecutive queries) rather than 3,
    would be needed before making a strong claim about sustained real-world usability.
  - **Retrieval correctness cross-check:** all 3 query types pulled genuinely
    different, topically-appropriate sources (not the same generic top-3 every time),
    which is a good sign the on-device query-type classification + adaptive RRF
    weighting (`HybridRetriever.kt`) is actually functioning, not just always falling
    back to one default behavior.
  - **Bottom line: the GPU speedup is real, consistent across query types, and not
    accompanied by throttling or quality loss in this short session.** Reasonable to
    treat GPU as the practical default for this device class, with the unplugged/
    longer-session caveat above still open.

---

## Model Choice

- **Runtime:** LiteRT-LM (successor to the MediaPipe LLM Inference API, which is in
  maintenance mode). Same toolchain lineage as Google AI Edge Gallery.
- **Primary model:** Gemma 3 1B.
- **Fallback/low-end tier:** Gemma 3 270M — quality on medical QA not yet validated;
  needs a spike test before committing beyond the 1B model.
- **Quantization / conversion tooling version:** pre-converted by Google —
  `litert-community/Gemma3-1B-IT`, file
  `Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task` (int4, 2048-token KV cache).
- **On-device size (after conversion):** 554,661,246 bytes (~529MB) — confirmed exact
  match between the HF-hosted file size and the on-device pushed file.
- **Embedder model (for query encoding, separate from the generation model above):**
  `sentence-transformers/all-MiniLM-L6-v2`, official arm64 int8-quantized ONNX export
  (`onnx/model_qint8_arm64.onnx` from the model's own HF repo, non-gated), ~22MB,
  bundled as an app asset (`minilm.onnx` + `minilm_vocab.txt`), run via
  `onnxruntime-android:1.29.0`. See Stage 2 decision-log entry above for the full
  verification trail.
- **Alternative considered and rejected for v1:** llama.cpp mobile bindings running
  the existing GGUF BioMistral or a TinyLlama GGUF conversion — would preserve "same
  model family as the paper" but forfeits Google AI Edge Gallery toolchain parity,
  which was the reason for choosing LiteRT-LM in the first place. Held in reserve as
  a fallback if Gemma-family quality proves inadequate.

---

## KB Subset Construction

- **Method:** derive from the existing 97-question eval set
  (`evaluation/test_set_v2.json`) — union of each question's top-15 retrieved docs
  from the desktop system's real dense-retrieval code path (`vector_store.search()`,
  same call `hybrid_retriever.py`'s `_dense_retrieve()` uses). Not hand-picked.
  **Neighbor-chunk expansion was planned but dropped** — see the Stage 1 decision-log
  entry above for why (`source` metadata is dataset-level, not per-document; no safe
  way to find "the same document's neighbor chunk" with the current metadata schema).
- **Actual size: 1,444 unique chunks, 6.6MB** (`evaluation/mobile_kb_subset.jsonl`).
- **Source doc IDs / provenance:** each record carries its original ChromaDB id,
  `source` (dataset label, e.g. MedQuAD/ChatDoctor/MedicalMeadow), and `chunk_id`. The
  script itself is `evaluation/build_mobile_kb_subset.py` — rerunning it against the
  same `data/knowledge_base_v2` + `test_set_v2.json` is fully reproducible.
- **Date built:** 2026-08-26. **Version hash:** not yet tagged — worth adding a
  content hash to the output filename if the KB is rebuilt again later, to keep
  mobile results traceable to a specific subset version.
- **Framing reminder for any future writeup:** this is a different (smaller) corpus
  than the 505,584-doc production KB — always report mobile retrieval as
  "mobile-subset vs. desktop-on-same-subset," never "mobile vs. full production KB."

---

## Retrieval Architecture

- **⚠️ Deviated from the original plan's recommendation.** The plan (written before
  the KB subset's actual size was known) recommended SQLite + sqlite-vec + FTS5.
  Once the subset came in at 1,444 chunks — not the full 505k corpus — that
  infrastructure stopped being worth its cost: sqlite-vec's Android support is young
  and unverified, and stock Android's `SQLiteDatabase.loadExtension()` isn't even
  available before API 34 (our test device is API 29), which would have forced a
  custom SQLite build (e.g. requery/sqlite-android) just to load a native extension.
  **Actual implementation: pure in-memory brute-force retrieval in Kotlin** — a
  `List<KbChunk>` loaded once at startup, cosine similarity computed in a loop for
  dense (a few thousand dot products, sub-millisecond on any modern-ish CPU), and a
  hand-rolled BM25Okapi port for sparse. Zero native/extension dependencies, works on
  any API level our `litertlm-android` dependency already requires (23+). This is the
  right call at this KB size — revisit only if the KB subset grows to tens of
  thousands+ chunks where brute-force stops being fast enough.
- **RRF fusion + adaptive weight table:** ported to Kotlin in
  `retrieval/HybridRetriever.kt` — same query-type regexes, same weight table (drug
  0.45/0.55, definition 0.80/0.20, symptom 0.65/0.35, comparison 0.80/0.20, default
  0.70/0.30), same RRF formula (`k=60`, rank starts at 1) as
  `src/retrieval/hybrid_retriever.py`. **Verified working** via the "Sources (3): ..."
  line shown in the app after a real query.
  - **Correction to the plan's stated assumption:** the plan said FTS5's BM25 defaults
    would differ numerically from `rank_bm25`'s but that RRF fusing ranks (not raw
    scores) would absorb it. Since FTS5 wasn't used at all, this is moot — but the
    same principle applied anyway: the hand-rolled BM25Okapi port matches
    `rank_bm25`'s formula (including its `epsilon=0.25` negative-IDF clipping)
    *exactly*, so there's no scoring-scheme mismatch to absorb in the first place at
    this implementation.
- **Embedding parity check** (recommended before trusting any mobile retrieval
  number, not yet done formally — the qualitative check so far is "the retrieved
  sources look topically relevant to the query," not a measured cosine-similarity
  delta against desktop):

  | Metric | Desktop | Mobile (converted) | Delta |
  |---|---|---|---|
  | Cosine similarity (fixed probe query set) | — | — | — |
  | Recall@k (desktop-on-subset vs mobile-on-subset) | — | — | — |

---

## XAI / Confidence Layer

- **Signal set:** same 5-signal composite as desktop (retrieval, source_agreement,
  generation, consistency, entity_coverage), with a new `MOBILE_WEIGHTS` dict mirroring
  the existing `OLLAMA_WEIGHTS` pattern (`multi_signal_confidence.py:113-119`).
  Verified in code: this pattern does not touch NLI/hallucination detection (that's a
  separate downstream flag, not one of the 5 weighted signals), so no special-case
  renormalization is needed.
- **Weights:** *[not yet set — to be tuned once mobile signals can actually be
  computed]*
- **Platt calibration:** skipped for mobile, mirroring the existing Ollama/OpenRouter
  precedent (Platt is documented as TinyLlama-specific).
- **Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90MB):** dropped for v1.
- **NLI hallucination detector (`microsoft/deberta-base-mnli`, ~550MB):** dropped for
  v1. Precedent: already optional/lazy-loaded on desktop, already skipped for
  Ollama/OpenRouter. Revisit only if retrieval-without-reranking underperforms badly
  on the eval subset.

---

## Measurement Environment

Required for every on-device measurement run — latency/quality numbers without this
recorded are not usable.

- Device model: Samsung Galaxy S9+ (SM-G965F), 2018-era, Snapdragon 845/Exynos 9810,
  no dedicated NPU
- Chipset / NPU or GPU delegate used: **both tried.** `Backend.CPU()` initially, then
  switched to `Backend.GPU()` (this device's Mali GPU via OpenCL/`LITERT_CL`
  delegate) — see the 2026-08-26 GPU decision-log entry above. GPU is now the
  configured backend in `MainActivity.kt`.
- RAM: ~5.7GB total (`MemTotal` from `/proc/meminfo`)
- Android OS version: 10 (API 29)
- Thermal state at measurement time: *[not recorded — worth checking
  `dumpsys thermalservice` on a future run, especially for any GPU-backend test]*
- Battery / charging state: *[not recorded]*
- Date of measurement: 2026-08-26

---

## Results

*(Empty until real runs happen — do not fill in speculatively.)*

### Retrieval quality (mobile-subset vs. desktop-on-same-subset)

| Query type | Desktop MRR@5 | Mobile MRR@5 | Desktop NDCG@5 | Mobile NDCG@5 |
|---|---|---|---|---|
| Drug | — | — | — | — |
| Definition | — | — | — | — |
| Symptom | — | — | — | — |
| Overall | — | — | — | — |

### Latency

Mobile numbers are single-run wall-clock observations via manual adb timestamping
(tap → screen-state change), not an averaged/instrumented benchmark — good enough to
show the CPU→GPU magnitude shift, not precise enough to cite as a formal number.
Desktop column is the paper's reported warm-path mean (`paper.tex`, TinyLlama, CPU).

| Stage | Desktop (CPU, warm, TinyLlama) | Mobile CPU (Gemma3-1B) | Mobile GPU (Gemma3-1B) |
|---|---|---|---|
| Model load | — (loaded once at startup) | ~120s | ~29s |
| Retrieval | 43s (505k-doc corpus) | sub-second (1,444-chunk subset, not comparable corpus size) | sub-second |
| Generation | 70s | ~180-360s (est., not precisely timed) | ~18s |
| Total (per query, warm) | 106s | ~3-6 min | ~18s |

### Confidence / calibration

| Metric | Desktop (TinyLlama) | Mobile (Gemma 3 1B) |
|---|---|---|
| ECE | 0.14 (paper.tex) | — |
| Mean confidence | 0.025 (paper.tex) | — |

---

## UI / Visual Design

- **2026-08-26 — Full UI overhaul.** The first working spike (Stage 0-2) used a
  single-turn, default-Material-purple layout — functionally proven but visually
  generic. Replaced with:
  - **Real brand palette**, extracted directly from the desktop app rather than
    invented: primary blue `#1D4ED8` and pulse-accent `#93C5FD` from
    `public/mediquery-icon.svg`; dark navy background `#0F172A` and near-white
    foreground `#F8FAFC` converted from `frontend-react/app/globals.css`'s HSL
    custom properties (`--background: 222.2 47.4% 11.2%`, etc. — desktop forces this
    dark palette regardless of system theme, matched here for consistency).
    `ui/theme/Color.kt` + `Theme.kt`.
  - **Real app icon**, not a placeholder — Android adaptive icon
    (`res/drawable/ic_launcher_{background,foreground}.xml` +
    `res/mipmap-anydpi-v26/ic_launcher.xml`) built by scaling
    `public/mediquery-icon.svg`'s 48x48 cross+pulse-line design 2.25x onto the
    108dp adaptive-icon canvas, preserving the original's proportions. Verified
    rendering correctly (masked into the launcher's circle shape) via the device's
    App Info screen.
  - **Multi-turn chat transcript** instead of single-question/single-answer —
    `LazyColumn` of `ChatTurn` items, auto-scrolls to the latest message. Model
    loading now starts automatically on launch (no manual "Load model" tap needed).
  - **Evaluated and rejected Stream Chat SDK** (user's suggestion) as a UI source —
    its open-source Compose components are real but tightly coupled to Stream's own
    `ChatClient`/`Channel`/`Message` backend model; adopting it would mean fighting
    their architecture to wire in a local LLM instead of their hosted backend.
  - **Selectively adapted (not wholesale-copied) pieces of Google AI Edge Gallery's
    own chat UI**, which already exists for this exact use case (LiteRT-LM streaming
    chat) but is too tightly coupled to Gallery's broader app (image/audio input,
    benchmark mode, model downloads, prompt templates — `ChatPanel`/`ChatView`/
    `ChatViewModel`/`MessageInputText` together are ~3,900 lines) to import
    wholesale without dragging that complexity back in. Took two small,
    self-contained, brand-agnostic pieces instead (both Apache-2.0, attribution
    comments in the files):
    - `ui/MessageBubbleShape.kt` — ported near-verbatim. Custom `Shape` giving chat
      bubbles one sharp "tail" corner instead of a uniform rounded rectangle.
    - `ui/LongPressCopyContainer.kt` — ported and simplified (fixed "Copy" label
      instead of a string resource, inlined clipboard write). Long-press any
      message to copy its text.
    - **Adopted Gallery's more important structural decision, not just its
      code**: agent (assistant) text responses get **no bubble at all** — plain
      full-width markdown text. Gallery's own source has an explicit
      `isAgentResponseText` check that disables the bubble specifically for this
      case. This fixed a real problem with the first version (long markdown
      answers cramped into a fixed-max-width bubble); only the user's own message
      still gets the blue bubble treatment.
  - **Custom "thinking" indicator** (three dots with a staggered breathing-fade
    animation, brand-colored) replacing the plain `CircularProgressIndicator` — this
    one was *not* lifted from Gallery (their `RotationalLoader` depends on 4 custom
    drawable assets and Gallery's own multi-color gradient theme, which would look
    off-brand here); reimplemented the same animation *pattern* with MediQuery's
    palette instead.
  - **Source chips** — retrieved sources now render as small rounded pill chips
    (deduplicated by source name) instead of a plain comma-separated text line.
  - Verified on-device: icon renders correctly, chat bubbles/tail-corner/no-bubble
    convention all render as designed, thinking animation and source chips both
    confirmed working via a live query ("What is metformin used for?").
  - `app_name` changed from "MediQuery Mobile Spike" to "MediQuery" to match the
    in-app header and read cleanly in system UI (launcher, App Info, recents).

---

## Play Store Readiness

- **2026-08-26 — Prepped everything short of what requires the user's own Google
  account/payment.** Two hard blockers flagged to the user up front: (1) only they
  can create the Play Console developer account (one-time $25 fee, tied to their own
  identity/payment — cannot be done on their behalf); (2) this is a health-information
  app built on an LLM, and Google Play's Health Content + AI-generated-content
  disclosure policies mean real review scrutiny should be expected, not assumed
  smooth approval.
- **Signing:** generated `android/mediquery-release.keystore` (RSA 2048,
  10,000-day validity) — **this uses a hardcoded placeholder password
  (`mediquery-dev-2026`) and exists only to verify the release-build pipeline
  works end-to-end. Do not use this keystore for the real Play Store submission.**
  Before real submission: either generate a fresh keystore with a strong,
  privately-stored password, or (recommended) use Play App Signing — upload an
  "upload key" (which can be reset if lost) and let Google hold the actual app
  signing key server-side, which avoids the catastrophic failure mode of losing a
  self-managed signing key entirely (an app's signing identity can't be rotated
  without users reinstalling from scratch). `signing.properties` (gitignored) holds
  the current placeholder credentials; `app/build.gradle.kts` reads from it if
  present.
- **Release build verified working:** `./gradlew :app:bundleRelease` produces a
  signed `app-release.aab` (104MB) — confirmed signed via `jarsigner -verify`
  ("jar verified"; the "certificate chain is invalid" warning is expected/harmless
  for a self-signed Android app-signing cert, not an error). Play Store requires
  `.aab` uploads, not the debug `.apk` used throughout Stage 0-2 testing.
- **Stripped the app's network permissions entirely — verified functionally.**
  `onnxruntime-android` bundles `INTERNET`/`ACCESS_NETWORK_STATE` in its own
  manifest for optional features (remote model fetch, telemetry) that this app
  never invokes — only local `createSession()` on a bundled asset is used.
  Explicitly removed both via `tools:node="remove"` in `AndroidManifest.xml`.
  Confirmed via `dumpsys package` that the shipped app requests no network
  permission at all (only an internal `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`,
  not a real data-access permission), **and re-verified full functionality
  afterward** (fresh install, model load, and a real query — "What are the
  symptoms of high blood pressure?" — correctly retrieved 3 sources and generated
  a grounded answer) to confirm stripping the permission didn't break anything.
  This makes "your data never leaves your phone" a literally true claim, not just
  a policy statement, and should make the Play Console Data Safety form about as
  simple as it can be ("no data collected").
- **Privacy policy drafted:** `android/store_listing/privacy_policy.md`. Written to
  match what the app *actually* does (checked against the manifest/code, not
  boilerplate) — no account, no network permission, no analytics/ads, no
  persistence (chat history is in-memory only, lost on app close). Has two
  `[FILL IN]` placeholders (publish date, support contact) — Play Console requires
  a hosted URL for this, which needs to be published somewhere (e.g. a page on the
  existing `mediquery-healthcare.vercel.app` domain) before submission; this repo
  can't host it.
- **Store listing copy drafted:** `android/store_listing/listing_copy.md` — short
  description (74/80 chars), full description (written carefully around Play's
  Health Content policy: "educational," explicit non-diagnosis/non-treatment
  disclaimer, matches the in-app disclaimer text and the privacy policy). Graphic
  assets (512x512 icon export, 1024x500 feature graphic, screenshots) are **not**
  produced yet — screenshots could reuse real in-app captures from this session's
  testing with light cleanup; the feature graphic needs actual design work not yet
  started.
- **2026-08-26 — Icon and feature graphic exported.** Both in `android/store_listing/`:
  - `play_store_icon_512.png` — 512x512, real alpha transparency outside the rounded
    shape (verified: corner pixel `srgba(0,0,0,0)`, not opaque white). Rendered from
    `public/mediquery-icon.svg` via ImageMagick with `-background none`.
  - `feature_graphic_1024x500.png` — 1024x500, no alpha (`Type: TrueColor`, required
    for Play's feature graphic slot), built from a new
    `android/store_listing/feature_graphic.html` (dark navy background with a subtle
    blue radial glow, the actual icon mark + "MediQuery" wordmark in the same
    `mediquery-logo-dark.svg` white-on-dark style as the desktop brand, plus the
    short/long taglines from the store listing copy).
  - **Real rendering-pipeline bug hit and fixed:** ImageMagick's built-in SVG
    delegate (MSVG) doesn't support the SVG `<text>` element at all — the wordmark
    silently failed to render (icon showed, text was blank), so text-bearing
    graphics needed a real browser instead. Then the claude-in-chrome extension's
    own screenshot capture path introduced a severe, confirmed color-space bug
    (page's own `getComputedStyle` correctly reported `rgb(29,78,216)` for the icon,
    but the extension's screenshot captured it as `rgb(67,0,217)` — green channel
    zeroed out, a visibly wrong purple). Diagnosed by comparing DOM computed style
    against sampled screenshot pixels (not just eyeballing it) before concluding it
    was a capture-path bug rather than a CSS/dark-mode issue. **Fix: bypassed the
    extension's screenshot entirely** — served the HTML over a local Python
    `http.server` (the extension also blocks `file://` navigation) and captured it
    with `google-chrome --headless --screenshot=...` directly via CLI, which
    reproduced the exact correct color (`srgb(29,78,216)`, verified pixel-for-pixel
    against the source hex). Worth remembering for any future asset-export work in
    this environment: don't trust the extension's screenshot tool for
    color-critical exports — use headless Chrome CLI directly instead.
- **Not yet done:** content rating questionnaire (can only be filled out inside Play
  Console itself, not prepared in advance), and — separately from Play Store
  mechanics — this build still uses the placeholder-password keystore, needs a
  version bump decision (currently `0.1.0`), and hasn't had minification/R8
  evaluated (`isMinifyEnabled = false` currently; enabling it would shrink the
  104MB bundle but risks breaking JNI-bound classes in litertlm-android/
  onnxruntime-android if their consumer ProGuard rules aren't complete — would need
  a full functional re-test if enabled, not done here).

- **2026-08-26 — XAI confidence layer ported to Android.** `retrieval/ConfidenceScorer.kt`
  ports the desktop's `_retrieval_confidence()` and `_source_agreement_score()` from
  `src/xai/multi_signal_confidence.py`, renormalized to just those two signals
  (retrieval 0.45, source_agreement 0.35 → renormalized to 0.5625/0.4375) since
  LiteRT-LM exposes no token-probability API at all (confirmed via direct API
  inspection — decided to drop `generation_confidence` and renormalize the other two
  rather than fake a third signal). `CalibratedConfidence` wraps the raw score in a
  Platt sigmoid (`1/(1+exp(-(a*raw+b)))`) with `level()` thresholds matching desktop
  (high ≥0.75, medium ≥0.45). Calibration constants (a, b) are fit fresh for Gemma
  rather than reusing desktop's TinyLlama-fitted `platt_a=14.44, platt_b=-11.25` —
  chosen deliberately since the desktop docstring states those were "fitted on
  TinyLlama outputs only," and a different base model's raw-score distribution isn't
  assumed to be interchangeable.
- **2026-08-26 — `CalibrationRunner.kt` built and a real hang bug found + fixed.**
  Batch-runs the full on-device RAG pipeline over all 97 questions in
  `evaluation/test_set_v2.json` (pushed to device unmodified), writing one JSON line
  per case to `<app external files dir>/calibration_results.jsonl` — this is data
  collection only; the actual Platt fit happens offline in Python
  (`evaluation/fit_mobile_calibration.py`, not yet written), mirroring how desktop's
  own `fit_calibration()` works.
  - **Output path bug (fixed quickly):** first attempt wrote to
    `/data/local/tmp/calibration_results.jsonl` and crashed instantly with
    `FileNotFoundException: ... EACCES (Permission denied)`. SELinux blocks an app
    process from *creating new files* in `/data/local/tmp` even though it can read
    existing ones there (world-readable POSIX bits don't imply world-writable-by-apps
    under SELinux) — this had worked all session for reading the model/KB files, so
    it wasn't obvious until it broke on a write. Fixed: output goes to the app's
    external files dir instead (`context.getExternalFilesDir(null)`), still directly
    `adb pull`-able, no `run-as` needed.
  - **Real hang bug (took three full debugging sessions to pin down):** after the
    path fix, the run would reliably die silently after a small, inconsistent number
    of cases (sometimes 0, sometimes a few) — process stayed alive, foreground,
    no crash, no exception, just total silence in logcat forever. Wrongly diagnosed
    twice before being isolated:
    1. First suspected screen-doze / background-kill (Samsung's aggressive task
       killer had genuinely killed the app once before, for an unrelated reason —
       a leftover Settings "InstalledAppDetails" screen stealing focus — which
       muddied the diagnosis).
    2. Then suspected a broader GPU-backend regression, ruled out by an isolation
       test: launched the *same build* in normal (non-calibration) chat mode and
       confirmed a single query completed cleanly in ~6s. This proved the bug was
       specific to `CalibrationRunner`'s code path, not the engine/GPU backend
       generally.
    3. **Root cause, found only by instrumenting both code paths with `Log.i` calls
       in `onMessage`/`onDone`/`onError` and rebuilding** (inference from timing
       alone was actively misleading — see below): `generateOnce()` used
       `suspendCancellableCoroutine`, resuming the continuation (`cont.resume(...)`)
       directly from inside `onDone()`/`onError()`, which fire on a native
       callback thread, not the calling coroutine's `Dispatchers.Default` thread.
       Logging proved `onDone` fired correctly and fast (~4s) for the *first*
       conversation — but nothing after `cont.resume()` ever ran: the loop never
       reached the second question, with zero further log output, forever. The
       cross-thread continuation resumption itself was silently not working,
       despite `suspendCancellableCoroutine` being documented as thread-safe for
       this — never fully root-caused at the native/coroutine-internals level, but
       reliably reproduced twice with the logging in place.
       **A plausible false lead worth recording:** the *first* debugging pass (no
       per-callback logging yet) assumed the hang was mid-decode on the very first
       question, based on log timing proximity to the GPU-sampler-unavailable→
       CPU-sampling-fallback warning that both chat and calibration modes hit. That
       was wrong — the fallback warning is benign and unrelated; the actual hang
       point was only found once real per-callback instrumentation existed. Timing
       inference from logcat proximity was not a reliable substitute for direct
       evidence here.
  - **Fix:** replaced the `suspendCancellableCoroutine` bridge with a plain
    `AtomicBoolean` polling loop (`while (!done.get()) delay(100)`) inside
    `generateOnce()` — `onDone`/`onError` just set the flag from whatever thread
    they're called on; the suspend point is an ordinary `delay()` on the coroutine's
    own dispatcher, never a cross-thread continuation resume. **Verified fixed**:
    reran calibration mode with this fix and watched 12+ consecutive cases complete
    cleanly (~4-13s each) with no hang, vs. 0-3 cases before. Full 97-case run
    completed after this fix (see Results section once the offline Platt fit is
    done). This is a good general lesson for wrapping any LiteRT-LM (or likely any
    native-callback-based) async API in coroutines: prefer polling a flag over
    `suspendCancellableCoroutine` if the callback may fire from an unexpected native
    thread in a tight loop — untested, but the individual-conversation fire-and-
    forget pattern in normal chat mode (which never suspends on the result) never
    exercised this path, which is why it looked fine there.
  - **Full 97-case run completed after the fix**, then fit offline via
    `evaluation/fit_mobile_calibration.py` (mirrors desktop's `fit_calibration()`:
    same Nelder-Mead NLL minimization on `sigmoid(a*raw+b)`, correctness labels from
    the same `keyword_coverage`/`is_correct` used everywhere else in this repo — not
    reimplemented). Results (n=97, 0 error rows): raw ECE 0.161 → calibrated ECE
    0.032, fitted `a=5.6881, b=-3.6041`, 37.1% positive label rate. Output at
    `evaluation/results/mobile_calibration.json` +
    `evaluation/results/mobile_calibration_results.jsonl` (the raw per-question
    device output) + two reliability-diagram PNGs.
  - **UI wired up (2026-08-26):** `ChatTurn` gained `confidenceLevel`/
    `confidenceScore` state; `sendMessage()`'s `onDone` now computes retrieval
    confidence + source agreement on the actual generated answer, applies the
    fitted Platt constants (hardcoded in `CalibratedConfidence.default`, matching
    the project's existing "no unnecessary abstraction" convention — two doubles
    don't need an asset-loading path), and a new `ConfidenceBadge` composable shows
    it under each answer (color-coded emerald/amber/red-300, matching the
    high/medium/low semantic colors the desktop frontend already uses in
    `answer-card.tsx`). **Verified live on-device**: sent a real query, confirmed
    via `uiautomator dump` that "Confidence: Low (15%)" rendered correctly beneath
    a correct, well-grounded answer — a low score here isn't a bug, it reflects
    that the Platt fit was trained against a strict keyword-coverage correctness
    proxy where only 37% of the 97-question set counted as "correct," so the
    calibration is intentionally conservative.
  - The temporary `Log.i("MQCAL", ...)` diagnostic lines used to find the hang were
    removed from `MainActivity.kt`'s normal chat path after the fix (production
    code); left in `CalibrationRunner.kt` since it's explicitly a one-off
    diagnostic tool, not user-facing.

- **2026-08-26 — Model + KB bundled into the APK; app now installs standalone.**
  Previously the app depended on `adb push`-ing the ~530MB `.task` model and the
  KB subset to `/data/local/tmp/` by hand — meaning it only ever worked on this
  dev's own test device, never for anyone who'd just install the APK. Fixed:
  - Pulled the model back off the device (it only existed there + in the
    phone's Downloads folder, not on this machine) into
    `android/app/src/main/assets/gemma3-1b-it.task`; copied the KB subset
    (`evaluation/mobile_kb_subset.jsonl`) in as `mobile_kb.jsonl`. Both
    git-ignored (~530MB — GitHub blocks anything over 100MB in a normal push
    anyway) but present locally for Gradle to bundle into the built APK.
  - `app/build.gradle.kts`: added `noCompress += listOf("task", "jsonl")` —
    needed because `AssetManager.openFd()` (used for upfront total-size
    progress reporting) only works on assets stored uncompressed. Also
    restricted `ndk.abiFilters` to `arm64-v8a` only — a 1B-param model with a
    GPU delegate isn't realistically usable on 32-bit ARM or x86 hardware, and
    shipping all 4 ABIs' native libs alongside a 530MB model would have been
    wasteful (covers the large majority of real phones from the last ~8
    years; excludes emulators and legacy 32-bit devices).
  - New `AssetInstaller.kt`: copies a bundled asset out to a real file on
    first run (both LiteRT-LM's `modelPath` and `KnowledgeBase.load()` need an
    actual filesystem path, not an asset stream), idempotent via a
    size-match check so every launch after the first is instant. Reports
    progress via a callback — `MainActivity.loadModel()` shows "Setting up
    model... NN%" during the one-time copy.
  - `KnowledgeBase.KB_PATH`'s hardcoded `/data/local/tmp/...` constant and
    `MainActivity`'s `MODEL_PATH` constant were both removed in favor of
    dynamic `filesDir`-relative paths computed after the asset copy.
  - **Verified twice, not just build success:** (1) debug build — force-stopped
    the app, `pm clear`'d its data, deleted every file this app had ever relied
    on under `/data/local/tmp` on the device, fresh-launched, confirmed
    "1444 sources loaded" and a correct answer with confidence score, all with
    zero adb pushes. (2) release build — full uninstall/reinstall of the
    actual signed release APK (not debug), same standalone verification,
    ready in ~20s, correct answer with sources and confidence. This is the
    real distributable artifact, tested as an end user would experience it.
- **2026-08-26 — Release keystore regenerated.** The placeholder keystore used
  throughout earlier Play Store prep had a weak hardcoded password
  (`mediquery-dev-2026`) — fine for testing the build pipeline, not fine for
  an APK that's about to be a real public download (whoever holds the signing
  key controls all future "trusted update" releases). Generated a fresh 2048-
  bit RSA keystore with a strong random password (`openssl rand -base64 24`);
  both the keystore file and `signing.properties` remain git-ignored as
  before. This is the only keystore that should be used for any real release
  going forward — the old placeholder should be treated as retired.
- **Distribution decision:** Play Store submission is explicitly not
  happening. Distribution is via GitHub Releases instead — a single signed
  APK attached to a repo release, downloaded and sideloaded directly. This
  sidesteps Play Store's base-APK size limits (which would have forced Play
  Asset Delivery for the 530MB model) since GitHub Releases allows up to 2GB
  per file with no special packaging needed. Final release APK: **619MB**,
  arm64-v8a only, signed with the new keystore above.
- **Disk space incident during this work:** the release build initially failed
  with `IOException: No space left on device` — the machine's root
  filesystem was at 100% (221GB/233GB used, 110MB free), unrelated to this
  session specifically. Freed space by deleting
  `android/app/build/` (4.8GB of pure regenerable Gradle build output, no
  different from a `dist/` folder) — nothing else on the machine was touched.
  Disk sits at ~2.8GB free after the release build completed — worth a real
  cleanup pass (Docker build cache showed 2.09GB fully reclaimable, `.npm`
  cache was 6.9GB) before the next large build on this machine.

- **2026-08-27 — Real cross-device bug found and fixed: no GPU→CPU fallback.**
  Tested the release APK on a second, independent physical device (Samsung Galaxy S21,
  Exynos, Android 15/API 35) — a different phone with a different GPU vendor/driver stack
  than the dev device (Galaxy S9+, Mali, Android 10). The app failed to load at all:
  `Failed to create engine: INTERNAL: ERROR: ...llm_litert_compiled_model_executor.cc:1928`.
  Full logcat trace showed the actual cause: this device has no working OpenCL
  (`OpenCL not supported on this platform. Using OpenGL instead.` /
  `dlopen failed: library "libvndksupport.so" not found`), and LiteRT-LM's OpenGL GPU
  delegate path also failed —
  `Failed to create litert::ml_drift::DelegateKernelLiteRt: UNIMPLEMENTED:
  CreateSharedMemoryManager is not implemented.` `Backend.GPU()` was hardcoded with no
  fallback, so engine creation just threw and the app showed an unrecoverable "Failed to
  load" error — exactly the risk flagged (but not yet tested) when this was first asked
  about ("does the APK install on any Android phone").
  **Fix:** `MainActivity.loadModel()` now tries `Backend.GPU()` first and, if engine
  creation throws, catches it and retries with `Backend.CPU()`, updating `modelStatus` to
  "GPU unavailable, falling back to CPU..." in between. New `createEngine()` helper shared
  by both attempts.
  **Verified on the same S21 that failed before the fix:** rebuilt, reinstalled, watched
  logcat confirm the GPU attempt failed and the CPU attempt succeeded
  (`backend: GPU` → failure → `backend: CPU` → `CPU accelerator registered`), then sent a
  real query — answered correctly with sources and confidence score within ~20 seconds of
  tapping Send (CPU-only inference on this device's chip is notably fast, faster than the
  dev device's GPU-fallback-to-CPU-sampler path even).

---

## Open Items / Pre-Publication Flags

- **Patent scope note:** the guide's patent target is the confidence-gated RAG
  architecture (the multi-signal scoring + gating mechanism), not the specific base
  model weights. Neither the mobile-only swap nor a hypothetical whole-project swap
  changes that claim. Recorded here for completeness, not adjudicated in this file —
  see `project_paperwork/patent/invention_disclosure.md`.
- **Must resolve before writing up any mobile results:** `evaluation/test_set_v2.json`'s
  actual source composition (PubMedQA 40 / ChatDoctor-HealthCareMagic 25 /
  MedQA-USMLE 32) does not match `paper.tex`'s stated composition ("40 PubMedQA, 13
  ChatDoctor, rest MedMCQA" — MedMCQA isn't in the eval set's sources at all). If
  mobile results get reported "against the same 97-question eval set," the mobile
  section will inherit whichever description ends up in the paper — fix the paper's
  description (or figure out which set is actually canonical) before that happens.
