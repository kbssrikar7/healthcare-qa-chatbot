## MediQuery — Android (on-device)

An on-device port of the Healthcare QA Chatbot: a fully local, offline RAG chatbot running **Gemma 3 1B** via Google's LiteRT-LM (GPU-accelerated with automatic CPU fallback), with hybrid BM25 + dense retrieval over a curated medical knowledge-base subset, and a ported multi-signal confidence layer (Platt-calibrated against a real 97-question on-device evaluation).

**Standalone install** — the model and knowledge base are bundled inside the APK and set themselves up on first launch. No `adb`, no manual file pushes, no internet permission (the app has none — everything runs and stays on-device).

**2026-08-27 update:** fixed a real bug found while testing on a second physical device (Galaxy S21) — the app now automatically falls back to CPU if a device has no working GPU compute delegate, instead of failing to load entirely. Previous builds without this fix would not work on such devices.

### Requirements
- Android 8.0 (API 26) or newer
- **arm64-v8a** device (the large majority of phones from the last ~8 years; this build will not install on 32-bit-only or x86 devices/emulators)
- ~3GB+ RAM recommended, ~1.2GB free storage for the app + first-run setup
- Uses the GPU when the device supports it (noticeably faster); automatically falls back to CPU otherwise — verified working on both paths on real hardware

### Install
1. Download `app-release.apk` below.
2. On your phone, allow "install unknown apps" for the browser/file manager you're using (Android will prompt you if you haven't already).
3. Open the downloaded file and install.
4. First launch takes ~15–30s to set up (copying the bundled model out to app storage) — subsequent launches are instant.

### Important
This is a research/capstone-project demo build, **not a certified medical device**. Answers are for educational purposes only and are not a substitute for professional medical advice — this disclaimer is shown in-app on every screen.

APK is signed with a dedicated release key (not the Play Store — this project isn't distributed there). SHA-256 of `app-release.apk`:
```
1eb9cc4290590e200cb75e6765c4f59f79d15f63c830c9a7a3c0152f6a6cc254
```

