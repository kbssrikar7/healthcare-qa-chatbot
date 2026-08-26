# MediQuery — Privacy Policy

*Last updated: [FILL IN DATE WHEN PUBLISHED]*

## Summary

MediQuery is designed to work entirely on your device. **We do not collect,
store, transmit, or have access to any of your data.** There is no server
component to this app, no account, no analytics, and no advertising.

## What the app does

MediQuery answers medical questions by running a language model and a
document-retrieval search directly on your phone. Your question, the
retrieved reference passages, and the generated answer are processed
entirely in your device's memory and are never sent anywhere.

## What we collect

**Nothing.** Specifically:

- **No account or sign-in** is required or offered.
- **No network permission.** The app does not request or hold Android's
  `INTERNET` permission — it is technically incapable of sending data over
  the network, not merely configured not to.
- **No analytics, telemetry, or crash-reporting SDKs.**
- **No advertising SDKs.**
- **No persistence.** Your conversation history exists only in the app's
  running memory and is cleared when the app is closed — nothing is written
  to a database or file that could be recovered later.
- **No device identifiers, location, contacts, camera, microphone, or any
  other sensitive permission** is requested.

## Third-party components

The app bundles open-source, on-device machine learning models and
libraries (Google's LiteRT-LM and Gemma, and an ONNX Runtime-based sentence
embedding model). These run entirely locally and do not communicate with
their providers or any other party while the app is in use.

## Children's privacy

MediQuery is not directed at children and is not designed to collect any
information from anyone, including children.

## Medical disclaimer

MediQuery provides general educational information retrieved from public
medical reference sources. It is not a licensed medical device and does not
provide medical advice, diagnosis, or treatment. Always consult a qualified
healthcare professional for medical concerns.

## Changes to this policy

If this policy changes, an updated version will be published at this same
location with a revised "Last updated" date.

## Contact

[FILL IN: a real support email or contact URL — Google Play requires this]

---

*Drafting note (remove before publishing): this document was drafted based
on a direct technical check of the app's manifest and code — confirmed zero
`INTERNET`/`ACCESS_NETWORK_STATE` permissions, no analytics/ads dependencies
in build.gradle.kts, and no local persistence in the current chat-history
implementation. If any of that changes in a future version (e.g. you add
crash reporting, or persist chat history to disk), this policy needs to be
updated to match — Play Store review checks that the policy accurately
reflects what the app does.*
