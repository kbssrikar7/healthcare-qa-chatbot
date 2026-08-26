package com.mediquery.mobile

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.mediquery.mobile.retrieval.ConfidenceScorer
import com.mediquery.mobile.retrieval.HybridRetriever
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Batch-runs the full on-device RAG pipeline (retrieval + generation) over
 * evaluation/test_set_v2.json — pushed to the device unmodified, no
 * simplified copy needed — computing the two on-device confidence signals
 * for each question and writing one JSON line per result. This is
 * calibration DATA COLLECTION only: correctness labeling (keyword coverage)
 * and the actual Platt fit happen offline in Python
 * (evaluation/fit_mobile_calibration.py), mirroring how the desktop's own
 * calibration works — see mobile_port_notes.md.
 *
 * Triggered via an intent extra rather than a permanent UI button, since
 * this is a one-off diagnostic run, not a feature end users need:
 *   adb shell am start -n com.mediquery.mobile/.MainActivity --es mode calibrate
 *
 * Input:  /data/local/tmp/test_set_v2.json (pushed as-is from the repo) — apps
 *         CAN read /data/local/tmp (world-readable POSIX bits), confirmed
 *         working throughout this session for the model/KB files.
 * Output: <app external files dir>/calibration_results.jsonl — NOT
 *         /data/local/tmp. First attempt tried writing there too and crashed
 *         immediately with `FileNotFoundException: ... EACCES (Permission
 *         denied)` — SELinux blocks an app process from *creating new files*
 *         in /data/local/tmp even though it can read existing ones (world-
 *         readable POSIX bits don't imply world-writable-by-apps under
 *         SELinux). The external files dir
 *         (/sdcard/Android/data/com.mediquery.mobile/files/) is fully
 *         writable by the app and still directly `adb pull`-able, no
 *         `run-as` needed.
 */
object CalibrationRunner {
    const val INPUT_PATH = "/data/local/tmp/test_set_v2.json"
    const val OUTPUT_FILENAME = "calibration_results.jsonl"
    private const val RETRIEVE_K = 3
    private const val MAX_CONTEXT_CHARS_PER_CHUNK = 600

    data class CaseResult(val index: Int, val total: Int, val id: String, val error: String?)

    suspend fun run(
        context: Context,
        engine: Engine,
        retriever: HybridRetriever,
        onProgress: (CaseResult) -> Unit,
    ) {
        val inputFile = File(INPUT_PATH)
        if (!inputFile.exists()) {
            onProgress(CaseResult(0, 0, "", "Input not found at $INPUT_PATH"))
            return
        }
        val root = JSONObject(inputFile.readText())
        val cases = root.getJSONArray("test_cases")
        val outputFile = File(context.getExternalFilesDir(null), OUTPUT_FILENAME)
        outputFile.writeText("") // truncate any previous run

        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val id = case.getString("id")
            val query = case.getString("query")
            try {
                val retrieved = retriever.retrieve(query, k = RETRIEVE_K)
                val prompt = buildRagPrompt(query, retrieved)
                val answer = generateOnce(engine, prompt)

                val retrievalConf = ConfidenceScorer.retrievalConfidence(retrieved)
                val sourceAgreement = ConfidenceScorer.sourceAgreement(answer, retrieved)
                val rawScore = ConfidenceScorer.rawScore(retrievalConf, sourceAgreement)

                val record = JSONObject().apply {
                    put("id", id)
                    put("query", query)
                    put("answer", answer)
                    put("retrieval_confidence", retrievalConf)
                    put("source_agreement", sourceAgreement)
                    put("raw_score", rawScore)
                    put("num_sources", retrieved.size)
                    put("expected_keywords", case.optJSONArray("expected_keywords") ?: JSONArray())
                }
                outputFile.appendText(record.toString() + "\n")
                onProgress(CaseResult(i + 1, cases.length(), id, null))
            } catch (e: Exception) {
                outputFile.appendText(
                    JSONObject().apply {
                        put("id", id); put("query", query); put("error", e.message ?: "unknown")
                    }.toString() + "\n"
                )
                onProgress(CaseResult(i + 1, cases.length(), id, e.message))
            }
        }
    }

    private fun buildRagPrompt(question: String, retrieved: List<HybridRetriever.Result>): String {
        if (retrieved.isEmpty()) return question
        val context = retrieved.withIndex().joinToString("\n\n") { (idx, r) ->
            "[${idx + 1}] ${truncateAtSentence(r.chunk.text, MAX_CONTEXT_CHARS_PER_CHUNK)}"
        }
        return "Answer the medical question using ONLY the context passages below. " +
            "If the context does not contain the answer, say so explicitly rather than guessing.\n\n" +
            "Context:\n$context\n\nQuestion: $question"
    }

    private fun truncateAtSentence(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val window = text.take(maxChars)
        val lastSentenceEnd = window.lastIndexOfAny(charArrayOf('.', '!', '?'))
        return if (lastSentenceEnd > maxChars / 2) window.take(lastSentenceEnd + 1)
        else window.substringBeforeLast(' ') + "..."
    }

    /**
     * Runs one fresh-conversation generation to completion, polling a flag rather than
     * suspending on a continuation resumed from the native callback thread. A
     * suspendCancellableCoroutine-based version of this (resuming `cont` directly from
     * onDone/onError) was found to hang indefinitely after the *first* successful
     * generation in a tight loop: onDone logged successfully, but nothing downstream of
     * cont.resume() ever ran — the resumed coroutine never continued. Polling avoids
     * cross-thread continuation resumption entirely. See mobile_port_notes.md.
     */
    private suspend fun generateOnce(engine: Engine, prompt: String): String {
        val conversation: Conversation = engine.createConversation(
            ConversationConfig(
                samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 1.0),
            )
        )
        val sb = StringBuilder()
        val done = AtomicBoolean(false)
        Log.i("MQCAL", "calib: sendMessageAsync called")
        var firstMsg = true
        conversation.sendMessageAsync(
            Contents.of(prompt),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    if (firstMsg) { Log.i("MQCAL", "calib: onMessage first"); firstMsg = false }
                    sb.append(message.toString())
                }
                override fun onDone() {
                    Log.i("MQCAL", "calib: onDone")
                    done.set(true)
                }
                override fun onError(throwable: Throwable) {
                    Log.i("MQCAL", "calib: onError ${throwable.message}")
                    done.set(true)
                }
            },
        )
        while (!done.get()) {
            delay(100)
        }
        conversation.close()
        Log.i("MQCAL", "calib: conversation closed, returning")
        return sb.toString()
    }
}
