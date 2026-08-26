package com.mediquery.mobile.retrieval

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlin.math.sqrt

/**
 * On-device sentence embedder for all-MiniLM-L6-v2, matching the desktop
 * `sentence-transformers` behavior exactly: masked mean-pooling over token
 * embeddings (confirmed via the model's own 1_Pooling/config.json —
 * pooling_mode_mean_tokens=true, not CLS), then L2 normalization (desktop
 * calls encode(..., normalize_embeddings=True) — see
 * src/embeddings/embedding_models.py). Model: sentence-transformers/
 * all-MiniLM-L6-v2, arm64 int8-quantized ONNX export (~22MB), bundled as an
 * app asset. See project_paperwork/scratch/mobile_port_notes.md for the
 * verification trail (graph inputs/outputs inspected before writing this).
 */
class OnnxEmbedder(context: Context, modelAsset: String = "minilm.onnx") {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer = WordPieceTokenizer(context)
    private val maxLength = 256 // matches sentence_bert_config.json max_seq_length

    init {
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        session = env.createSession(modelBytes)
    }

    /** Encode [text] into a 384-dim, L2-normalized embedding. */
    fun embed(text: String): FloatArray {
        val encoded = tokenizer.encode(text, maxLength)
        val shape = longArrayOf(1, maxLength.toLong())

        OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(encoded.inputIds), shape).use { inputIds ->
            OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(encoded.attentionMask), shape).use { attentionMask ->
                OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(encoded.tokenTypeIds), shape).use { tokenTypeIds ->
                    val inputs = mapOf(
                        "input_ids" to inputIds,
                        "attention_mask" to attentionMask,
                        "token_type_ids" to tokenTypeIds,
                    )
                    session.run(inputs).use { result ->
                        val lastHiddenState = result.get("last_hidden_state").get() as OnnxTensor
                        // Shape [1, maxLength, 384] flattened.
                        val flat = lastHiddenState.floatBuffer
                        return meanPoolAndNormalize(flat, encoded.attentionMask, maxLength, 384)
                    }
                }
            }
        }
    }

    private fun meanPoolAndNormalize(
        flat: java.nio.FloatBuffer,
        attentionMask: LongArray,
        seqLen: Int,
        hidden: Int,
    ): FloatArray {
        val pooled = FloatArray(hidden)
        var maskSum = 0f
        for (t in 0 until seqLen) {
            val m = attentionMask[t]
            if (m == 0L) continue
            maskSum += 1f
            val base = t * hidden
            for (h in 0 until hidden) {
                pooled[h] += flat.get(base + h) * m
            }
        }
        val denom = if (maskSum > 0f) maskSum else 1f
        for (h in 0 until hidden) pooled[h] /= denom

        var norm = 0f
        for (v in pooled) norm += v * v
        norm = sqrt(norm)
        if (norm > 0f) {
            for (h in 0 until hidden) pooled[h] /= norm
        }
        return pooled
    }

    fun close() {
        session.close()
    }
}
