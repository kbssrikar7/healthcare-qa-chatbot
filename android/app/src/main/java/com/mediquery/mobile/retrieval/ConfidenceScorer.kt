package com.mediquery.mobile.retrieval

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * On-device port of the two signals from src/xai/multi_signal_confidence.py
 * that still carry non-zero weight in the desktop's *current* production
 * `DEFAULT_WEIGHTS` (verified directly in code this session — consistency
 * and entity_coverage are both weighted 0.00 already, generation_confidence
 * is weighted 0.20 but requires token-level probabilities that LiteRT-LM's
 * public API does not expose, in either the version this app uses or the
 * latest release — confirmed by inspecting both via javap before deciding to
 * drop it). So: retrieval_confidence (0.45) and source_agreement (0.35),
 * renormalized to sum to 1.0 since only two of the three non-zero-weight
 * desktop signals are computable on-device. See
 * project_paperwork/scratch/mobile_port_notes.md ("XAI / Confidence Layer").
 *
 * Platt scaling (a, b — sigmoid(a*raw+b)) is NOT hardcoded here; it's loaded
 * from calibration.json (see CalibratedConfidence below), fitted offline
 * against real on-device generations the same way
 * multi_signal_confidence.py::fit_calibration() does (Nelder-Mead on
 * negative log-likelihood) — see evaluation/fit_mobile_calibration.py.
 */
object ConfidenceScorer {

    const val RETRIEVAL_WEIGHT = 0.45 / (0.45 + 0.35) // 0.5625
    const val SOURCE_AGREEMENT_WEIGHT = 0.35 / (0.45 + 0.35) // 0.4375
    private const val RRF_K = 60.0

    private val stopwords = setOf(
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "for", "and", "or",
        "but", "in", "on", "at", "to", "of", "by", "with", "from", "as",
        "this", "that", "these", "those", "it", "its", "not", "also", "if",
        "such", "which", "when", "who", "how", "what", "you", "your", "i",
        "my", "we", "our", "they", "their", "him", "her", "his", "any",
        "all", "so", "than", "then", "there", "about", "up", "out", "into",
        "through", "during", "including", "without", "please", "consult",
    )

    /** Port of _retrieval_confidence (score_type="rrf" branch). */
    fun retrievalConfidence(results: List<HybridRetriever.Result>): Double {
        if (results.isEmpty()) return 0.0
        val scaleFactor = (RRF_K + 1) / 2.0
        val normScores = results.map { min(it.rrfScore * scaleFactor, 1.0) }

        val topScore = normScores.first()
        val meanScore = normScores.average()
        val dropoff = if (normScores.size > 1) {
            (normScores.first() - normScores.last()) / (normScores.first() + 1e-8)
        } else 0.5
        val nStrong = normScores.count { it > 0.5 }
        val coverage = min(nStrong.toDouble() / max(normScores.size, 1), 1.0)

        val score = 0.3 * topScore + 0.2 * meanScore + 0.25 * dropoff + 0.25 * coverage
        return score.coerceIn(0.0, 1.0)
    }

    /** Port of _source_agreement_score (stopword removal + suffix stemming + 0.10 overlap threshold). */
    fun sourceAgreement(answer: String, results: List<HybridRetriever.Result>): Double {
        if (results.isEmpty()) return 0.0
        val answerTokens = tokenizeForAgreement(answer)
        if (answerTokens.isEmpty()) return 0.0

        var supporting = 0
        for (r in results) {
            val contentTokens = tokenizeForAgreement(r.chunk.text)
            if (contentTokens.isEmpty()) continue
            val overlap = answerTokens.intersect(contentTokens).size.toDouble() / answerTokens.size
            if (overlap > 0.10) supporting++
        }
        return min(supporting.toDouble() / max(results.size, 1), 1.0)
    }

    private fun tokenizeForAgreement(text: String): Set<String> {
        val words = Regex("[a-z0-9]+").findAll(text.lowercase()).map { it.value }
        return words.filter { it !in stopwords && it.length >= 3 }.map { normaliseWord(it) }.toSet()
    }

    private val suffixes = listOf(
        "tion", "tions", "ing", "ings", "ness", "ment", "ments",
        "ical", "ically", "ity", "ies", "ied", "ed", "es", "s",
    )

    private fun normaliseWord(word: String): String {
        for (suffix in suffixes) {
            if (word.endsWith(suffix) && word.length - suffix.length >= 4) {
                return word.substring(0, word.length - suffix.length)
            }
        }
        return word
    }

    /** Raw (pre-calibration) weighted score — this is what Platt scaling is fit against. */
    fun rawScore(retrievalConf: Double, sourceAgreement: Double): Double =
        RETRIEVAL_WEIGHT * retrievalConf + SOURCE_AGREEMENT_WEIGHT * sourceAgreement
}

/** Applies fitted Platt scaling: sigmoid(a * raw + b). Params loaded from an asset written by fit_mobile_calibration.py. */
class CalibratedConfidence(private val plattA: Double, private val plattB: Double) {
    fun calibrate(rawScore: Double): Double {
        val z = plattA * rawScore + plattB
        return 1.0 / (1.0 + exp(-z))
    }

    companion object {
        /**
         * Fitted via evaluation/fit_mobile_calibration.py against the 97-question
         * on-device calibration run (2026-08-26, Gemma 3 1B, GPU backend):
         * raw ECE 0.161 -> calibrated ECE 0.032, n=97, 37.1% positive label rate.
         * Fresh fit for Gemma, not reused from desktop's TinyLlama-fitted values —
         * see ConfidenceScorer's class doc.
         */
        const val PLATT_A = 5.6881
        const val PLATT_B = -3.6041
        val default = CalibratedConfidence(PLATT_A, PLATT_B)

        fun level(calibratedScore: Double): String = when {
            calibratedScore >= 0.75 -> "high"
            calibratedScore >= 0.45 -> "medium"
            else -> "low"
        }
    }
}
