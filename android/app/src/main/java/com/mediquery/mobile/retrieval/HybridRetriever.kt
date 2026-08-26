package com.mediquery.mobile.retrieval

import kotlin.math.ln
import kotlin.math.sqrt

/**
 * On-device port of src/retrieval/hybrid_retriever.py's fusion logic (query-type
 * classification, adaptive RRF weights, BM25Okapi, RRF formula) running over the
 * small curated KB subset instead of the full 505k-doc corpus — see
 * project_paperwork/scratch/mobile_port_notes.md ("Retrieval Architecture") for
 * why brute-force dense + hand-rolled BM25 was chosen over a native ANN/FTS5
 * extension at this KB size (low thousands of chunks).
 */
class HybridRetriever(private val chunks: List<KbChunk>, private val embedder: OnnxEmbedder) {

    // ── Query-type classification — same patterns as hybrid_retriever.py ────────
    private val qtDrug = Regex(
        "\\b(?:drug|medication|medicine|dose|dosage|mg|tablet|capsule|pill|" +
            "prescri|pharmac|antibiotic|aspirin|ibuprofen|metformin|insulin|" +
            "interaction|side\\s*effect|contraindic)\\b",
        RegexOption.IGNORE_CASE,
    )
    private val qtDefinition = Regex(
        "^(?:what\\s+is|what\\s+are|define|explain|describe|tell\\s+me\\s+about|" +
            "meaning\\s+of|definition\\s+of)\\b",
        RegexOption.IGNORE_CASE,
    )
    private val qtSymptom = Regex(
        "\\b(?:symptom|sign|feel|pain|ache|discomfort|nausea|fever|cough|" +
            "fatigue|dizzy|swelling|bleed|rash|itch)\\b",
        RegexOption.IGNORE_CASE,
    )
    private val qtComparison = Regex(
        "\\b(?:difference|vs\\.?|versus|compare|better|worse|than|between)\\b",
        RegexOption.IGNORE_CASE,
    )

    // (denseWeight, sparseWeight) per query type — identical to _ADAPTIVE_WEIGHTS.
    private val adaptiveWeights = mapOf(
        "drug" to (0.45 to 0.55),
        "definition" to (0.80 to 0.20),
        "symptom" to (0.65 to 0.35),
        "comparison" to (0.80 to 0.20),
        "default" to (0.70 to 0.30),
    )

    private fun detectQueryType(query: String): String = when {
        qtDrug.containsMatchIn(query) -> "drug"
        qtDefinition.containsMatchIn(query) -> "definition"
        qtSymptom.containsMatchIn(query) -> "symptom"
        qtComparison.containsMatchIn(query) -> "comparison"
        else -> "default"
    }

    // ── BM25Okapi — matches rank_bm25's BM25Okapi exactly (k1=1.2, b=0.5,
    //    epsilon=0.25 default), including negative-idf clipping. ────────────────
    private val k1 = 1.2
    private val b = 0.5
    private val epsilon = 0.25

    private val docTokens: List<List<String>> = chunks.map { tokenize(it.text) }
    private val docLen: IntArray = IntArray(chunks.size) { docTokens[it].size }
    private val avgdl: Double = if (chunks.isEmpty()) 0.0 else docLen.sum().toDouble() / chunks.size
    private val docFreqs: List<Map<String, Int>> = docTokens.map { tokens ->
        val freq = HashMap<String, Int>()
        for (t in tokens) freq[t] = (freq[t] ?: 0) + 1
        freq
    }
    private val idf: Map<String, Double> = run {
        val nd = HashMap<String, Int>() // number of docs containing each word
        for (freq in docFreqs) {
            for (word in freq.keys) nd[word] = (nd[word] ?: 0) + 1
        }
        val n = chunks.size
        val idfMap = HashMap<String, Double>()
        var idfSum = 0.0
        val negative = ArrayList<String>()
        for ((word, freq) in nd) {
            val value = ln(n - freq + 0.5) - ln(freq + 0.5)
            idfMap[word] = value
            idfSum += value
            if (value < 0) negative.add(word)
        }
        val averageIdf = if (idfMap.isNotEmpty()) idfSum / idfMap.size else 0.0
        val eps = epsilon * averageIdf
        for (word in negative) idfMap[word] = eps
        idfMap
    }

    private fun tokenize(text: String): List<String> {
        val lower = text.lowercase()
        val cleaned = lower.replace(Regex("[^\\w\\s\\-]"), " ")
        return cleaned.split(Regex("\\s+")).filter { it.length >= 2 }
    }

    /** BM25 scores for [queryTokens] over every doc; index-aligned with [chunks]. */
    private fun bm25Scores(queryTokens: List<String>): DoubleArray {
        val scores = DoubleArray(chunks.size)
        for (q in queryTokens.toSet()) {
            val wordIdf = idf[q] ?: continue
            for (i in chunks.indices) {
                val qFreq = docFreqs[i][q] ?: 0
                if (qFreq == 0) continue
                val denom = qFreq + k1 * (1 - b + b * docLen[i] / avgdl)
                scores[i] += wordIdf * (qFreq * (k1 + 1) / denom)
            }
        }
        return scores
    }

    private fun cosineSim(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot // both sides are already L2-normalized, so dot == cosine similarity
    }

    data class Result(val chunk: KbChunk, val rrfScore: Double)

    /** Hybrid retrieve: dense (ONNX embedder) + sparse (BM25) fused via adaptive RRF. */
    fun retrieve(query: String, k: Int = 5, fetchK: Int = 20): List<Result> {
        if (chunks.isEmpty()) return emptyList()

        val queryType = detectQueryType(query)
        val (denseWeight, sparseWeight) = adaptiveWeights.getValue(queryType)

        // Dense ranking.
        val queryEmbedding = embedder.embed(query)
        val denseRanked = chunks.indices
            .map { i -> i to cosineSim(queryEmbedding, chunks[i].embedding) }
            .sortedByDescending { it.second }
            .take(fetchK)

        // Sparse ranking.
        val queryTokens = tokenize(query)
        val sparseScores = bm25Scores(queryTokens)
        val sparseRanked = chunks.indices
            .map { i -> i to sparseScores[i] }
            .filter { it.second > 0.0 }
            .sortedByDescending { it.second }
            .take(fetchK)

        // RRF fusion — score(d) = sum(weight_i / (rrfK + rank_i(d))), rank starts at 1.
        val rrfK = 60
        val fused = HashMap<Int, Double>()
        denseRanked.forEachIndexed { idx, (docIdx, _) ->
            val rank = idx + 1
            fused[docIdx] = (fused[docIdx] ?: 0.0) + denseWeight * (1.0 / (rrfK + rank))
        }
        sparseRanked.forEachIndexed { idx, (docIdx, _) ->
            val rank = idx + 1
            fused[docIdx] = (fused[docIdx] ?: 0.0) + sparseWeight * (1.0 / (rrfK + rank))
        }

        return fused.entries
            .sortedByDescending { it.value }
            .take(k)
            .map { (idx, score) -> Result(chunks[idx], score) }
    }
}
