package com.mediquery.mobile.retrieval

import org.json.JSONObject
import java.io.File

data class KbChunk(
    val id: String,
    val text: String,
    val source: String,
    val chunkId: Int,
    val embedding: FloatArray,
)

/**
 * Loads the mobile knowledge-base subset built by
 * evaluation/build_mobile_kb_subset.py (one JSON object per line: id, text,
 * source, chunk_id, embedding). Bundled as an app asset (`mobile_kb.jsonl`)
 * and copied out to a real file by AssetInstaller on first run — see
 * MainActivity.loadModel().
 */
object KnowledgeBase {
    fun load(path: String): List<KbChunk> {
        val file = File(path)
        if (!file.exists()) return emptyList()
        val chunks = ArrayList<KbChunk>()
        file.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val obj = JSONObject(line)
            val embJson = obj.getJSONArray("embedding")
            val embedding = FloatArray(embJson.length()) { i -> embJson.getDouble(i).toFloat() }
            chunks.add(
                KbChunk(
                    id = obj.getString("id"),
                    text = obj.getString("text"),
                    source = obj.optString("source", ""),
                    chunkId = obj.optInt("chunk_id", 0),
                    embedding = embedding,
                )
            )
        }
        return chunks
    }
}
