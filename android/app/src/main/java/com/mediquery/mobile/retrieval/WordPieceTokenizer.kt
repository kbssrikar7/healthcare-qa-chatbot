package com.mediquery.mobile.retrieval

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Minimal BERT WordPiece tokenizer for all-MiniLM-L6-v2 (uncased), matching the
 * HF `BertTokenizer` config: do_lower_case=true, standard punctuation splitting,
 * greedy longest-match WordPiece with "##" continuation prefix. No external
 * tokenizer library — the algorithm is small and this avoids a heavy dependency
 * for one model.
 */
class WordPieceTokenizer(context: Context, assetName: String = "minilm_vocab.txt") {

    private val vocab: Map<String, Int>
    private val clsId: Int
    private val sepId: Int
    private val padId: Int
    private val unkId: Int
    private val unkToken = "[UNK]"

    init {
        val map = HashMap<String, Int>(32000)
        context.assets.open(assetName).use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).useLines { lines ->
                lines.forEachIndexed { index, line -> map[line.trim()] = index }
            }
        }
        vocab = map
        clsId = vocab["[CLS]"] ?: error("[CLS] not found in vocab")
        sepId = vocab["[SEP]"] ?: error("[SEP] not found in vocab")
        padId = vocab["[PAD]"] ?: error("[PAD] not found in vocab")
        unkId = vocab[unkToken] ?: error("[UNK] not found in vocab")
    }

    data class Encoded(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray,
    )

    /** Tokenize [text], truncate/pad to [maxLength], and return model-ready arrays. */
    fun encode(text: String, maxLength: Int = 256): Encoded {
        val wordPieceIds = tokenizeToIds(text)
        // Reserve room for [CLS] and [SEP].
        val truncated = wordPieceIds.take(maxLength - 2)

        val ids = LongArray(maxLength)
        val mask = LongArray(maxLength)
        val typeIds = LongArray(maxLength) // all zeros — single segment

        ids[0] = clsId.toLong()
        mask[0] = 1L
        var pos = 1
        for (id in truncated) {
            ids[pos] = id.toLong()
            mask[pos] = 1L
            pos++
        }
        ids[pos] = sepId.toLong()
        mask[pos] = 1L
        pos++
        while (pos < maxLength) {
            ids[pos] = padId.toLong() // attention_mask stays 0 for these
            pos++
        }
        return Encoded(ids, mask, typeIds)
    }

    private fun tokenizeToIds(text: String): List<Int> {
        val basicTokens = basicTokenize(text.lowercase())
        val ids = ArrayList<Int>(basicTokens.size * 2)
        for (token in basicTokens) {
            ids.addAll(wordpieceTokenize(token))
        }
        return ids
    }

    /** Whitespace split, then split off punctuation as standalone tokens. */
    private fun basicTokenize(text: String): List<String> {
        val tokens = ArrayList<String>()
        for (whitespaceToken in text.split(Regex("\\s+")).filter { it.isNotEmpty() }) {
            var current = StringBuilder()
            for (ch in whitespaceToken) {
                if (isPunctuation(ch)) {
                    if (current.isNotEmpty()) {
                        tokens.add(current.toString())
                        current = StringBuilder()
                    }
                    tokens.add(ch.toString())
                } else {
                    current.append(ch)
                }
            }
            if (current.isNotEmpty()) tokens.add(current.toString())
        }
        return tokens
    }

    private fun isPunctuation(ch: Char): Boolean {
        val code = ch.code
        // Matches BERT's is_punctuation: ASCII punctuation ranges plus Unicode punctuation.
        if ((code in 33..47) || (code in 58..64) || (code in 91..96) || (code in 123..126)) {
            return true
        }
        return Character.getType(ch).let {
            it == Character.CONNECTOR_PUNCTUATION.toInt() ||
                it == Character.DASH_PUNCTUATION.toInt() ||
                it == Character.START_PUNCTUATION.toInt() ||
                it == Character.END_PUNCTUATION.toInt() ||
                it == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
                it == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                it == Character.OTHER_PUNCTUATION.toInt()
        }
    }

    /** Greedy longest-match-first WordPiece over a single basic token. */
    private fun wordpieceTokenize(token: String): List<Int> {
        if (token.length > 200) return listOf(unkId) // matches BERT's max_input_chars_per_word
        val result = ArrayList<Int>()
        var start = 0
        while (start < token.length) {
            var end = token.length
            var matched: Int? = null
            while (start < end) {
                var candidate = token.substring(start, end)
                if (start > 0) candidate = "##$candidate"
                val id = vocab[candidate]
                if (id != null) {
                    matched = id
                    break
                }
                end--
            }
            if (matched == null) return listOf(unkId)
            result.add(matched)
            start = end
        }
        return result
    }
}
