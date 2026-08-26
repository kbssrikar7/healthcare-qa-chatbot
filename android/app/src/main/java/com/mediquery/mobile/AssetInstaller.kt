package com.mediquery.mobile

import android.content.Context
import java.io.File

/**
 * Copies a large bundled asset (the LLM model, the KB subset) out to a real
 * filesystem path on first run. Both LiteRT-LM's `EngineConfig.modelPath` and
 * `KnowledgeBase.load()` need an actual file, not an asset stream — assets
 * only expose a real fd via `AssetManager.openFd()`, and only when stored
 * uncompressed (see the `noCompress` block in app/build.gradle.kts).
 *
 * Idempotent: if [destFile] already exists with the exact expected size, the
 * copy is skipped — this makes every app launch after the first one instant,
 * not just the very first.
 */
object AssetInstaller {
    fun ensureInstalled(
        context: Context,
        assetName: String,
        destFile: File,
        onProgress: (copiedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): File {
        val totalBytes = context.assets.openFd(assetName).use { it.length }

        if (destFile.exists() && destFile.length() == totalBytes) {
            return destFile
        }

        destFile.parentFile?.mkdirs()
        val tmpFile = File(destFile.parentFile, "${destFile.name}.part")
        context.assets.open(assetName).use { input ->
            tmpFile.outputStream().use { output ->
                val buffer = ByteArray(1 shl 20) // 1MB
                var copied = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)
                    copied += read
                    onProgress(copied, totalBytes)
                }
            }
        }
        if (!tmpFile.renameTo(destFile)) {
            tmpFile.copyTo(destFile, overwrite = true)
            tmpFile.delete()
        }
        return destFile
    }
}
