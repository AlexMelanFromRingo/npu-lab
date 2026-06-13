package io.melan.npulab.install

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import io.melan.npulab.inference.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

private const val TAG = "ModelImporter"

/**
 * Imports a user-picked model file into models/custom/ — the "no adb, just
 * pick a file from your phone" path. Accepts:
 *   - a raw .bin (HTP context binary) or .dlc (cross-backend),
 *   - a .zip containing one or more .bin/.dlc (+ optional labels.txt),
 *     e.g. an unmodified AI Hub release archive.
 *
 * Everything lands flat under models/custom/ and is auto-discovered by
 * [ModelStore.customAssets] on the Benchmark and Vision tabs.
 */
class ModelImporter(private val ctx: Context) {

    suspend fun import(uri: Uri): String = withContext(Dispatchers.IO) {
        try {
            val name = displayName(uri) ?: "model"
            val customDir = File(ModelStore(ctx).root, "custom").apply { mkdirs() }
            when {
                name.endsWith(".zip", ignoreCase = true) -> importZip(uri, customDir)
                name.endsWith(".bin", ignoreCase = true) ||
                    name.endsWith(".dlc", ignoreCase = true) ||
                    name.endsWith(".labels.txt", ignoreCase = true) -> {
                    copyTo(uri, File(customDir, sanitize(name)))
                    "Imported $name"
                }
                else -> "Unsupported file: $name — pick a .bin, .dlc or .zip"
            }
        } catch (t: Throwable) {
            Log.e(TAG, "import failed", t)
            "Import failed: ${t.message ?: t::class.java.simpleName}"
        }
    }

    private fun importZip(uri: Uri, customDir: File): String {
        var imported = 0
        ctx.contentResolver.openInputStream(uri).use { raw ->
            ZipInputStream(raw!!.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) { zip.closeEntry(); continue }
                    val base = entry.name.substringAfterLast('/')
                    val keep = base.endsWith(".bin") || base.endsWith(".dlc") ||
                        base.endsWith(".labels.txt") || base == "labels.txt"
                    if (!keep) { zip.closeEntry(); continue }
                    val out = File(customDir, sanitize(base))
                    FileOutputStream(out).use { fos ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            val n = zip.read(buf); if (n <= 0) break; fos.write(buf, 0, n)
                        }
                    }
                    if (base.endsWith(".bin") || base.endsWith(".dlc")) imported++
                    zip.closeEntry()
                }
            }
        }
        return if (imported > 0) "Imported $imported model file(s) from zip"
        else "No .bin/.dlc found inside the zip"
    }

    private fun copyTo(uri: Uri, dest: File) {
        ctx.contentResolver.openInputStream(uri).use { input ->
            FileOutputStream(dest).use { output -> input!!.copyTo(output) }
        }
    }

    private fun displayName(uri: Uri): String? {
        ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) return c.getString(idx)
        }
        return uri.lastPathSegment?.substringAfterLast('/')
    }

    /** Strip any path components / odd chars a content provider might hand us. */
    private fun sanitize(name: String): String =
        name.substringAfterLast('/').substringAfterLast('\\')
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
}
