package io.melan.npulab.install

import android.content.Context
import android.util.Log
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.yield
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

private const val TAG = "ModelInstaller"

/**
 * Streams model installation as a flow of [Progress] events.
 *
 * Flow:
 *   1. HEAD the zip to learn total size.
 *   2. Stream-download to a .tmp file under cache, emitting Downloading events.
 *   3. Unzip on the fly: walk entries, write only those listed in extractMap to
 *      the final ModelStore location.
 *   4. Download aux files (HF tokenizer for SD).
 *   5. Emit Installed.
 *
 * Cancellation: caller cancels the collector; we observe CancellationException
 * and clean up partial files.
 */
class ModelInstaller(private val ctx: Context) {

    sealed interface Progress {
        data class Downloading(val bytesDone: Long, val bytesTotal: Long, val phase: String) : Progress
        data object Extracting : Progress
        data class Installed(val totalBytes: Long) : Progress
        data class Failed(val message: String) : Progress
    }

    fun install(asset: ModelAsset): Flow<Progress> = flow {
        val source = asset.installSource ?: run {
            emit(Progress.Failed("This model has no public S3 mirror — an AI Hub account is required"))
            return@flow
        }
        val store = ModelStore(ctx)
        val zipTmp = File(ctx.cacheDir, "install_${asset.kind.name}.zip.tmp")
        var totalBytes = 0L
        try {
            // ── 1. Download the zip ─────────────────────────────────
            emit(Progress.Downloading(0, source.approxMib * 1024L * 1024L, "Downloading zip"))
            val conn = (URL(source.zipUrl).openConnection() as HttpURLConnection).apply {
                connectTimeout = 10_000
                readTimeout = 20_000
                requestMethod = "GET"
            }
            try {
                val total = conn.contentLengthLong.takeIf { it > 0 }
                    ?: (source.approxMib * 1024L * 1024L)
                conn.inputStream.use { input ->
                    FileOutputStream(zipTmp).use { output ->
                        val buf = ByteArray(64 * 1024)
                        var done = 0L
                        var lastEmit = 0L
                        while (true) {
                            val n = input.read(buf)
                            if (n <= 0) break
                            output.write(buf, 0, n)
                            done += n
                            if (done - lastEmit > 1 * 1024 * 1024) {
                                emit(Progress.Downloading(done, total, "Downloading zip"))
                                lastEmit = done
                                yield()
                            }
                        }
                        emit(Progress.Downloading(done, total, "Download complete"))
                    }
                }
            } finally {
                conn.disconnect()
            }

            // ── 2. Unzip selected members ───────────────────────────
            emit(Progress.Extracting)
            ZipInputStream(zipTmp.inputStream().buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) { zip.closeEntry(); continue }
                    val basename = entry.name.substringAfterLast('/')
                    val dest = source.extractMap.entries.firstOrNull { (suffix, _) ->
                        basename.endsWith(suffix) || entry.name.endsWith(suffix)
                    }?.value
                    if (dest == null) { zip.closeEntry(); continue }
                    val out = store.pathOf(dest).apply { parentFile?.mkdirs() }
                    FileOutputStream(out).use { fos ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            val n = zip.read(buf)
                            if (n <= 0) break
                            fos.write(buf, 0, n)
                        }
                    }
                    totalBytes += out.length()
                    Log.i(TAG, "extracted -> ${out.absolutePath} (${out.length() / (1024 * 1024)} MiB)")
                    zip.closeEntry()
                    yield()
                }
            }

            // ── 3. Aux files (tokenizer, etc.) ──────────────────────
            for (aux in source.auxFiles) {
                emit(Progress.Downloading(0, 0, "Aux: ${aux.destRelative}"))
                val dest = store.pathOf(aux.destRelative).apply { parentFile?.mkdirs() }
                val conn2 = (URL(aux.url).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 10_000
                    readTimeout = 15_000
                    requestMethod = "GET"
                }
                try {
                    conn2.inputStream.use { input ->
                        FileOutputStream(dest).use { output -> input.copyTo(output) }
                    }
                    totalBytes += dest.length()
                } finally {
                    conn2.disconnect()
                }
                yield()
            }

            emit(Progress.Installed(totalBytes))
        } catch (c: CancellationException) {
            emit(Progress.Failed("Cancelled"))
            throw c
        } catch (t: Throwable) {
            Log.e(TAG, "install ${asset.kind} failed", t)
            emit(Progress.Failed(t.message ?: t::class.java.simpleName))
        } finally {
            if (zipTmp.exists()) zipTmp.delete()
        }
    }.flowOn(Dispatchers.IO)

    /** Deletes the model's files from external storage. */
    fun uninstall(asset: ModelAsset): Int {
        val store = ModelStore(ctx)
        var deleted = 0
        for (rel in asset.expectedFiles) {
            val f = store.pathOf(rel)
            if (f.exists() && f.delete()) deleted += 1
            // try to remove empty parent dirs
            f.parentFile?.takeIf { it.isDirectory && (it.list()?.isEmpty() == true) }?.delete()
        }
        return deleted
    }
}
