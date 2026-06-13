package io.melan.npulab.inference

import android.content.Context
import java.io.File

/**
 * On-device compiler: turns a device-agnostic DLC into a chip-specific QNN
 * **context binary** right on the phone.
 *
 * A DLC is compiled (composed + finalized) for the connected Hexagon arch every
 * time it loads — that prepare step is the slow part. Here we do it once and
 * serialize the finalized context to a `.bin` (via QnnContext_getBinary). The
 * result lands in `models/custom/` and is auto-discovered by the Benchmark /
 * Vision tabs as a fast-loading, HTP-only context binary — exactly the format
 * AI Hub ships, but produced locally for *this* chip.
 *
 * ONNX→DLC still needs the x86 `qairt-converter` (PC). This is the second,
 * device-side half of the toolchain.
 */
object OnDeviceCompiler {

    data class CompileResult(
        val name: String,
        val ok: Boolean,
        val composeMs: Long,
        val outBytes: Long,
        val error: String?,
    )

    /** Tag for the active HTP arch, e.g. "htp81". 0 → generic "htp". */
    fun archTag(): String {
        val a = QnnRuntime.resolveHtpArch()
        return if (a > 0) "htp$a" else "htp"
    }

    private val COMPILED_RE = Regex(""".*\.htp\d*\.bin""")

    /** Relative path for a compiled binary — pure, unit-tested. */
    internal fun compiledRelPath(dlcRel: String, archTag: String): String {
        val base = File(dlcRel).name.removeSuffix(".dlc")
        return "custom/$base.$archTag.bin"
    }

    /** True if [fileName] looks like one of our compiled binaries. */
    internal fun isCompiledBinary(fileName: String): Boolean = COMPILED_RE.matches(fileName)

    /** Installed single-graph DLC models: (displayName, on-device relative path). */
    fun installedDlcs(ctx: Context): List<Pair<String, String>> {
        val store = ModelStore(ctx)
        val catalog = ModelCatalog.all
            .filter { it.isDlc && store.isInstalled(it) && it.expectedFiles.size == 1 }
            .map { it.displayName to it.expectedFiles.first() }
        val custom = store.customAssets()
            .filter { it.isDlc }
            .map { it.displayName to it.expectedFiles.first() }
        return (catalog + custom).distinctBy { it.second }
    }

    /** Where a compiled binary for [dlcRel] goes, e.g. custom/mobilenet_v2.htp81.bin. */
    fun outputFor(ctx: Context, dlcRel: String): File =
        ModelStore(ctx).pathOf(compiledRelPath(dlcRel, archTag()))

    /**
     * Compile every installed DLC to a context binary. One HTP runtime for the
     * whole batch (backend bring-up is the expensive part). [onProgress] fires
     * before each model with (index, total, name).
     */
    fun compileAll(ctx: Context, onProgress: (Int, Int, String) -> Unit): List<CompileResult> {
        val store = ModelStore(ctx)
        val dlcs = installedDlcs(ctx)
        if (dlcs.isEmpty()) return emptyList()

        val results = ArrayList<CompileResult>(dlcs.size)
        val rt = QnnRuntime(
            backend = NpuLabNative.Backend.HTP,
            nativeLibDir = QnnRuntimeLibs.runtimeDir(ctx),
        )
        rt.use {
            dlcs.forEachIndexed { i, (name, dlcRel) ->
                onProgress(i + 1, dlcs.size, name)
                results += compileOne(rt, store, ctx, name, dlcRel)
            }
        }
        return results
    }

    private fun compileOne(
        rt: QnnRuntime, store: ModelStore, ctx: Context, name: String, dlcRel: String,
    ): CompileResult {
        return try {
            val dlc = store.pathOf(dlcRel)
            if (!dlc.isFile) return CompileResult(name, false, 0, 0, "missing: $dlcRel")
            val out = outputFor(ctx, dlcRel).apply { parentFile?.mkdirs() }
            val t0 = System.nanoTime()
            val model = rt.loadModel(dlc.absolutePath)   // composes on-device
            val composeMs = (System.nanoTime() - t0) / 1_000_000
            model.use { it.serializeTo(out.absolutePath) }
            CompileResult(name, true, composeMs, out.length(), null)
        } catch (t: Throwable) {
            CompileResult(name, false, 0, 0, (t.message ?: t::class.java.simpleName).take(160))
        }
    }

    /** Total bytes of compiled binaries in models/custom/. */
    fun cacheBytes(ctx: Context): Long {
        val dir = File(ModelStore(ctx).root, "custom")
        return dir.listFiles { f -> f.isFile && COMPILED_RE.matches(f.name) }
            ?.sumOf { it.length() } ?: 0L
    }

    /** Delete compiled binaries (the DLC sources stay). Returns count removed. */
    fun clearCache(ctx: Context): Int {
        val dir = File(ModelStore(ctx).root, "custom")
        return dir.listFiles { f -> f.isFile && COMPILED_RE.matches(f.name) }
            ?.count { it.delete() } ?: 0
    }
}
