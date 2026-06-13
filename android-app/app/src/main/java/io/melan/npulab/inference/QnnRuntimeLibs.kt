package io.melan.npulab.inference

import android.content.Context
import android.system.Os
import android.util.Log
import java.io.File

/**
 * Points the Hexagon DSP loader at the QNN runtime libraries we ship.
 *
 * Deployment model (canonical, used by Qualcomm's ai-hub-apps ChatApp):
 *   - ALL QNN libs — including the Hexagon-side `libQnnHtpV81Skel.so` and
 *     `libQnnHtpV81.so` (32-bit DSP6 ELFs) — live in `jniLibs/arm64-v8a/`.
 *   - `packaging { jniLibs.useLegacyPackaging = true }` makes the installer
 *     extract them into `applicationInfo.nativeLibraryDir` as real files.
 *   - ADSP_LIBRARY_PATH (a ';'-separated list read by the FastRPC loader from
 *     the host process env) gets nativeLibraryDir first, then the device's
 *     stock vendor DSP roots as fallbacks.
 *
 * Earlier revisions carried the DSP ELFs in assets/ and extracted them to
 * filesDir at first launch. That broke silently: AssetManager.openFd() throws
 * on deflate-compressed assets (.so is not in AAPT's default noCompress list),
 * so the extraction failed and the DSP never saw the skel. jniLibs has no such
 * failure mode and is what Qualcomm's own apps do.
 *
 * MUST run before System.loadLibrary("npulab"): the QNN HTP dispatcher reads
 * ADSP_LIBRARY_PATH once during its first FastRPC session setup; setting it
 * later produces the misleading rc=14001 (DEVICE_INVALID_CONFIG — actually
 * "DSP skeleton not found").
 */
object QnnRuntimeLibs {

    private const val TAG = "QnnRuntimeLibs"

    private val SKEL_RE = Regex("libQnnHtpV(\\d+)Skel\\.so")

    /** The standard ADSP search list appended after nativeLibraryDir. */
    private const val ADSP_FALLBACKS =
        "/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;" +
        "/vendor/dsp/dsp;/vendor/dsp/images;/dsp"

    /** Idempotent — safe to call on every process start. */
    fun setup(context: Context) {
        val libDir = context.applicationInfo.nativeLibraryDir
        try {
            val dspPath = "$libDir;$ADSP_FALLBACKS"
            // Historic name, read by FastRPC for every DSP domain…
            Os.setenv("ADSP_LIBRARY_PATH", dspPath, true)
            // …but newer FastRPC builds consult domain-specific / generic
            // variants. HTP lives on the compute DSP (CDSP). Setting all
            // three is harmless and removes a whole class of "skel not
            // found" failures on new firmware.
            Os.setenv("CDSP_LIBRARY_PATH", dspPath, true)
            Os.setenv("DSP_LIBRARY_PATH", dspPath, true)
            val existing = Os.getenv("LD_LIBRARY_PATH").orEmpty()
            val ld = if (existing.isEmpty()) libDir else "$libDir:$existing"
            Os.setenv("LD_LIBRARY_PATH", ld, true)
            Log.i(TAG, "ADSP/CDSP/DSP_LIBRARY_PATH=$dspPath")
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to set DSP/LD env", t)
        }
        val arches = bundledArches(context)
        val want = QnnRuntime.resolveHtpArch()
        Log.i(TAG, "Bundled HTP arches: ${arches.ifEmpty { listOf("NONE") }}; device wants v$want")
        if (arches.isEmpty()) {
            Log.e(TAG, "No libQnnHtpV*Skel.so in $libDir — NPU will not initialize. " +
                "Run scripts/copy-qnn-libs.sh and check useLegacyPackaging=true.")
        } else if (want > 0 && want !in arches) {
            Log.w(TAG, "This device's HTP arch v$want has no bundled skel " +
                "(bundled: $arches). Rebuild with HTP_ARCHES including v$want.")
        }
    }

    /** Absolute path of the dir holding the QNN libs. Passed to native init. */
    fun runtimeDir(context: Context): String = context.applicationInfo.nativeLibraryDir

    /** HTP arch numbers whose skel is bundled (e.g. [73, 75, 79, 81]). */
    fun bundledArches(context: Context): List<Int> {
        val libDir = File(context.applicationInfo.nativeLibraryDir)
        return libDir.listFiles()?.mapNotNull {
            SKEL_RE.matchEntire(it.name)?.groupValues?.get(1)?.toIntOrNull()
        }?.sorted() ?: emptyList()
    }

    /**
     * True if this device's HTP architecture has a matching skel bundled.
     * When false the NPU path will fail and the Device screen can say why.
     */
    fun deviceArchSupported(context: Context): Boolean {
        val want = QnnRuntime.resolveHtpArch()
        return want > 0 && want in bundledArches(context)
    }
}
