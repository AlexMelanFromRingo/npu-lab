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

    /** Hexagon-side libraries that must be visible to the DSP loader. */
    private val DSP_LIBS = listOf("libQnnHtpV81Skel.so", "libQnnHtpV81.so")

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
        for (missing in missingDspLibs(context)) {
            Log.e(
                TAG,
                "$missing not found in $libDir — NPU will not initialize. " +
                    "Check that jniLibs/arm64-v8a contains it and that " +
                    "useLegacyPackaging=true is set."
            )
        }
    }

    /** Absolute path of the dir holding the QNN libs. Passed to native init. */
    fun runtimeDir(context: Context): String = context.applicationInfo.nativeLibraryDir

    /**
     * DSP-side libs that did NOT get extracted to nativeLibraryDir. Empty list
     * means the Hexagon loader can find everything. Surfaced on the Device
     * screen so a broken install is visible without logcat.
     */
    fun missingDspLibs(context: Context): List<String> {
        val libDir = File(context.applicationInfo.nativeLibraryDir)
        return DSP_LIBS.filterNot { File(libDir, it).isFile }
    }
}
