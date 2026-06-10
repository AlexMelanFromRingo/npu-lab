package io.melan.npulab.inference

/**
 * JNI bridge to the C++ QNN runtime in app/src/main/cpp/.
 *
 * Lifecycle:
 *   1. [initBackend] — load libQnnSystem.so + libQnnHtp.so (or Cpu/Gpu equivalents)
 *      and resolve QnnInterface_getProviders. Returns an opaque backend handle (Long).
 *   2. [loadContextBinary] — read a serialized context binary (.bin file produced
 *      by Qualcomm AI Hub or `qnn-context-binary-utility`) and deserialize it
 *      against the backend. Returns an opaque context handle.
 *   3. [getGraphInputs] / [getGraphOutputs] — describe tensor names/shapes/dtypes.
 *   4. [execute] — run inference. Inputs and outputs are flat ByteBuffers ordered
 *      to match the names from [getGraphInputs] / [getGraphOutputs].
 *   5. [freeContext] / [freeBackend] — release handles when done.
 *
 * All native methods may throw [QnnException] (a Java RuntimeException subclass)
 * with the underlying QNN error code wrapped as the message. They are blocking
 * and should be called off the main thread.
 */
object NpuLabNative {

    init {
        // libnpulab.so is built by CMake (see app/src/main/cpp/CMakeLists.txt) and
        // depends on libc++_shared.so + the QNN backend libs (libQnnHtp.so etc.)
        // that ship in src/main/jniLibs/arm64-v8a/.
        System.loadLibrary("npulab")
    }

    enum class Backend(val nativeId: Int) {
        HTP(0),     // Hexagon Tensor Processor — the NPU
        GPU(1),     // Adreno GPU through QNN-GPU backend
        CPU(2),     // Reference CPU backend (slow, for sanity-checking)
    }

    /**
     * Initialize a QNN backend. Returns a handle to be passed back into other
     * methods. The handle is 0 on failure (and an error is logged via __android_log_print).
     *
     * @param backend which backend to initialize
     * @param htpArch HTP architecture code (e.g. 81 for v81 on Snapdragon 8 Elite Gen 5).
     *                Ignored unless backend == HTP.
     * @param socModel QNN SoC model code (see QnnTypes.h). For S26 Ultra (SM8850) = 87.
     *                 0 = leave at default (no custom config).
     * @param nativeLibDir applicationInfo.nativeLibraryDir — where the installer
     *                     extracted all libQnn*.so incl. the Hexagon-side
     *                     libQnnHtpV81Skel.so. The dispatcher mirrors this into
     *                     LD_LIBRARY_PATH. The companion ADSP_LIBRARY_PATH is
     *                     already set by [QnnRuntimeLibs.setup] in Application.
     */
    external fun initBackend(backend: Int, htpArch: Int, socModel: Int, nativeLibDir: String): Long

    /**
     * Load a serialized context binary from a file. The binary must have been
     * produced by qnn-context-binary-utility or by Qualcomm AI Hub `--target_runtime qnn`.
     */
    external fun loadContextBinary(backendHandle: Long, path: String): Long

    /** Return ["name:dtype:shape", ...] for the graph's inputs in declared order. */
    external fun getGraphInputs(contextHandle: Long, graphIndex: Int): Array<String>

    /** Return ["name:dtype:shape", ...] for the graph's outputs in declared order. */
    external fun getGraphOutputs(contextHandle: Long, graphIndex: Int): Array<String>

    /**
     * Execute the graph. inputs / outputs are direct ByteBuffers; their order must
     * match the order returned by [getGraphInputs] / [getGraphOutputs]. Returns
     * wall-clock execution time in microseconds, or -1 on failure.
     */
    external fun execute(
        contextHandle: Long,
        graphIndex: Int,
        inputs: Array<java.nio.ByteBuffer>,
        outputs: Array<java.nio.ByteBuffer>,
    ): Long

    /** Returns a JSON blob with backend version, HTP arch, available memory, etc. */
    external fun queryDeviceInfo(backendHandle: Long): String

    /**
     * Field diagnostics for the NPU path. Runs every deviceCreate flavor
     * (explicit unsigned PD / default / SOC+ARCH) and tries to deserialize
     * [modelPath] under each, capturing the QNN internal log. Returns a
     * copy-pasteable text report. Blocking — call off the main thread.
     */
    external fun npuSelfTest(nativeLibDir: String, modelPath: String): String

    external fun freeContext(contextHandle: Long)
    external fun freeBackend(backendHandle: Long)
}

class QnnException(message: String) : RuntimeException(message)
