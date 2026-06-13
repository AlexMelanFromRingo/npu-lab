package io.melan.npulab.inference

import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "QnnRuntime"

/**
 * Thin idiomatic Kotlin wrapper around [NpuLabNative]. Each instance owns a
 * single backend + a list of loaded contexts. Use [use] / try-finally to free.
 */
class QnnRuntime(
    val backend: NpuLabNative.Backend,
    htpArch: Int = resolveHtpArch(),
    socModel: Int = resolveSocModel(),
    nativeLibDir: String = "",
) : Closeable {

    private val backendHandle: Long = NpuLabNative.initBackend(
        backend.nativeId, htpArch, socModel, nativeLibDir
    ).also {
        if (it == 0L) throw QnnException("QNN backend ${backend.name} failed to initialize")
    }
    private val contexts = mutableListOf<QnnModel>()

    fun loadModel(path: String): QnnModel {
        val handle = NpuLabNative.loadContextBinary(backendHandle, path)
        if (handle == 0L) throw QnnException("Failed to load context binary: $path")
        val inputs = NpuLabNative.getGraphInputs(handle, 0).map(::parseTensorSpec)
        val outputs = NpuLabNative.getGraphOutputs(handle, 0).map(::parseTensorSpec)
        Log.i(TAG, "Loaded $path — inputs=${inputs.size} outputs=${outputs.size}")
        return QnnModel(handle, inputs, outputs).also { contexts.add(it) }
    }

    fun deviceInfoJson(): String = NpuLabNative.queryDeviceInfo(backendHandle)

    override fun close() {
        contexts.forEach { it.close() }
        contexts.clear()
        NpuLabNative.freeBackend(backendHandle)
    }

    companion object {
        // HTP arch codes used by libQnnHtpV*.so name suffix.
        // Mapping SoC → HTP arch is documented in QAIRT-Docs/QNN/general/overview.html
        // of the SDK. The first iteration of this code shipped with v79 for SM8850
        // (Snapdragon 8 Elite Gen 5) — that was wrong, the actual arch is v81.
        // Using v79 with an SM8850-compiled binary gives rc=14001 (DEVICE INVALID_CONFIG).
        const val HTP_ARCH_V81_S8E_GEN5 = 81
        const val HTP_ARCH_V79_S8_ELITE = 79      // Snapdragon 8 Elite (SM8750)
        const val HTP_ARCH_V75_S8_GEN3 = 75
        const val HTP_ARCH_V73_S8_GEN2 = 73

        /**
         * Map Build.SOC_MODEL (e.g. "SM8850") to the QNN_SOC_MODEL_* enum from
         * QnnTypes.h. Returns 0 for unknown SoCs — in which case the native
         * code skips the HTP custom config and creates a default device.
         *
         * If you hit "rc=14001 (DEVICE)" on a new chip not listed here, add the
         * corresponding mapping by grep'ing QNN_SOC_MODEL_SM* in
         * $QNN_SDK_ROOT/include/QNN/QnnTypes.h.
         */
        fun resolveSocModel(): Int {
            val name = (android.os.Build.SOC_MODEL ?: "").uppercase()
            return when {
                "SM8850" in name -> 87   // 8 Elite Gen 5 (S26 Ultra) — our primary target
                "SM8750" in name -> 69   // 8 Elite
                "SM8650" in name -> 57   // 8 Gen 3
                "SM8550" in name -> 43   // 8 Gen 2
                "SM8475" in name -> 42   // 8+ Gen 1
                "SM8450" in name -> 36   // 8 Gen 1
                "SM8350" in name -> 30   // 888
                else -> 0                 // unknown — let native fall back to no custom config
            }
        }

        /**
         * HTP arch matching [resolveSocModel]. Returns 0 for unmapped chips —
         * the native layer then skips the custom device config and lets QNN
         * auto-detect the real arch and load whichever bundled skel matches.
         */
        fun resolveHtpArch(): Int = when (resolveSocModel()) {
            87 -> HTP_ARCH_V81_S8E_GEN5  // 8 Elite Gen 5 (SM8850)
            69 -> HTP_ARCH_V79_S8_ELITE  // 8 Elite (SM8750)
            57 -> HTP_ARCH_V75_S8_GEN3   // 8 Gen 3 (SM8650)
            43 -> HTP_ARCH_V73_S8_GEN2   // 8 Gen 2 (SM8550)
            42, 36 -> 69                 // 8(+) Gen 1 — Hexagon v69
            else -> 0                    // unknown — QNN auto-detects on device
        }
    }
}

data class TensorSpec(
    val name: String,
    val dtype: TensorDtype,
    val shape: IntArray,
    /** Quantization scale for fixed-point dtypes. 0f for non-quantized. */
    val scale: Float = 0f,
    /** Quantization offset (signed) for fixed-point dtypes. */
    val offset: Int = 0,
) {
    val numel: Int get() = shape.fold(1) { acc, d -> acc * d }
    val byteSize: Int get() = numel * dtype.bytes
    val isQuantized: Boolean get() = dtype.isQuantized && scale != 0f
}

enum class TensorDtype(val bytes: Int, val isQuantized: Boolean = false) {
    FLOAT32(4), FLOAT16(2),
    INT8(1), UINT8(1), INT16(2), UINT16(2), INT32(4), UINT32(4),
    UFIXED8(1, isQuantized = true), SFIXED8(1, isQuantized = true),
    UFIXED16(2, isQuantized = true), SFIXED16(2, isQuantized = true);

    companion object {
        fun parse(s: String): TensorDtype = when (s.lowercase()) {
            "f32", "float32" -> FLOAT32
            "f16", "float16" -> FLOAT16
            "i8", "int8" -> INT8
            "u8", "uint8" -> UINT8
            "i16", "int16" -> INT16
            "u16", "uint16" -> UINT16
            "i32", "int32" -> INT32
            "u32", "uint32" -> UINT32
            "ufixed8" -> UFIXED8
            "sfixed8" -> SFIXED8
            "ufixed16" -> UFIXED16
            "sfixed16" -> SFIXED16
            else -> error("Unknown dtype $s")
        }
    }
}

internal fun parseTensorSpec(s: String): TensorSpec {
    // Wire format: "name:dtype:d0,d1,...:scale:offset"
    val parts = s.split(':')
    require(parts.size >= 3) { "Bad tensor spec: $s" }
    val name = parts[0]
    val dtype = TensorDtype.parse(parts[1])
    val shape = parts[2].split(',').map { it.trim().toInt() }.toIntArray()
    val scale = parts.getOrNull(3)?.toFloatOrNull() ?: 0f
    val offset = parts.getOrNull(4)?.toIntOrNull() ?: 0
    return TensorSpec(name, dtype, shape, scale, offset)
}

class QnnModel(
    private val handle: Long,
    val inputs: List<TensorSpec>,
    val outputs: List<TensorSpec>,
) : Closeable {

    fun allocOutputs(): Array<ByteBuffer> = outputs.map { spec ->
        ByteBuffer.allocateDirect(spec.byteSize).order(ByteOrder.nativeOrder())
    }.toTypedArray()

    /**
     * Run a single forward pass. Returns the execution wall-time in microseconds.
     * Caller is responsible for filling [inputBuffers] before calling and reading
     * from [outputBuffers] after. Both arrays must match the model's tensor order.
     */
    fun execute(
        inputBuffers: Array<ByteBuffer>,
        outputBuffers: Array<ByteBuffer>,
    ): Long {
        require(inputBuffers.size == inputs.size) {
            "Expected ${inputs.size} inputs, got ${inputBuffers.size}"
        }
        require(outputBuffers.size == outputs.size) {
            "Expected ${outputs.size} outputs, got ${outputBuffers.size}"
        }
        return NpuLabNative.execute(handle, 0, inputBuffers, outputBuffers)
    }

    /**
     * Serialize this (finalized) context to a fast-loading QNN context binary
     * at [path]. Meaningful after loading a DLC: the on-device-composed graph
     * is cached so the next load skips the slow prepare. Throws on failure.
     */
    fun serializeTo(path: String): Boolean = NpuLabNative.serializeContext(handle, path)

    override fun close() = NpuLabNative.freeContext(handle)
}
