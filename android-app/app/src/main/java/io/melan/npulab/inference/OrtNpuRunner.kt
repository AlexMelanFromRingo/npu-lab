package io.melan.npulab.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import kotlin.math.min

/**
 * Runs ANY ONNX model on the Hexagon NPU via ONNX Runtime + the QNN Execution
 * Provider — on-device, no PC conversion. This is a SECOND inference engine,
 * separate from the hand-built QNN context-binary pipeline (SD/Whisper/zoo):
 * here ORT loads a standard `.onnx`, the QNN EP partitions+compiles the
 * supported subgraphs to the NPU on the device, and unsupported ops fall back
 * to ORT CPU automatically.
 *
 * The onnxruntime-android-qnn AAR ships only libonnxruntime.so; it loads the
 * QNN backend by `backend_path`, pointed at OUR bundled libQnn*.so (2.46).
 */
object OrtNpuRunner {

    enum class Backend(val lib: String, val label: String) {
        HTP("libQnnHtp.so", "NPU (HTP)"),
        GPU("libQnnGpu.so", "GPU (QNN)"),
        CPU("libQnnCpu.so", "CPU (QNN)"),
    }

    data class IoSpec(val name: String, val shape: List<Long>, val type: String)

    data class Result(
        val ok: Boolean,
        val medianMs: Double,
        val p95Ms: Double,
        val meanMs: Double,
        val minMs: Double,
        val maxMs: Double,
        val loadMs: Long,
        val iters: Int,
        val inputs: List<IoSpec>,
        val outputs: List<IoSpec>,
        val error: String?,
    )

    /**
     * Compile (on first run) + benchmark [modelPath] with synthetic inputs.
     * @param nativeLibDir applicationInfo.nativeLibraryDir — where libQnnHtp.so lives.
     */
    fun benchmark(
        modelPath: String,
        nativeLibDir: String,
        backend: Backend,
        iters: Int = 8,
        warmup: Int = 2,
    ): Result {
        val env = OrtEnvironment.getEnvironment()
        var session: OrtSession? = null
        val tensors = ArrayList<OnnxTensor>()
        try {
            val opts = OrtSession.SessionOptions()
            // Cache the on-device-compiled context next to the model so the next
            // load skips the (slow) QNN graph compilation.
            opts.addConfigEntry("ep.context_enable", "1")
            val qnn = HashMap<String, String>()
            qnn["backend_path"] = "$nativeLibDir/${backend.lib}"
            if (backend == Backend.HTP) qnn["htp_performance_mode"] = "burst"
            opts.addQnn(qnn)

            val tLoad = System.nanoTime()
            session = env.createSession(modelPath, opts)
            val loadMs = (System.nanoTime() - tLoad) / 1_000_000

            val inSpecs = ArrayList<IoSpec>()
            val feed = LinkedHashMap<String, OnnxTensor>()
            for ((name, node) in session.inputInfo) {
                val ti = node.info as TensorInfo
                val shape = ti.shape.map { if (it < 0) 1L else it }.toLongArray()
                val t = synthTensor(env, ti.type, shape)
                feed[name] = t; tensors.add(t)
                inSpecs.add(IoSpec(name, shape.toList(), ti.type.name))
            }

            repeat(warmup) { session.run(feed).close() }
            val samples = DoubleArray(iters)
            for (i in 0 until iters) {
                val t0 = System.nanoTime()
                session.run(feed).close()
                samples[i] = (System.nanoTime() - t0) / 1e6
            }
            samples.sort()

            val outSpecs = session.outputInfo.map { (n, node) ->
                val ti = node.info as TensorInfo
                IoSpec(n, ti.shape.toList(), ti.type.name)
            }
            return Result(
                ok = true,
                medianMs = samples[samples.size / 2],
                p95Ms = samples[min(samples.size - 1, (samples.size * 95) / 100)],
                meanMs = samples.average(),
                minMs = samples.first(),
                maxMs = samples.last(),
                loadMs = loadMs,
                iters = iters,
                inputs = inSpecs,
                outputs = outSpecs,
                error = null,
            )
        } catch (t: Throwable) {
            return Result(false, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, emptyList(), emptyList(),
                (t.message ?: t::class.java.simpleName).take(400))
        } finally {
            tensors.forEach { runCatching { it.close() } }
            runCatching { session?.close() }
        }
    }

    private fun synthTensor(env: OrtEnvironment, type: OnnxJavaType, shape: LongArray): OnnxTensor {
        val n = shape.fold(1L) { a, b -> a * b }.toInt().coerceAtLeast(1)
        return when (type) {
            OnnxJavaType.INT64 -> OnnxTensor.createTensor(env, LongBuffer.allocate(n), shape)
            OnnxJavaType.INT32 -> OnnxTensor.createTensor(env, IntBuffer.allocate(n), shape)
            else -> OnnxTensor.createTensor(env, FloatBuffer.allocate(n), shape) // FLOAT + fallback
        }
    }
}
