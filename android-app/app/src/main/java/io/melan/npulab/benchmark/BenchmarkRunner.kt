package io.melan.npulab.benchmark

import android.content.Context
import android.util.Log
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.inference.QnnRuntime
import io.melan.npulab.inference.TensorDtype
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

private const val TAG = "Benchmark"

data class BenchmarkRow(
    val modelName: String,
    val backend: NpuLabNative.Backend,
    val iterations: Int,
    val medianUs: Long,
    val p95Us: Long,
    val meanUs: Long,
    val minUs: Long,
    val maxUs: Long,
    val error: String?,
)

/**
 * Runs the *first* graph of each model N times on each requested backend, reporting
 * percentile latency. We don't compare numeric outputs across backends — those
 * differ slightly because of quantization. This is purely a wall-time benchmark.
 */
class BenchmarkRunner(private val ctx: Context) {

    fun run(
        models: List<ModelAsset>,
        backends: List<NpuLabNative.Backend>,
        iterationsPerCombination: Int = 8,
        warmupIterations: Int = 2,
        onProgress: ((BenchmarkRow) -> Unit)? = null,
    ): List<BenchmarkRow> {
        val store = ModelStore(ctx)
        val results = mutableListOf<BenchmarkRow>()
        for (backend in backends) {
            val runtime = try {
                QnnRuntime(
                    backend = backend,
                    nativeLibDir = io.melan.npulab.inference.QnnRuntimeLibs.runtimeDir(ctx),
                )
            } catch (t: Throwable) {
                Log.w(TAG, "Skipping backend $backend: ${t.message}")
                models.forEach { m ->
                    val row = BenchmarkRow(m.displayName, backend, 0, 0, 0, 0, 0, 0,
                        error = "backend init failed: ${t.message}")
                    results += row
                    onProgress?.invoke(row)
                }
                continue
            }
            runtime.use { rt ->
                for (asset in models) {
                    val row = runOne(rt, store, asset, backend, iterationsPerCombination, warmupIterations)
                    results += row
                    onProgress?.invoke(row)
                }
            }
        }
        return results
    }

    private fun runOne(
        runtime: QnnRuntime,
        store: ModelStore,
        asset: ModelAsset,
        backend: NpuLabNative.Backend,
        iters: Int,
        warmup: Int,
    ): BenchmarkRow {
        // Pick the first .bin file as the model to load — single-graph models only.
        val binRel = asset.expectedFiles.firstOrNull { it.endsWith(".bin") }
            ?: return BenchmarkRow(asset.displayName, backend, 0, 0, 0, 0, 0, 0,
                error = "no .bin in catalog entry")
        val binFile = store.pathOf(binRel)
        if (!binFile.exists()) {
            return BenchmarkRow(asset.displayName, backend, 0, 0, 0, 0, 0, 0,
                error = "missing on device: $binRel")
        }
        val model = try {
            runtime.loadModel(binFile.absolutePath)
        } catch (t: Throwable) {
            return BenchmarkRow(asset.displayName, backend, 0, 0, 0, 0, 0, 0,
                error = "load failed: ${t.message}")
        }
        model.use { m ->
            val inputs = m.inputs.map { spec ->
                val buf = ByteBuffer.allocateDirect(spec.byteSize).order(ByteOrder.nativeOrder())
                // Fill with deterministic non-zero pattern so quantized models don't shortcut.
                when (spec.dtype) {
                    TensorDtype.FLOAT32 -> repeat(spec.numel) { buf.putFloat(0.1f) }
                    TensorDtype.FLOAT16 -> repeat(spec.numel) { buf.putShort(0x2E66.toShort()) /*~=0.1*/ }
                    TensorDtype.INT32, TensorDtype.UINT32 -> repeat(spec.numel) { buf.putInt(1) }
                    TensorDtype.INT8, TensorDtype.UINT8,
                    TensorDtype.SFIXED8, TensorDtype.UFIXED8 -> repeat(spec.numel) { buf.put(64) }
                    TensorDtype.INT16, TensorDtype.UINT16,
                    TensorDtype.SFIXED16, TensorDtype.UFIXED16 -> repeat(spec.numel) { buf.putShort(64) }
                }
                buf.rewind()
                buf
            }.toTypedArray()
            val outputs = m.allocOutputs()
            // warmup
            repeat(warmup) { m.execute(inputs, outputs) }
            // measure
            val samples = LongArray(iters)
            for (i in 0 until iters) {
                val us = m.execute(inputs, outputs)
                if (us < 0) {
                    return BenchmarkRow(asset.displayName, backend, i, 0, 0, 0, 0, 0,
                        error = "execute returned -1 at iter $i")
                }
                samples[i] = us
            }
            samples.sort()
            val median = samples[samples.size / 2]
            val p95 = samples[min(samples.size - 1, (samples.size * 95) / 100)]
            val mean = samples.sum() / samples.size
            return BenchmarkRow(asset.displayName, backend, iters, median, p95, mean,
                samples.first(), samples.last(), error = null)
        }
    }
}
