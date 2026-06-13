package io.melan.npulab.vision

import android.content.Context
import android.graphics.Bitmap
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelCategory
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.inference.QnnRuntime
import io.melan.npulab.inference.QnnRuntimeLibs
import java.io.Closeable

/**
 * Runs a single-input image model from the zoo (or a custom .dlc/.bin) on the
 * NPU and turns its output into something showable: top-5 labels for
 * classifiers, a colored depth map, a segmentation overlay, or an upscaled
 * image for super-resolution. Geometry (NCHW/NHWC, H/W/C) is read from the
 * model; pixels are fed as RGB in [0,1] — the convention AI Hub image models
 * expect (ImageNet normalization is baked into the exported graph).
 */
class VisionPipeline(
    context: Context,
    private val asset: ModelAsset,
) : Closeable {

    private val store = ModelStore(context)
    private val runtime = QnnRuntime(
        backend = NpuLabNative.Backend.HTP,
        nativeLibDir = QnnRuntimeLibs.runtimeDir(context),
    )
    private val model = runtime.loadModel(
        store.pathOf(asset.primaryModelFile ?: error("asset has no model file")).absolutePath
    )

    private val labels: List<String>? =
        if (asset.category == ModelCategory.CLASSIFICATION)
            loadLabels(context, asset) else null

    val inputGeometry: InputGeometry =
        InputGeometry.parse(model.inputs[0].shape)
            ?: error("Unsupported input shape ${model.inputs[0].shape.toList()} for ${asset.displayName}")

    sealed interface Result {
        val inferenceMs: Long
        data class Labels(
            val top: List<Pair<String, Float>>,
            override val inferenceMs: Long,
        ) : Result
        data class Image(
            val bitmap: Bitmap,
            val caption: String,
            override val inferenceMs: Long,
        ) : Result
        data class Raw(val text: String, override val inferenceMs: Long) : Result
    }

    fun run(input: Bitmap): Result {
        val inSpec = model.inputs[0]
        var pixels = BitmapOps.toFloat01(input, inputGeometry)
        // Float / quantized image inputs expect RGB in [0,1] (AI Hub bakes the
        // ImageNet normalization into the graph). Raw integer-typed inputs want
        // [0,255]. Quantized (UFIXED) inputs keep [0,1] — encode() maps to the
        // quant grid via scale/offset.
        if (inSpec.dtype in INT_INPUT_TYPES) {
            pixels = FloatArray(pixels.size) { pixels[it] * 255f }
        }
        val inBuf = TensorIo.encode(inSpec, pixels)

        val outBufs = model.allocOutputs()
        val us = model.execute(arrayOf(inBuf), outBufs)
        val ms = us / 1000

        // Decode the first output (most zoo models are single-output).
        val outSpec = model.outputs[0]
        val out = TensorIo.decode(outSpec, outBufs[0])

        return when (asset.category) {
            ModelCategory.CLASSIFICATION -> classify(out, ms)
            ModelCategory.DEPTH -> depth(out, outSpec.shape, ms)
            ModelCategory.SEGMENTATION -> segment(out, outSpec.shape, ms)
            ModelCategory.SUPER_RESOLUTION -> superRes(out, outSpec.shape, ms)
            else -> Result.Raw(rawSummary(outSpec.shape, out), ms)
        }
    }

    private fun classify(out: FloatArray, ms: Long): Result {
        // Some exports omit the softmax; apply it for stable probabilities.
        val probs = TensorIo.softmax(out)
        val top = TensorIo.topK(probs, 5).map { i ->
            (labels?.getOrNull(classIndexToLabel(i, out.size)) ?: "class $i") to probs[i]
        }
        return Result.Labels(top, ms)
    }

    // 1001-class outputs (TF-style) have a leading "background" — shift by 1.
    private fun classIndexToLabel(i: Int, numClasses: Int): Int =
        if (numClasses == 1001) (i - 1).coerceAtLeast(0) else i

    private fun depth(out: FloatArray, shape: IntArray, ms: Long): Result {
        val (h, w) = spatialHW(shape, out.size)
        val bmp = BitmapOps.depthBitmap(out, w, h)
        return Result.Image(bmp, "Depth · ${w}×${h}", ms)
    }

    private fun segment(out: FloatArray, shape: IntArray, ms: Long): Result {
        // [1,C,H,W] or [1,H,W,C]. classes = the non-spatial, non-batch dim.
        val geo = parse4d(shape) ?: return Result.Raw(rawSummary(shape, out), ms)
        val classes = if (geo.nchw) geo.channels else geo.channels
        val pixels = geo.height * geo.width
        val cls = ImageOps.argmaxPerPixel(out, classes, pixels, channelFirst = geo.nchw)
        val bmp = BitmapOps.segmentationBitmap(cls, geo.width, geo.height)
        return Result.Image(bmp, "Segmentation · $classes classes · ${geo.width}×${geo.height}", ms)
    }

    private fun superRes(out: FloatArray, shape: IntArray, ms: Long): Result {
        val geo = parse4d(shape) ?: return Result.Raw(rawSummary(shape, out), ms)
        val bmp = BitmapOps.fromFloat01(out, geo.width, geo.height, geo.nchw, geo.channels)
        return Result.Image(bmp, "Super-resolution · ${geo.width}×${geo.height}", ms)
    }

    private fun rawSummary(shape: IntArray, out: FloatArray): String {
        var lo = Float.POSITIVE_INFINITY; var hi = Float.NEGATIVE_INFINITY
        for (v in out) { if (v < lo) lo = v; if (v > hi) hi = v }
        return "Ran OK. Output ${shape.toList()} · ${out.size} values · range " +
            "[%.3f, %.3f].\nRendering for this task isn't wired up yet — see the Benchmark tab for timing.".format(lo, hi)
    }

    /** Channel/H/W for a 4-D image output. */
    private fun parse4d(shape: IntArray): InputGeometry? = InputGeometry.parse(shape)

    /** Spatial H,W for a depth-like output of any rank. */
    private fun spatialHW(shape: IntArray, numel: Int): Pair<Int, Int> {
        val big = shape.filter { it > 1 }
        return when {
            big.size >= 2 -> big[big.size - 2] to big[big.size - 1]
            else -> {
                val s = Math.sqrt(numel.toDouble()).toInt()
                s to s
            }
        }
    }

    override fun close() {
        model.close()
        runtime.close()
    }

    companion object {
        private val INT_INPUT_TYPES = setOf(
            io.melan.npulab.inference.TensorDtype.UINT8,
            io.melan.npulab.inference.TensorDtype.INT8,
            io.melan.npulab.inference.TensorDtype.UINT16,
            io.melan.npulab.inference.TensorDtype.INT16,
            io.melan.npulab.inference.TensorDtype.UINT32,
            io.melan.npulab.inference.TensorDtype.INT32,
        )

        /** Categories the Vision screen can render. */
        val SUPPORTED = setOf(
            ModelCategory.CLASSIFICATION, ModelCategory.DEPTH,
            ModelCategory.SEGMENTATION, ModelCategory.SUPER_RESOLUTION,
        )

        private fun loadLabels(ctx: Context, asset: ModelAsset): List<String> {
            // Prefer the model's own labels.txt (extracted next to the .dlc as
            // "<dir>/<id>.labels.txt"); fall back to the bundled ImageNet list.
            val store = ModelStore(ctx)
            asset.primaryModelFile
                ?.removeSuffix(".dlc")?.removeSuffix(".bin")
                ?.let { store.pathOf("$it.labels.txt") }
                ?.takeIf { it.isFile }
                ?.let { f -> runCatching { return f.readLines().filter { it.isNotBlank() } } }
            return runCatching {
                ctx.assets.open("vision/imagenet_classes.txt").bufferedReader()
                    .readLines().filter { it.isNotBlank() }
            }.getOrDefault(emptyList())
        }
    }
}
