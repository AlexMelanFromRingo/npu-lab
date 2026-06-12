package io.melan.npulab.inference

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import io.melan.npulab.scheduler.DpmSolverMultistep
import io.melan.npulab.scheduler.EulerDiscreteSolver
import io.melan.npulab.tokenizer.ClipTokenizer
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val TAG = "SDPipeline"
private const val LATENT_CHANNELS = 4
private const val LATENT_HW = 64
private const val IMAGE_HW = 512
private const val MAX_TOKENS = 77
private const val EMBED_DIM = 768
private const val CFG_SCALE_DEFAULT = 7.5f

/**
 * Expected tensor schema of the Qualcomm AI Hub SD 1.5 context binaries.
 * Single source of truth shared by the pipeline and the unit tests that pin
 * these names against `qnn-context-binary-utility` dumps of the real .bin
 * files (see app/src/test/.../BinaryMetadataContractTest).
 */
object SdSchema {
    const val TEXT_ENCODER_GRAPH = "stable_diffusion_v1_5_text_encoder"
    const val UNET_GRAPH = "stable_diffusion_v1_5_unet"
    const val VAE_GRAPH = "stable_diffusion_v1_5_vae"

    const val IN_TOKENS = "tokens"
    const val OUT_TEXT_EMBEDDING = "text_embedding"
    const val IN_TIMESTEP = "timestep"
    const val IN_LATENT = "latent"
    const val IN_TEXT_EMB = "text_emb"
    const val OUT_LATENT = "output_latent"
    const val OUT_IMAGE = "image"

    /** Max timestep our scheduler may emit; must fit every quantized
     *  `timestep` input grid (leading spacing → 951 for 20 steps). */
    const val MAX_SCHEDULER_TIMESTEP = 951f
}

/**
 * Stable Diffusion 1.5 pipeline against Qualcomm's public S3 release
 * (`stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_8_elite_gen5.zip`).
 *
 * Schema (verified with qnn-context-binary-utility on the actual binaries —
 * socModel=87 / dspArch=81, i.e. SM8850 + Hexagon V81):
 *   text_encoder:  in  tokens (i32, [1,77])
 *                  out text_embedding (ufixed16, [1,77,768])
 *   unet:          in  timestep (ufixed16, [1,1])                 ← raw t, NOT an embedding
 *                  in  latent (ufixed16, NHWC [1,64,64,4])        ← NHWC, not NCHW
 *                  in  text_emb (ufixed16, [1,77,768])
 *                  out output_latent (ufixed16, NHWC)
 *   vae_decoder:   in  latent (ufixed16, NHWC) — RAW latents! The graph divides
 *                  by the SD scaling factor (0.18215) internally (AI Hub bakes
 *                  `z / config.scaling_factor` into the exported model), so the
 *                  app must NOT pre-divide.
 *                  out image (ufixed16, NHWC [1,512,512,3], range 0..1)
 *
 * All ufixed16 tensors are quantized as `real = scale * (q + offset)`. The
 * scale/offset come from the binary itself (TensorSpec.scale/offset).
 */
class StableDiffusionPipeline(
    private val runtime: QnnRuntime,
    modelStore: ModelStore,
    asset: ModelAsset,
) : Closeable {

    private val tokenizer = ClipTokenizer(
        vocabFile = modelStore.pathOf(asset.expectedFiles.first { it.endsWith("vocab.json") }),
        mergesFile = modelStore.pathOf(asset.expectedFiles.first { it.endsWith("merges.txt") }),
    )
    private val textEncoder = runtime.loadModel(
        modelStore.pathOf(asset.expectedFiles.first { it.endsWith("text_encoder.bin") }).absolutePath
    )
    private val unet = runtime.loadModel(
        modelStore.pathOf(asset.expectedFiles.first { it.endsWith("unet.bin") }).absolutePath
    )
    private val vaeDecoder = runtime.loadModel(
        modelStore.pathOf(asset.expectedFiles.first { it.endsWith("vae_decoder.bin") }).absolutePath
    )

    // Resolve UNet inputs BY NAME — robust against input-order changes in
    // future AI Hub releases. Missing names still fail loudly at load time.
    private val unetTimestepIdx = unetInputIndex(SdSchema.IN_TIMESTEP)
    private val unetLatentIdx = unetInputIndex(SdSchema.IN_LATENT)
    private val unetTextEmbIdx = unetInputIndex(SdSchema.IN_TEXT_EMB)

    private fun unetInputIndex(name: String): Int {
        val idx = unet.inputs.indexOfFirst { it.name == name }
        check(idx >= 0) {
            "UNet schema changed — input '$name' not found among " +
                unet.inputs.map { it.name }
        }
        return idx
    }

    init {
        // The scheduler's timestep range must fit the quantized `timestep`
        // input grid; otherwise the first (structurally most important) steps
        // silently clamp. With leading spacing max t=951 and the AI Hub binary
        // grid topping out at scale*65535 ≈ 968 this holds; fail loudly if a
        // future binary ships a narrower grid.
        val tSpec = unet.inputs[unetTimestepIdx]
        if (tSpec.isQuantized) {
            val qMax = tSpec.scale * (0xFFFF + tSpec.offset)
            check(qMax >= SdSchema.MAX_SCHEDULER_TIMESTEP) {
                "UNet timestep quant range tops out at $qMax — below scheduler max " +
                    "${SdSchema.MAX_SCHEDULER_TIMESTEP}; image quality would degrade"
            }
        }
    }

    /** Sampler choice for [generate]. */
    enum class Sampler {
        /** diffusers EulerDiscrete — what the AI Hub binaries were calibrated
         *  against (Qualcomm's reference app). Safest default for quality. */
        EULER,

        /** DPM-Solver++ 2M — usually sharper at 20 steps on float models;
         *  formulas verified against diffusers. */
        DPMPP_2M,
    }

    /** Uniform facade over the two schedulers used by [generate]. */
    private interface Sched {
        val steps: Int
        fun timestep(i: Int): Float
        fun initialLatents(numel: Int, seed: Long): FloatArray
        fun scaleModelInput(latents: FloatArray, i: Int): FloatArray
        fun step(i: Int, latents: FloatArray, eps: FloatArray): FloatArray
    }

    private fun makeSched(sampler: Sampler, numSteps: Int): Sched = when (sampler) {
        Sampler.EULER -> object : Sched {
            private val s = EulerDiscreteSolver(numInferenceSteps = numSteps)
            override val steps = numSteps
            override fun timestep(i: Int) = s.timesteps[i]
            override fun initialLatents(numel: Int, seed: Long) = s.initialLatents(numel, seed)
            override fun scaleModelInput(latents: FloatArray, i: Int) = s.scaleModelInput(latents, i)
            override fun step(i: Int, latents: FloatArray, eps: FloatArray) = s.step(i, latents, eps)
        }
        Sampler.DPMPP_2M -> object : Sched {
            private val s = DpmSolverMultistep(numInferenceSteps = numSteps)
            override val steps = numSteps
            override fun timestep(i: Int) = s.timesteps[i].toFloat()
            override fun initialLatents(numel: Int, seed: Long) =
                s.initialLatents(intArrayOf(numel), seed)
            override fun scaleModelInput(latents: FloatArray, i: Int) = latents // VP: identity
            override fun step(i: Int, latents: FloatArray, eps: FloatArray) = s.step(i, latents, eps)
        }
    }

    data class StepTiming(val step: Int, val totalSteps: Int, val unetUs: Long)

    data class Result(
        val bitmap: Bitmap,
        val textEncoderUs: Long,
        val unetTotalUs: Long,
        val vaeUs: Long,
        val wallUs: Long,
        val perStep: List<StepTiming>,
    )

    fun generate(
        prompt: String,
        negativePrompt: String = "",
        numSteps: Int = 20,
        cfgScale: Float = CFG_SCALE_DEFAULT,
        seed: Long = -1,
        sampler: Sampler = Sampler.EULER,
        onProgress: ((StepTiming) -> Unit)? = null,
    ): Result {
        val resolvedSeed = if (seed < 0) System.nanoTime() else seed
        val wallStart = System.nanoTime()

        val condEmbed = runTextEncoder(prompt)
        val uncondEmbed = runTextEncoder(negativePrompt)
        val textEncoderUs = (System.nanoTime() - wallStart) / 1000L

        val sched = makeSched(sampler, numSteps)
        // Latents are NHWC in the binary, but the scheduler is shape-agnostic — it
        // operates on a flat FloatArray of size 4*64*64.
        var latents = sched.initialLatents(
            LATENT_CHANNELS * LATENT_HW * LATENT_HW,
            resolvedSeed,
        )

        val perStep = ArrayList<StepTiming>(numSteps)
        var unetTotal = 0L
        for (s in 0 until numSteps) {
            val tStart = System.nanoTime()
            val t = sched.timestep(s)
            // What the UNet consumes (Euler: x/√(σ²+1) = VP x_t; DPM++: as-is).
            val unetInput = sched.scaleModelInput(latents, s)
            val condEps = runUnet(unetInput, t, condEmbed)
            val uncondEps = runUnet(unetInput, t, uncondEmbed)
            val guided = FloatArray(condEps.size) { i ->
                uncondEps[i] + cfgScale * (condEps[i] - uncondEps[i])
            }
            latents = sched.step(s, latents, guided)
            val us = (System.nanoTime() - tStart) / 1000L
            unetTotal += us
            val st = StepTiming(s, numSteps, us)
            perStep.add(st)
            onProgress?.invoke(st)
            Log.d(TAG, "step ${s + 1}/$numSteps  unet=${us / 1000}ms")
        }

        val vaeStart = System.nanoTime()
        val pixels = runVae(latents)
        val vaeUs = (System.nanoTime() - vaeStart) / 1000L

        return Result(
            bitmap = pixelsToBitmap(pixels),
            textEncoderUs = textEncoderUs,
            unetTotalUs = unetTotal,
            vaeUs = vaeUs,
            wallUs = (System.nanoTime() - wallStart) / 1000L,
            perStep = perStep,
        )
    }

    /* ------------------------- Stage runners ------------------------- */

    private fun runTextEncoder(text: String): FloatArray {
        val ids = tokenizer.encode(text, MAX_TOKENS)
        val input = ByteBuffer.allocateDirect(MAX_TOKENS * 4).order(ByteOrder.nativeOrder())
        for (id in ids) input.putInt(id)
        input.rewind()

        val outputs = textEncoder.allocOutputs()
        textEncoder.execute(arrayOf(input), outputs)

        // Dequantize text_embedding (ufixed16) into FP32.
        val outSpec = textEncoder.outputs[0]
        return dequantizeUfixed16(outputs[0], outSpec.scale, outSpec.offset, outSpec.numel)
    }

    /**
     * UNet inputs are resolved by name (timestep, latent, text_emb — the AI Hub
     * order, but we don't depend on it). Latent is NHWC ([1, 64, 64, 4]) on
     * disk; our scheduler keeps it in CHW order (channels-first, flat 4*64*64).
     * We transpose during quant-encode and after dequant.
     */
    private fun runUnet(latentsChw: FloatArray, timestep: Float, textEmbed: FloatArray): FloatArray {
        val tSpec = unet.inputs[unetTimestepIdx]
        val lSpec = unet.inputs[unetLatentIdx]
        val eSpec = unet.inputs[unetTextEmbIdx]
        val oSpec = unet.outputs[0]

        val tBuf = ByteBuffer.allocateDirect(2).order(ByteOrder.nativeOrder())
        tBuf.putShort(quantizeOneUfixed16(timestep, tSpec.scale, tSpec.offset))
        tBuf.rewind()

        val latentsNhwc = chwToNhwc(latentsChw, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        val lBuf = quantizeUfixed16(latentsNhwc, lSpec.scale, lSpec.offset)
        val eBuf = quantizeUfixed16(textEmbed, eSpec.scale, eSpec.offset)

        val inputBufs = Array(unet.inputs.size) { idx ->
            when (idx) {
                unetTimestepIdx -> tBuf
                unetLatentIdx -> lBuf
                unetTextEmbIdx -> eBuf
                else -> error("unexpected UNet input #$idx: ${unet.inputs[idx].name}")
            }
        }

        val outputs = unet.allocOutputs()
        unet.execute(inputBufs, outputs)

        val outNhwc = dequantizeUfixed16(outputs[0], oSpec.scale, oSpec.offset, oSpec.numel)
        return nhwcToChw(outNhwc, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
    }

    private fun runVae(latentsChw: FloatArray): FloatArray {
        val inSpec = vaeDecoder.inputs[0]
        val outSpec = vaeDecoder.outputs[0]

        // Feed RAW latents: the AI Hub VAE graph contains the division by the
        // SD scaling factor (0.18215) itself. Dividing here as well would wash
        // the image out (everything pulled towards mid-gray). The quant range
        // of the input (scale*(0..65535)+offset ≈ ±11) is sized for raw
        // latents — pre-divided ones (±60) would clip hard.
        val latentsNhwc = chwToNhwc(latentsChw, LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        val input = quantizeUfixed16(latentsNhwc, inSpec.scale, inSpec.offset)

        val outputs = vaeDecoder.allocOutputs()
        vaeDecoder.execute(arrayOf(input), outputs)
        return dequantizeUfixed16(outputs[0], outSpec.scale, outSpec.offset, outSpec.numel)
    }

    private fun pixelsToBitmap(nhwc: FloatArray): Bitmap {
        // VAE image output is NHWC [1, 512, 512, 3] in the 0..1 range
        // (vae quantize: scale ≈ 1.5e-5, offset=0 → max q=65535 → ≈1.0).
        val bmp = Bitmap.createBitmap(IMAGE_HW, IMAGE_HW, Bitmap.Config.ARGB_8888)
        val intPixels = IntArray(IMAGE_HW * IMAGE_HW)
        var pi = 0
        for (y in 0 until IMAGE_HW) {
            for (x in 0 until IMAGE_HW) {
                val base = (y * IMAGE_HW + x) * 3
                val r = clamp01(nhwc[base])
                val g = clamp01(nhwc[base + 1])
                val b = clamp01(nhwc[base + 2])
                intPixels[pi++] = Color.argb(
                    255,
                    (r * 255f).roundToInt(),
                    (g * 255f).roundToInt(),
                    (b * 255f).roundToInt(),
                )
            }
        }
        bmp.setPixels(intPixels, 0, IMAGE_HW, 0, 0, IMAGE_HW, IMAGE_HW)
        return bmp
    }

    private fun clamp01(v: Float): Float = max(0f, min(1f, v))

    override fun close() {
        textEncoder.close()
        unet.close()
        vaeDecoder.close()
    }
}

/* ----------------------- Quantize helpers ----------------------- */

/** Encodes float array as ufixed16, returns direct ByteBuffer (size = src.size * 2). */
internal fun quantizeUfixed16(src: FloatArray, scale: Float, offset: Int): ByteBuffer {
    val buf = ByteBuffer.allocateDirect(src.size * 2).order(ByteOrder.nativeOrder())
    val invScale = 1f / scale
    for (v in src) {
        val q = (v * invScale).roundToInt() - offset
        buf.putShort(q.coerceIn(0, 0xFFFF).toShort())
    }
    buf.rewind()
    return buf
}

/** Encodes a single FP32 scalar as ufixed16. */
internal fun quantizeOneUfixed16(v: Float, scale: Float, offset: Int): Short {
    val q = (v / scale).roundToInt() - offset
    return q.coerceIn(0, 0xFFFF).toShort()
}

internal fun dequantizeUfixed16(buf: ByteBuffer, scale: Float, offset: Int, count: Int): FloatArray {
    buf.rewind()
    val out = FloatArray(count)
    for (i in 0 until count) {
        val q = buf.short.toInt() and 0xFFFF
        out[i] = scale * (q + offset)
    }
    return out
}

/* ----------------------- Layout helpers ----------------------- */

/** Transpose [C,H,W] (flat) → [H,W,C] (flat). */
internal fun chwToNhwc(chw: FloatArray, c: Int, h: Int, w: Int): FloatArray {
    val out = FloatArray(chw.size)
    val plane = h * w
    for (ch in 0 until c) {
        for (y in 0 until h) {
            for (x in 0 until w) {
                out[(y * w + x) * c + ch] = chw[ch * plane + y * w + x]
            }
        }
    }
    return out
}

/** Transpose [H,W,C] (flat) → [C,H,W] (flat). */
internal fun nhwcToChw(nhwc: FloatArray, c: Int, h: Int, w: Int): FloatArray {
    val out = FloatArray(nhwc.size)
    val plane = h * w
    for (y in 0 until h) {
        for (x in 0 until w) {
            for (ch in 0 until c) {
                out[ch * plane + y * w + x] = nhwc[(y * w + x) * c + ch]
            }
        }
    }
    return out
}
