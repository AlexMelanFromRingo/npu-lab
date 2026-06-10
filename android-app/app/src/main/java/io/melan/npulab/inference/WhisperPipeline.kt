package io.melan.npulab.inference

import android.content.Context
import android.util.Log
import io.melan.npulab.audio.Fp16
import io.melan.npulab.audio.LogMelFrontend
import io.melan.npulab.tokenizer.WhisperTokenizer
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "WhisperPipeline"

/**
 * Whisper speech-to-text on the NPU (any [WhisperVariant]), mirroring
 * Qualcomm's reference `HfWhisperApp` loop:
 *
 *   mel [1,nMels,3000] f16 → encoder → per-layer cross K/V caches (constant
 *   per chunk); decoder: autoregressive, ≤maskLen−1 steps, greedy argmax:
 *     - prompt is FORCED to [SOT, lang, transcribe, notimestamps] — language
 *       either set by the caller or detected at step 0 via argmax restricted
 *       to the language-token range (official detect_language method);
 *     - attention_mask [1,1,1,maskLen] starts at −100.0 everywhere, one slot
 *       (index maskLen−n−1) is zeroed per step — the valid window grows from
 *       the right edge;
 *     - self K/V caches ride OUT of step n straight INTO step n+1
 *       (ping-pong buffers, no copies);
 *     - stop on <|endoftext|>.
 *
 * Geometry (layer count, cache shapes, mel bins, window) is read from the
 * context binaries; all decoder I/O slots are resolved BY NAME — so model
 * re-exports and bigger variants (Small, Large-v3 Turbo) work unchanged.
 */
class WhisperPipeline(
    private val runtime: QnnRuntime,
    private val store: ModelStore,
    context: Context,
    private val variant: WhisperVariant = WhisperVariant.BASE,
) : Closeable {

    private val tokenizer = WhisperTokenizer(
        vocabFile = store.pathOf(variant.vocabPath),
    )
    private val encoder = runtime.loadModel(store.pathOf(variant.encoderPath).absolutePath)
    private val decoder = runtime.loadModel(store.pathOf(variant.decoderPath).absolutePath)

    // Mel-bin count comes from the encoder itself: input_features [1, nMels, 3000].
    // 80 for tiny…medium, 128 for the large-v3 family.
    private val nMels = encoder.inputs[0].shape[1]
    private val frontend = LogMelFrontend(
        context.assets.open(LogMelFrontend.filterAssetFor(nMels)),
        nMels,
    )

    private val decInName = decoder.inputs.map { it.name }
    private val decOutName = decoder.outputs.map { it.name }

    private fun inIdx(name: String) = decInName.indexOf(name).also {
        check(it >= 0) { "decoder input '$name' not found among $decInName" }
    }
    private fun outIdx(name: String) = decOutName.indexOf(name).also {
        check(it >= 0) { "decoder output '$name' not found among $decOutName" }
    }

    private val idxInputIds = inIdx("input_ids")
    private val idxPositionIds = inIdx("position_ids")
    private val idxAttnMask = inIdx("attention_mask")
    private val idxLogits = outIdx("logits")
    private val numLayers = decoder.inputs.count { it.name.startsWith("k_cache_self_") }

    // Decode window: attention_mask is [1, 1, 1, maskLen]; self caches hold
    // maskLen-1 positions. 200 in current AI Hub exports, but read it from the
    // binary so a re-export with a longer window keeps working.
    private val maskLen = decoder.inputs[idxAttnMask].shape.last()

    data class Result(
        val text: String,
        val tokens: Int,
        val melMs: Long,
        val encoderMs: Long,
        val decodeMs: Long,
        val tokensPerSecond: Float,
        /** Detected (or forced) language code of the LAST chunk, e.g. "ru". */
        val language: String?,
    )

    /**
     * Transcribe audio of any length: split into 30 s windows (the hard
     * Whisper context size), run each through the model, concatenate.
     *
     * @param language ISO code ("ru", "en", …) to FORCE, or null for
     *        auto-detect per chunk (official Whisper detect_language method:
     *        argmax over language tokens only at the first decode step).
     */
    fun transcribeLong(
        pcm: FloatArray,
        language: String? = null,
        onProgress: ((chunk: Int, totalChunks: Int, partial: String) -> Unit)? = null,
    ): Result {
        val chunks = splitIntoChunks(pcm, LogMelFrontend.N_SAMPLES)
        val texts = StringBuilder()
        var tokens = 0; var melMs = 0L; var encMs = 0L; var decMs = 0L
        var steps = 0; var lang: String? = null
        for ((ci, chunk) in chunks.withIndex()) {
            val r = transcribe(chunk, language) { partial ->
                onProgress?.invoke(ci + 1, chunks.size, texts.toString() + partial)
            }
            if (r.text.isNotBlank()) {
                if (texts.isNotEmpty()) texts.append(' ')
                texts.append(r.text)
            }
            tokens += r.tokens; melMs += r.melMs; encMs += r.encoderMs
            decMs += r.decodeMs; lang = r.language ?: lang
            steps += (r.tokensPerSecond * r.decodeMs / 1000f).toInt()
            onProgress?.invoke(ci + 1, chunks.size, texts.toString())
        }
        return Result(
            text = texts.toString(),
            tokens = tokens,
            melMs = melMs,
            encoderMs = encMs,
            decodeMs = decMs,
            tokensPerSecond = if (decMs > 0) steps * 1000f / decMs else 0f,
            language = lang,
        )
    }

    /**
     * @param pcm 16 kHz mono float PCM, up to 30 s.
     * @param language ISO code to force, or null for auto-detect.
     * @param onPartial called with the partial transcription as tokens arrive.
     */
    fun transcribe(
        pcm: FloatArray,
        language: String? = null,
        onPartial: ((String) -> Unit)? = null,
    ): Result {
        val tMel0 = System.nanoTime()
        val mel = frontend.compute(pcm)
        val melMs = (System.nanoTime() - tMel0) / 1_000_000

        // ── Encoder ────────────────────────────────────────────────
        val tEnc0 = System.nanoTime()
        val encIn = fp16Buffer(mel)
        val encOut = encoder.allocOutputs()
        encoder.execute(arrayOf(encIn), encOut)
        val encoderMs = (System.nanoTime() - tEnc0) / 1_000_000

        // Cross K/V cache buffers are constant for the whole decode — wire the
        // encoder outputs into the matching decoder inputs by tensor name.
        val crossByName = encoder.outputs.indices.associate { i ->
            encoder.outputs[i].name to encOut[i]
        }

        // ── Decoder loop ───────────────────────────────────────────
        val tDec0 = System.nanoTime()
        val maskBuf = ByteBuffer.allocateDirect(maskLen * 2).order(ByteOrder.nativeOrder())
        val maskNeg = Fp16.floatToHalfBits(MASK_NEG)
        repeat(maskLen) { maskBuf.putShort(maskNeg) }
        val idsBuf = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        val posBuf = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())

        // Two interchangeable self-cache sets: outputs of step n feed step n+1.
        var selfIn = decoder.outputs.indices.map { i ->
            ByteBuffer.allocateDirect(decoder.outputs[i].byteSize).order(ByteOrder.nativeOrder())
        }.toMutableList()  // index space of OUTPUTS; logits slot unused as input
        var outSet = decoder.allocOutputs()

        val selfOutToIn = HashMap<Int, Int>() // decoder output idx -> decoder input idx
        for (l in 0 until numLayers) {
            selfOutToIn[outIdx("k_cache_self_${l}_out")] = inIdx("k_cache_self_${l}_in")
            selfOutToIn[outIdx("v_cache_self_${l}_out")] = inIdx("v_cache_self_${l}_in")
        }

        // Decoder prompt. Greedy decoding from a bare SOT lets the model pick
        // the language AND the task itself — on short/noisy audio it loves to
        // fall into <|en|> or even <|translate|> (Russian speech came out as
        // English). So we do what openai/whisper does:
        //   [SOT, <|lang|>, <|transcribe|>, <|notimestamps|>]
        // with the language either forced by the caller or detected at step 0
        // via argmax restricted to the language-token range.
        val ids = ArrayList<Int>(64).apply { add(WhisperTokenizer.SOT) }
        val forcedLang = language?.let { WhisperTokenizer.LANGUAGE_IDS[it] }
        if (forcedLang != null) {
            ids.add(forcedLang)
            ids.add(variant.transcribeId)
            ids.add(variant.noTimestampsId)
        }
        var detectedLangId: Int? = forcedLang
        val logitsSpec = decoder.outputs[idxLogits]
        var decodeSteps = 0

        for (n in 0 until maskLen - 1) {
            // current token + position
            idsBuf.clear(); idsBuf.putInt(ids[n]); idsBuf.rewind()
            posBuf.clear(); posBuf.putInt(n); posBuf.rewind()
            // unmask one more slot from the right edge
            maskBuf.putShort((maskLen - n - 1) * 2, 0)
            maskBuf.rewind()

            val inputs = arrayOfNulls<ByteBuffer>(decoder.inputs.size)
            inputs[idxInputIds] = idsBuf
            inputs[idxPositionIds] = posBuf
            inputs[idxAttnMask] = maskBuf
            for ((outI, inI) in selfOutToIn) inputs[inI] = selfIn[outI]
            for ((name, buf) in crossByName) {
                val i = decInName.indexOf(name)
                if (i >= 0) inputs[i] = buf
            }
            check(inputs.none { it == null }) {
                "decoder input not wired: " +
                    decInName.filterIndexed { i, _ -> inputs[i] == null }
            }
            @Suppress("UNCHECKED_CAST")
            decoder.execute(inputs as Array<ByteBuffer>, outSet)
            decodeSteps++

            val logitsBuf = outSet[idxLogits]

            // ping-pong: this step's outputs become next step's self-cache in
            val tmp = selfIn
            selfIn = outSet.toMutableList()
            outSet = tmp.toTypedArray()

            if (n == 0 && forcedLang == null) {
                // Language detection step (official method): consider ONLY
                // language tokens, then force the transcribe prompt.
                val langTok = argmaxFp16(
                    logitsBuf, logitsSpec.numel,
                    from = variant.langFirst,
                    until = variant.langLast + 1,
                )
                detectedLangId = langTok
                ids.add(langTok)
                ids.add(variant.transcribeId)
                ids.add(variant.noTimestampsId)
                continue
            }
            if (n < ids.size - 1) continue  // teacher-forcing the prompt tokens

            val tok = argmaxFp16(logitsBuf, logitsSpec.numel)
            ids.add(tok)
            if (tok == WhisperTokenizer.EOT || n == maskLen - 2) break
            onPartial?.invoke(tokenizer.decode(ids))
        }
        val decodeMs = (System.nanoTime() - tDec0) / 1_000_000

        val text = tokenizer.decode(ids).trim()
        val langCode = detectedLangId?.let { id ->
            WhisperTokenizer.LANGUAGE_IDS.entries.firstOrNull { it.value == id }?.key
                ?: "#${id - variant.langFirst}"
        }
        Log.i(TAG, "transcribed ${ids.size} tokens in ${decodeMs}ms " +
            "(enc ${encoderMs}ms, lang=$langCode)")
        return Result(
            text = text,
            tokens = ids.size,
            melMs = melMs,
            encoderMs = encoderMs,
            decodeMs = decodeMs,
            tokensPerSecond = if (decodeMs > 0) decodeSteps * 1000f / decodeMs else 0f,
            language = langCode,
        )
    }

    private fun fp16Buffer(src: FloatArray): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(src.size * 2).order(ByteOrder.nativeOrder())
        for (v in src) buf.putShort(Fp16.floatToHalfBits(v))
        buf.rewind()
        return buf
    }

    private fun argmaxFp16(buf: ByteBuffer, count: Int, from: Int = 0, until: Int = -1): Int {
        buf.rewind()
        val hi = if (until in 1..count) until else count
        var best = -Float.MAX_VALUE
        var bestIdx = from
        for (i in from until hi) {
            val v = Fp16.halfBitsToFloat(buf.getShort(i * 2))
            if (v > best) { best = v; bestIdx = i }
        }
        return bestIdx
    }

    override fun close() {
        encoder.close()
        decoder.close()
    }

    companion object {
        const val MASK_NEG = -100.0f
    }
}

/** Split PCM into ≤chunkSize windows; always returns at least one chunk. */
internal fun splitIntoChunks(pcm: FloatArray, chunkSize: Int): List<FloatArray> {
    if (pcm.size <= chunkSize) return listOf(pcm)
    val out = ArrayList<FloatArray>((pcm.size + chunkSize - 1) / chunkSize)
    var pos = 0
    while (pos < pcm.size) {
        val end = minOf(pos + chunkSize, pcm.size)
        out.add(pcm.copyOfRange(pos, end))
        pos = end
    }
    return out
}
