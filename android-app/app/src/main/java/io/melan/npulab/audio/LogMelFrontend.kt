package io.melan.npulab.audio

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * Whisper log-mel frontend — a 1:1 port of HF `WhisperFeatureExtractor`
 * (which itself mirrors openai/whisper):
 *
 *   1. pad/truncate PCM (16 kHz mono float) to exactly 480 000 samples (30 s);
 *   2. STFT: hann(400, periodic), n_fft=400, hop=160, center=true with
 *      REFLECT padding → 3001 frames, the last one dropped → 3000;
 *   3. power spectrum (|X|²) → 80-bin mel via the precomputed filter bank
 *      (201×80, slaney scale + slaney norm — shipped as an asset, generated
 *      by transformers so the coefficients are bit-identical);
 *   4. log10(max(x, 1e-10)); clamp to global max − 8; (x + 4) / 4.
 *
 * Output layout matches the encoder input `input_features` [1, 80, 3000]
 * (mel-major). Verified against a golden fixture from the real HF extractor
 * in LogMelFrontendTest.
 *
 * The 400-point DFT is computed as a dense matmul with precomputed cos/sin
 * tables (201 × 400 each) — no FFT library, and 400 isn't a power of two
 * anyway. ~0.5 GFLOP per 30 s chunk, parallelized across frames.
 */
class LogMelFrontend(
    melFiltersStream: InputStream,
    /** 80 for Whisper tiny…medium, 128 for the large-v3 family. */
    val nMels: Int = 80,
) {

    private val melFilters: FloatArray // [N_FREQS * nMels], freq-major

    init {
        val bytes = melFiltersStream.use { it.readBytes() }
        require(bytes.size == N_FREQS * nMels * 4) {
            "mel filterbank: expected ${N_FREQS * nMels * 4} bytes for $nMels mels, got ${bytes.size}"
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        melFilters = FloatArray(N_FREQS * nMels) { bb.getFloat(it * 4) }
    }

    // Periodic hann window: 0.5 − 0.5·cos(2πn / N)
    private val window = FloatArray(N_FFT) { n ->
        (0.5 - 0.5 * cos(2.0 * Math.PI * n / N_FFT)).toFloat()
    }

    // DFT tables for the one-sided spectrum: [bin][sample]
    private val cosTable = FloatArray(N_FREQS * N_FFT)
    private val sinTable = FloatArray(N_FREQS * N_FFT)

    init {
        for (k in 0 until N_FREQS) {
            for (n in 0 until N_FFT) {
                val ang = -2.0 * Math.PI * k * n / N_FFT
                cosTable[k * N_FFT + n] = cos(ang).toFloat()
                sinTable[k * N_FFT + n] = sin(ang).toFloat()
            }
        }
    }

    /**
     * @param pcm mono float PCM at 16 kHz, any length ≤ 30 s (longer input is
     *            truncated). Values in [-1, 1].
     * @return log-mel features, mel-major [nMels * 3000] — ready to be fed as
     *         `input_features` after fp16 packing.
     */
    fun compute(pcm: FloatArray): FloatArray = runBlocking { computeSuspend(pcm) }

    suspend fun computeSuspend(pcm: FloatArray): FloatArray {
        // 1. Pad / truncate to exactly 30 s.
        val samples = FloatArray(N_SAMPLES)
        val n = minOf(pcm.size, N_SAMPLES)
        System.arraycopy(pcm, 0, samples, 0, n)

        // 2. Reflect-pad by half a window on both sides (center=true).
        val half = N_FFT / 2
        val padded = FloatArray(N_SAMPLES + N_FFT)
        System.arraycopy(samples, 0, padded, half, N_SAMPLES)
        for (i in 0 until half) {
            padded[half - 1 - i] = samples[i + 1]                  // left reflect
            padded[half + N_SAMPLES + i] = samples[N_SAMPLES - 2 - i] // right reflect
        }

        // 3. Windowed DFT power → mel, parallel over frame blocks.
        val mel = FloatArray(nMels * N_FRAMES)
        coroutineScope {
            val block = 256
            var start = 0
            while (start < N_FRAMES) {
                val from = start
                val to = minOf(start + block, N_FRAMES)
                launch(Dispatchers.Default) { melFrames(padded, mel, from, to) }
                start = to
            }
        }

        // 4. log10 / dynamic-range / scale — matches HF order exactly.
        var maxVal = -Float.MAX_VALUE
        for (i in mel.indices) {
            val v = log10(max(mel[i], 1e-10f))
            mel[i] = v
            if (v > maxVal) maxVal = v
        }
        val floor = maxVal - 8f
        for (i in mel.indices) {
            mel[i] = (max(mel[i], floor) + 4f) / 4f
        }
        return mel
    }

    private fun melFrames(padded: FloatArray, melOut: FloatArray, from: Int, to: Int) {
        val frame = FloatArray(N_FFT)
        val power = FloatArray(N_FREQS)
        for (f in from until to) {
            val base = f * HOP
            for (i in 0 until N_FFT) frame[i] = padded[base + i] * window[i]
            for (k in 0 until N_FREQS) {
                var re = 0f
                var im = 0f
                val off = k * N_FFT
                for (i in 0 until N_FFT) {
                    val s = frame[i]
                    re += s * cosTable[off + i]
                    im += s * sinTable[off + i]
                }
                power[k] = re * re + im * im
            }
            // mel[m, f] = Σ_k power[k] · fb[k, m]
            for (m in 0 until nMels) {
                var acc = 0f
                for (k in 0 until N_FREQS) {
                    acc += power[k] * melFilters[k * nMels + m]
                }
                melOut[m * N_FRAMES + f] = acc
            }
        }
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val N_FFT = 400
        const val HOP = 160
        const val N_FREQS = N_FFT / 2 + 1   // 201
        const val N_SAMPLES = 30 * SAMPLE_RATE  // 480 000
        const val N_FRAMES = 3000

        /** Asset path of the filterbank for the given mel-bin count. */
        fun filterAssetFor(nMels: Int): String = when (nMels) {
            80 -> "whisper/mel_filters.bin"
            128 -> "whisper/mel_filters_128.bin"
            else -> error("no mel filterbank asset for $nMels bins")
        }
    }
}
