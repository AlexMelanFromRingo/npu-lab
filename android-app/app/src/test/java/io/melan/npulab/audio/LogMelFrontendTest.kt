package io.melan.npulab.audio

import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Golden test against Hugging Face `WhisperFeatureExtractor`: the fixtures in
 * src/test/resources/whisper/ hold a deterministic 3-second signal and the
 * exact log-mel features the HF extractor produced for it. The Kotlin port
 * must match to float tolerance — this is what the NPU encoder actually eats.
 */
class LogMelFrontendTest {

    private fun resourceFloats(path: String): FloatArray {
        val bytes = javaClass.getResourceAsStream(path)!!.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { bb.getFloat(it * 4) }
    }

    @Test
    fun `matches HF WhisperFeatureExtractor on the golden signal`() {
        val pcm = resourceFloats("/whisper/pcm_3s.f32")
        val expected = resourceFloats("/whisper/logmel_expected.f32")
        assertEquals(80 * 3000, expected.size)

        val frontend = LogMelFrontend(
            javaClass.getResourceAsStream("/assets-mirror/mel_filters.bin")
                ?: javaClass.getResourceAsStream("/whisper/mel_filters.bin")!!,
        )
        val actual = frontend.compute(pcm)
        assertEquals(expected.size, actual.size)

        var worst = 0f
        var worstIdx = -1
        for (i in expected.indices) {
            val d = abs(actual[i] - expected[i])
            if (d > worst) { worst = d; worstIdx = i }
        }
        assertTrue(
            worst < 2e-3f,
            "max |Δ|=$worst at $worstIdx (mel=${worstIdx / 3000}, frame=${worstIdx % 3000})",
        )
    }

    @Test
    fun `matches HF extractor in 128-mel mode (large-v3 family)`() {
        val pcm = resourceFloats("/whisper/pcm_3s.f32")
        val expected = resourceFloats("/whisper/logmel128_expected.f32")
        assertEquals(128 * 3000, expected.size)

        val frontend = LogMelFrontend(
            javaClass.getResourceAsStream("/whisper/mel_filters_128.bin")!!,
            nMels = 128,
        )
        val actual = frontend.compute(pcm)
        var worst = 0f
        for (i in expected.indices) {
            val d = abs(actual[i] - expected[i])
            if (d > worst) worst = d
        }
        assertTrue(worst < 2e-3f, "max |Δ|=$worst")
    }

    @Test
    fun `silence maps to the documented constant`() {
        val frontend = LogMelFrontend(
            javaClass.getResourceAsStream("/whisper/mel_filters.bin")!!,
        )
        val out = frontend.compute(FloatArray(16000))
        // All-zero audio → log10(1e-10) everywhere → max-8 clamp → (x+4)/4.
        // max = -10 → floor = -18 → value = (-10+4)/4 = -1.5.
        for (i in 0 until 100) {
            assertEquals(-1.5f, out[i], 1e-4f)
        }
    }
}
