package io.melan.npulab.inference

import io.melan.npulab.tokenizer.WhisperTokenizer
import org.junit.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class WhisperHelpersTest {

    @Test
    fun `chunking covers all samples without overlap`() {
        val pcm = FloatArray(75_000) { it.toFloat() }
        val chunks = splitIntoChunks(pcm, 30_000)
        assertEquals(3, chunks.size)
        assertEquals(30_000, chunks[0].size)
        assertEquals(30_000, chunks[1].size)
        assertEquals(15_000, chunks[2].size)
        // Concatenation reproduces the original exactly.
        val joined = FloatArray(pcm.size)
        var pos = 0
        for (c in chunks) { c.copyInto(joined, pos); pos += c.size }
        assertContentEquals(pcm, joined)
    }

    @Test
    fun `short audio is a single chunk`() {
        val pcm = FloatArray(10)
        val chunks = splitIntoChunks(pcm, 30_000)
        assertEquals(1, chunks.size)
        assertContentEquals(pcm, chunks[0])
    }

    @Test
    fun `language ids match HF whisper-base`() {
        // Verified with transformers: tok.convert_tokens_to_ids("<|xx|>").
        assertEquals(50259, WhisperTokenizer.LANGUAGE_IDS["en"])
        assertEquals(50263, WhisperTokenizer.LANGUAGE_IDS["ru"])
        assertEquals(50280, WhisperTokenizer.LANGUAGE_IDS["uk"])
        assertEquals(50261, WhisperTokenizer.LANGUAGE_IDS["de"])
        for ((code, id) in WhisperTokenizer.LANGUAGE_IDS) {
            assertTrue(
                id in WhisperTokenizer.LANG_FIRST..WhisperTokenizer.LANG_LAST,
                "$code=$id outside language token range",
            )
        }
        assertEquals(50359, WhisperTokenizer.TRANSCRIBE)
        assertEquals(50363, WhisperTokenizer.NO_TIMESTAMPS)
    }
}
