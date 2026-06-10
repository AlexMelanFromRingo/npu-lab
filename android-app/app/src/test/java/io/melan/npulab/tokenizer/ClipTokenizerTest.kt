package io.melan.npulab.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pins ClipTokenizer to ground truth produced by Hugging Face `CLIPTokenizer`
 * (the slow/reference implementation) over the same vocab.json + merges.txt.
 * Fixtures: src/test/resources/tokenizer/fixtures.json — includes punctuation,
 * Cyrillic, emoji, repeated-char and over-length prompts.
 */
class ClipTokenizerTest {

    companion object {
        private lateinit var tokenizer: ClipTokenizer

        private fun resourceToTemp(path: String, suffix: String): File {
            val stream = ClipTokenizerTest::class.java.getResourceAsStream(path)
                ?: error("missing test resource $path")
            val f = File.createTempFile("cliptok", suffix)
            f.deleteOnExit()
            stream.use { input -> f.outputStream().use { input.copyTo(it) } }
            return f
        }

        @JvmStatic
        @BeforeClass
        fun setUp() {
            tokenizer = ClipTokenizer(
                vocabFile = resourceToTemp("/tokenizer/vocab.json", ".json"),
                mergesFile = resourceToTemp("/tokenizer/merges.txt", ".txt"),
            )
        }
    }

    @Test
    fun `matches HF CLIPTokenizer on all fixture prompts`() {
        val text = ClipTokenizerTest::class.java.getResourceAsStream("/tokenizer/fixtures.json")!!
            .readBytes().decodeToString()
        val fixtures = Json.parseToJsonElement(text).jsonArray
        assertTrue(fixtures.size >= 10, "fixture file should cover many prompts")
        for (fx in fixtures) {
            val prompt = fx.jsonObject["prompt"]!!.jsonPrimitive.content
            val expected = fx.jsonObject["ids"]!!.jsonArray.map { it.jsonPrimitive.int }
            val actual = tokenizer.encode(prompt, maxLength = 77)
            assertContentEquals(
                expected.toIntArray(),
                actual,
                "prompt: ${prompt.take(60)}",
            )
        }
    }

    @Test
    fun `always returns exactly 77 ids with BOS first`() {
        for (p in listOf("", "hi", "a".repeat(1000), "слово", "!!!")) {
            val ids = tokenizer.encode(p)
            assertEquals(77, ids.size)
            assertEquals(49406, ids[0], "BOS")
            assertTrue(ids.contains(49407), "EOS somewhere")
        }
    }

    @Test
    fun `cyrillic does not collapse to unk`() {
        // Byte-level BPE: every byte is representable, so a Cyrillic prompt
        // must produce content tokens, not just BOS + EOS padding.
        val ids = tokenizer.encode("котик в космосе")
        val content = ids.drop(1).filter { it != 49407 }
        assertTrue(content.size >= 4, "expected real tokens, got $content")
    }

    @Test
    fun `punctuation is split off words`() {
        // "cat," must tokenize as cat</w> + ,</w> — same ids as "cat ,".
        val withComma = tokenizer.encode("a cat, sleeping")
        val spaced = tokenizer.encode("a cat , sleeping")
        assertContentEquals(withComma, spaced)
    }

    @Test
    fun `byte encoder is a 256-entry bijection`() {
        val map = ClipTokenizer.BYTE_ENCODER
        assertEquals(256, map.size)
        assertEquals(256, map.toSet().size, "must be injective")
        // Printable ASCII maps to itself
        assertEquals('a', map['a'.code])
        assertEquals('!', map['!'.code])
        // Control bytes shift into the 0x100+ plane
        assertTrue(map[0].code >= 256)
        assertTrue(map[' '.code].code >= 256)
    }
}
