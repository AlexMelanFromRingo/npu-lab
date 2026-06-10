package io.melan.npulab.inference

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CustomModelScanTest {

    private fun tempDir(): File =
        File.createTempFile("custom", "").let {
            it.delete(); it.mkdirs(); it.deleteOnExit(); it
        }

    @Test
    fun `bins become assets sorted by name`() {
        val dir = tempDir()
        File(dir, "zeta.bin").writeBytes(ByteArray(4) { 1 })
        File(dir, "Alpha.bin").writeBytes(ByteArray(4) { 1 })
        File(dir, "notes.txt").writeText("ignore me")
        File(dir, "empty.bin").writeBytes(ByteArray(0))   // skipped: zero length

        val assets = scanCustomBins(dir)
        assertEquals(listOf("Alpha", "zeta"), assets.map { it.displayName })
        for (a in assets) {
            assertEquals(ModelKind.CUSTOM, a.kind)
            assertEquals(1, a.expectedFiles.size)
            assertTrue(a.expectedFiles[0].startsWith("custom/"))
            assertTrue(a.expectedFiles[0].endsWith(".bin"))
        }
    }

    @Test
    fun `missing dir yields empty list`() {
        assertEquals(emptyList(), scanCustomBins(File("/nonexistent/custom")))
    }
}
