package io.melan.npulab.inference

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CatalogIdTest {

    @Test
    fun `every catalog asset id is unique`() {
        // The Models/Benchmark lists key LazyColumn items and install state by
        // id. Duplicate ids → "key was already used" crash on scroll. The whole
        // zoo shared ModelKind.ZOO before ids existed — this guards the fix.
        val ids = ModelCatalog.all.map { it.id }
        assertEquals(ids.size, ids.toSet().size, "duplicate ids: " +
            ids.groupingBy { it }.eachCount().filter { it.value > 1 })
    }

    @Test
    fun `every asset has a usable model file and a category`() {
        for (a in ModelCatalog.all) {
            assertTrue(a.primaryModelFile != null, "${a.id} has no .bin/.dlc")
            assertTrue(a.expectedFiles.isNotEmpty(), "${a.id} has no files")
        }
    }

    @Test
    fun `zoo entries are DLCs that run on any backend`() {
        val zoo = ModelCatalog.all.filter { it.kind == ModelKind.ZOO }
        assertTrue(zoo.size >= 25, "expected a sizable zoo, got ${zoo.size}")
        for (z in zoo) assertTrue(z.isDlc, "${z.id} should be a DLC")
    }

    @Test
    fun `classification zoo models pull labels alongside the dlc`() {
        val classifiers = ModelCatalog.all.filter { it.category == ModelCategory.CLASSIFICATION }
        assertTrue(classifiers.isNotEmpty())
        for (c in classifiers) {
            val src = c.installSource!!
            assertTrue(
                src.extractMap.containsKey("labels.txt"),
                "${c.id} must extract labels.txt for the Vision tab",
            )
        }
    }

    @Test
    fun `custom scan accepts both bin and dlc and skips empty`() {
        val dir = createTempDir()
        java.io.File(dir, "a.dlc").writeBytes(ByteArray(8))
        java.io.File(dir, "b.bin").writeBytes(ByteArray(8))
        java.io.File(dir, "c.dlc").writeBytes(ByteArray(0)) // empty → skipped
        java.io.File(dir, "notes.txt").writeText("x")
        val assets = scanCustomBins(dir)
        assertEquals(listOf("a", "b"), assets.map { it.displayName })
        assertEquals(setOf("custom_a.dlc", "custom_b.bin"), assets.map { it.id }.toSet())
        dir.deleteRecursively()
    }
}
