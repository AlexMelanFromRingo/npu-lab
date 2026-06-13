package io.melan.npulab.inference

import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class OnDeviceCompilerTest {

    @Test
    fun `compiled path lands in custom with arch tag`() {
        assertEquals(
            "custom/mobilenet_v2.htp81.bin",
            OnDeviceCompiler.compiledRelPath("zoo/mobilenet_v2.dlc", "htp81"),
        )
        assertEquals(
            "custom/my_model.htp.bin",
            OnDeviceCompiler.compiledRelPath("custom/my_model.dlc", "htp"),
        )
    }

    @Test
    fun `compiled binaries are recognized for cache cleanup and discovery`() {
        // What compiledRelPath produces must be matched by isCompiledBinary…
        val name = File(OnDeviceCompiler.compiledRelPath("zoo/resnet50.dlc", "htp81")).name
        assertTrue(OnDeviceCompiler.isCompiledBinary(name))
        assertTrue(OnDeviceCompiler.isCompiledBinary("foo.htp75.bin"))
        assertTrue(OnDeviceCompiler.isCompiledBinary("foo.htp.bin"))
        // …and a user-imported plain .bin must NOT be treated as our cache.
        assertFalse(OnDeviceCompiler.isCompiledBinary("whisper_encoder.bin"))
        assertFalse(OnDeviceCompiler.isCompiledBinary("mobilenet.dlc"))
    }

    @Test
    fun `compiled binary is discoverable as a custom asset`() {
        // The compiled .bin lands in custom/ so scanCustomBins surfaces it in
        // Benchmark/Vision as a fast context binary.
        val dir = createTempDir()
        File(dir, "mobilenet_v2.htp81.bin").writeBytes(ByteArray(16))
        val assets = scanCustomBins(dir)
        assertEquals(1, assets.size)
        assertEquals("mobilenet_v2.htp81", assets[0].displayName)
        assertFalse(assets[0].isDlc)  // it's a context binary now
        dir.deleteRecursively()
    }
}
