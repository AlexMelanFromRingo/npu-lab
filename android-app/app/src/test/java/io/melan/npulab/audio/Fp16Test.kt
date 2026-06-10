package io.melan.npulab.audio

import org.junit.Test
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Fp16Test {

    @Test
    fun `known constants round trip`() {
        // Reference half-precision bit patterns.
        assertEquals(0x3C00.toShort(), Fp16.floatToHalfBits(1.0f))
        assertEquals(0xBC00.toShort(), Fp16.floatToHalfBits(-1.0f))
        assertEquals(0x0000.toShort(), Fp16.floatToHalfBits(0.0f))
        assertEquals(0x7C00.toShort(), Fp16.floatToHalfBits(Float.POSITIVE_INFINITY))
        assertEquals(0xD640.toShort(), Fp16.floatToHalfBits(-100.0f), "Whisper MASK_NEG")
        assertEquals(-100.0f, Fp16.halfBitsToFloat(0xD640.toShort()))
        assertEquals(1.0f, Fp16.halfBitsToFloat(0x3C00.toShort()))
        // 65504 = max finite half
        assertEquals(0x7BFF.toShort(), Fp16.floatToHalfBits(65504f))
        assertEquals(65504f, Fp16.halfBitsToFloat(0x7BFF.toShort()))
        // overflow → inf
        assertEquals(0x7C00.toShort(), Fp16.floatToHalfBits(70000f))
    }

    @Test
    fun `round trip across the normal range stays within half ulp`() {
        var v = 1e-4f
        while (v < 6e4f) {
            for (s in floatArrayOf(v, -v)) {
                val back = Fp16.halfBitsToFloat(Fp16.floatToHalfBits(s))
                val rel = abs(back - s) / abs(s)
                assertTrue(rel <= 1.0f / 1024f, "v=$s back=$back rel=$rel")
            }
            v *= 1.37f
        }
    }

    @Test
    fun `subnormals survive`() {
        val tiny = 3.0e-6f // subnormal half territory
        val back = Fp16.halfBitsToFloat(Fp16.floatToHalfBits(tiny))
        assertTrue(abs(back - tiny) < 1e-7f, "back=$back")
    }
}
