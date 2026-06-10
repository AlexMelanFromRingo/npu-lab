package io.melan.npulab.inference

import org.junit.Test
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Quantization helpers are exercised with the REAL scale/offset values read out
 * of the AI Hub SD 1.5 context binaries (see binmeta fixtures) — so the ranges
 * and signs in these tests are exactly what the NPU sees.
 */
class QuantizeAndLayoutTest {

    // models/sd15/unet.bin → input "latent"
    private val latentScale = 0.00024176308943424374f
    private val latentOffset = -33983

    // models/sd15/unet.bin → input "timestep"
    private val timestepScale = 0.014770733192563057f

    @Test
    fun `quantize-dequantize round trip stays within half a step`() {
        val values = floatArrayOf(-8f, -1f, -0.1f, 0f, 0.1f, 1f, 7.5f)
        val buf = quantizeUfixed16(values, latentScale, latentOffset)
        val back = dequantizeUfixed16(buf, latentScale, latentOffset, values.size)
        for (i in values.indices) {
            assertTrue(
                abs(back[i] - values[i]) <= latentScale / 2 + 1e-7f,
                "v=${values[i]} back=${back[i]}",
            )
        }
    }

    @Test
    fun `quantization clamps out-of-range values instead of wrapping`() {
        // Representable range: scale*(0+offset) .. scale*(65535+offset) ≈ [-8.2, 7.6]
        val lo = latentScale * latentOffset
        val hi = latentScale * (0xFFFF + latentOffset)
        val buf = quantizeUfixed16(floatArrayOf(lo - 100f, hi + 100f), latentScale, latentOffset)
        val back = dequantizeUfixed16(buf, latentScale, latentOffset, 2)
        assertEquals(lo, back[0], latentScale)
        assertEquals(hi, back[1], latentScale)
    }

    @Test
    fun `leading schedule timesteps fit the timestep quant grid but 999 does not`() {
        // Documented reason for the "leading" timestep spacing: the binary's
        // grid tops out at scale*65535 ≈ 968.
        val maxRepresentable = timestepScale * 0xFFFF
        assertTrue(maxRepresentable < 999f, "sanity: grid really is narrower than 999")
        assertTrue(maxRepresentable > 951f, "sanity: grid covers leading max 951")

        val q951 = quantizeOneUfixed16(951f, timestepScale, 0).toInt() and 0xFFFF
        assertTrue(q951 < 0xFFFF, "951 must not clamp")
        val q999 = quantizeOneUfixed16(999f, timestepScale, 0).toInt() and 0xFFFF
        assertEquals(0xFFFF, q999, "999 would clamp — the schedule must avoid it")
    }

    @Test
    fun `negative offsets follow the QNN convention real = scale times q plus offset`() {
        // q=0 → most negative value; q=65535 → most positive.
        val buf = quantizeUfixed16(floatArrayOf(0f), latentScale, latentOffset)
        buf.rewind()
        val q = buf.short.toInt() and 0xFFFF
        assertEquals(-latentOffset, q, "0.0 must map to q=-offset (33983)")
    }

    @Test
    fun `chw to nhwc and back is the identity`() {
        val c = 4; val h = 8; val w = 8
        val src = FloatArray(c * h * w) { it.toFloat() }
        val round = nhwcToChw(chwToNhwc(src, c, h, w), c, h, w)
        assertContentEquals(src, round)
    }

    @Test
    fun `chw to nhwc places channels innermost`() {
        // CHW [2,1,2]: ch0=[0,1], ch1=[10,11] → NHWC: [0,10, 1,11]
        val chw = floatArrayOf(0f, 1f, 10f, 11f)
        val nhwc = chwToNhwc(chw, 2, 1, 2)
        assertContentEquals(floatArrayOf(0f, 10f, 1f, 11f), nhwc)
    }

    @Test
    fun `tensor spec wire format parses scale and offset`() {
        val spec = parseTensorSpec("latent:ufixed16:1,64,64,4:0.00024176309:-33983")
        assertEquals("latent", spec.name)
        assertEquals(TensorDtype.UFIXED16, spec.dtype)
        assertContentEquals(intArrayOf(1, 64, 64, 4), spec.shape)
        assertEquals(0.00024176309f, spec.scale, 1e-12f)
        assertEquals(-33983, spec.offset)
        assertTrue(spec.isQuantized)
        assertEquals(1 * 64 * 64 * 4, spec.numel)
        assertEquals(spec.numel * 2, spec.byteSize)
    }

    @Test
    fun `tensor spec without quant fields defaults to non-quantized`() {
        val spec = parseTensorSpec("tokens:int32:1,77:0:0")
        assertEquals(TensorDtype.INT32, spec.dtype)
        assertTrue(!spec.isQuantized)
        assertEquals(77 * 4, spec.byteSize)
    }
}
