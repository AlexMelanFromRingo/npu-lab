package io.melan.npulab.vision

import io.melan.npulab.inference.TensorDtype
import io.melan.npulab.inference.TensorSpec
import org.junit.Test
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VisionMathTest {

    /* -------- TensorIo -------- */

    @Test
    fun `softmax sums to one and ranks preserved`() {
        val p = TensorIo.softmax(floatArrayOf(1f, 2f, 3f, 0f))
        assertEquals(1f, p.sum(), 1e-5f)
        assertTrue(p[2] > p[1] && p[1] > p[0] && p[0] > p[3])
    }

    @Test
    fun `topK returns largest indices descending`() {
        val k = TensorIo.topK(floatArrayOf(0.1f, 0.9f, 0.5f, 0.95f, 0.2f), 3)
        assertContentEquals(intArrayOf(3, 1, 2), k)
    }

    @Test
    fun `float32 encode-decode round trips`() {
        val spec = TensorSpec("x", TensorDtype.FLOAT32, intArrayOf(1, 4))
        val v = floatArrayOf(-1.5f, 0f, 3.25f, 100f)
        val back = TensorIo.decode(spec, TensorIo.encode(spec, v))
        assertContentEquals(v, back)
    }

    @Test
    fun `float16 encode-decode round trips within half precision`() {
        val spec = TensorSpec("x", TensorDtype.FLOAT16, intArrayOf(1, 3))
        val v = floatArrayOf(0.1f, -2.5f, 12.0f)
        val back = TensorIo.decode(spec, TensorIo.encode(spec, v))
        for (i in v.indices) assertTrue(abs(back[i] - v[i]) / abs(v[i]) < 1e-2f)
    }

    /* -------- InputGeometry -------- */

    @Test
    fun `geometry detects NCHW NHWC and grayscale`() {
        InputGeometry.parse(intArrayOf(1, 3, 224, 224))!!.let {
            assertTrue(it.nchw); assertEquals(3, it.channels); assertEquals(224, it.height)
        }
        InputGeometry.parse(intArrayOf(1, 256, 256, 3))!!.let {
            assertTrue(!it.nchw); assertEquals(3, it.channels); assertEquals(256, it.width)
        }
        InputGeometry.parse(intArrayOf(1, 1, 384, 384))!!.let {
            assertTrue(it.nchw); assertEquals(1, it.channels)
        }
        assertEquals(null, InputGeometry.parse(intArrayOf(1, 1000)))
    }

    /* -------- ImageOps -------- */

    @Test
    fun `normalize01 maps to unit range`() {
        val n = ImageOps.normalize01(floatArrayOf(2f, 4f, 6f))
        assertContentEquals(floatArrayOf(0f, 0.5f, 1f), n)
        // flat input → all zeros (no divide by zero)
        assertContentEquals(floatArrayOf(0f, 0f), ImageOps.normalize01(floatArrayOf(5f, 5f)))
    }

    @Test
    fun `argmax per pixel works for NCHW and NHWC`() {
        // 2 pixels, 3 classes. pixel0 → class2, pixel1 → class0.
        // NCHW layout [C][P]: c0=[0.1,0.9] c1=[0.2,0.1] c2=[0.8,0.0]
        val nchw = floatArrayOf(0.1f, 0.9f,  0.2f, 0.1f,  0.8f, 0.0f)
        assertContentEquals(intArrayOf(2, 0), ImageOps.argmaxPerPixel(nchw, 3, 2, channelFirst = true))
        // NHWC layout [P][C]: p0=[0.1,0.2,0.8] p1=[0.9,0.1,0.0]
        val nhwc = floatArrayOf(0.1f, 0.2f, 0.8f,  0.9f, 0.1f, 0.0f)
        assertContentEquals(intArrayOf(2, 0), ImageOps.argmaxPerPixel(nhwc, 3, 2, channelFirst = false))
    }

    @Test
    fun `colormap endpoints differ and stay in range`() {
        val lo = ImageOps.colormap(0f)
        val hi = ImageOps.colormap(1f)
        assertTrue(lo != hi)
        // alpha is fully opaque
        assertEquals(0xFF, (lo ushr 24) and 0xFF)
        assertEquals(0xFF, (hi ushr 24) and 0xFF)
    }

    @Test
    fun `class color zero is transparent background`() {
        assertEquals(0, (ImageOps.classColor(0) ushr 24) and 0xFF)
        assertTrue(((ImageOps.classColor(7) ushr 24) and 0xFF) > 0)
    }
}
