package io.melan.npulab.vision

import android.graphics.Bitmap
import android.graphics.Color

/**
 * Input tensor geometry for an image model: channel order + H/W/C, derived
 * from the model's declared input shape. NCHW `[1,3,H,W]` and NHWC `[1,H,W,3]`
 * (and single-channel variants) are both handled.
 */
data class InputGeometry(val nchw: Boolean, val channels: Int, val height: Int, val width: Int) {
    companion object {
        /** Returns null if [shape] is not a recognizable 4-D image input. */
        fun parse(shape: IntArray): InputGeometry? {
            if (shape.size != 4 || shape[0] != 1) return null
            val a = shape[1]; val b = shape[2]; val c = shape[3]
            return when {
                a == 3 || a == 1 -> InputGeometry(nchw = true, channels = a, height = b, width = c)
                c == 3 || c == 1 -> InputGeometry(nchw = false, channels = c, height = a, width = b)
                else -> null
            }
        }
    }
}

/**
 * Pure tensor↔image math (no Bitmap), so JVM tests can cover it. Bitmap-bound
 * helpers live in [BitmapOps].
 */
object ImageOps {

    /** Min-max normalize to [0,1]; flat input → all zeros. */
    fun normalize01(values: FloatArray): FloatArray {
        var lo = Float.POSITIVE_INFINITY; var hi = Float.NEGATIVE_INFINITY
        for (v in values) { if (v < lo) lo = v; if (v > hi) hi = v }
        val span = hi - lo
        if (span <= 0f) return FloatArray(values.size)
        val inv = 1f / span
        return FloatArray(values.size) { (values[it] - lo) * inv }
    }

    /**
     * Per-pixel argmax over the class dimension of a segmentation output.
     * @param data flat logits
     * @param classes number of classes
     * @param pixels H*W
     * @param channelFirst true → layout [C, H*W] (NCHW); false → [H*W, C] (NHWC)
     * @return IntArray of length [pixels] with the winning class index.
     */
    fun argmaxPerPixel(data: FloatArray, classes: Int, pixels: Int, channelFirst: Boolean): IntArray {
        val out = IntArray(pixels)
        for (p in 0 until pixels) {
            var best = Float.NEGATIVE_INFINITY; var bestC = 0
            for (cl in 0 until classes) {
                val v = if (channelFirst) data[cl * pixels + p] else data[p * classes + cl]
                if (v > best) { best = v; bestC = cl }
            }
            out[p] = bestC
        }
        return out
    }

    /** A compact "turbo"-style colormap: t in [0,1] → packed ARGB (pure). */
    fun colormap(t: Float): Int {
        val x = t.coerceIn(0f, 1f)
        // 5-stop perceptual gradient (deep blue → cyan → green → yellow → red).
        val stops = intArrayOf(0x224488, 0x22AACC, 0x44CC44, 0xEEDD22, 0xCC2222)
        val seg = (x * (stops.size - 1))
        val i = seg.toInt().coerceIn(0, stops.size - 2)
        val f = seg - i
        val a = stops[i]; val b = stops[i + 1]
        val r = lerp((a shr 16) and 0xFF, (b shr 16) and 0xFF, f)
        val g = lerp((a shr 8) and 0xFF, (b shr 8) and 0xFF, f)
        val bl = lerp(a and 0xFF, b and 0xFF, f)
        return packArgb(255, r, g, bl)
    }

    /** Distinct-ish palette color for a class index (segmentation overlay). */
    fun classColor(cls: Int): Int {
        if (cls == 0) return 0 // background → fully transparent
        // Golden-ratio hue stepping for visually separated colors.
        val hue = (cls * 137.508f) % 360f
        return hsvToArgb(hue, 0.65f, 0.95f, 180)
    }

    /** Pack ARGB without android.graphics.Color, so this stays unit-testable. */
    fun packArgb(a: Int, r: Int, g: Int, b: Int): Int =
        ((a and 0xFF) shl 24) or ((r and 0xFF) shl 16) or ((g and 0xFF) shl 8) or (b and 0xFF)

    private fun lerp(a: Int, b: Int, f: Float): Int = (a + (b - a) * f).toInt().coerceIn(0, 255)

    private fun hsvToArgb(h: Float, s: Float, v: Float, alpha: Int): Int {
        val c = v * s
        val x = c * (1 - kotlin.math.abs((h / 60f) % 2 - 1))
        val m = v - c
        val (r, g, b) = when {
            h < 60 -> Triple(c, x, 0f)
            h < 120 -> Triple(x, c, 0f)
            h < 180 -> Triple(0f, c, x)
            h < 240 -> Triple(0f, x, c)
            h < 300 -> Triple(x, 0f, c)
            else -> Triple(c, 0f, x)
        }
        return packArgb(
            alpha,
            ((r + m) * 255).toInt(), ((g + m) * 255).toInt(), ((b + m) * 255).toInt(),
        )
    }
}

/** Bitmap-bound helpers (not unit-tested — require android.graphics). */
object BitmapOps {

    /** Resize [src] to [w]×[h] with bilinear filtering, ARGB_8888. */
    fun resize(src: Bitmap, w: Int, h: Int): Bitmap =
        Bitmap.createScaledBitmap(src, w, h, true)

    /**
     * Pixels of a [w]×[h] bitmap → CHW or HWC float array in [0,1], RGB order.
     */
    fun toFloat01(bmp: Bitmap, geo: InputGeometry): FloatArray {
        val w = geo.width; val h = geo.height
        val scaled = if (bmp.width != w || bmp.height != h) resize(bmp, w, h) else bmp
        val px = IntArray(w * h)
        scaled.getPixels(px, 0, w, 0, 0, w, h)
        val c = geo.channels
        val out = FloatArray(c * w * h)
        for (p in 0 until w * h) {
            val pixel = px[p]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            if (c == 1) {
                val gray = 0.299f * r + 0.587f * g + 0.114f * b
                out[p] = gray
            } else if (geo.nchw) {
                out[p] = r
                out[w * h + p] = g
                out[2 * w * h + p] = b
            } else {
                out[p * 3] = r
                out[p * 3 + 1] = g
                out[p * 3 + 2] = b
            }
        }
        return out
    }

    /** Render an RGB image tensor in [0,1] (CHW or HWC) to a Bitmap. */
    fun fromFloat01(data: FloatArray, w: Int, h: Int, nchw: Boolean, channels: Int): Bitmap {
        val px = IntArray(w * h)
        for (p in 0 until w * h) {
            val r: Float; val g: Float; val b: Float
            if (channels == 1) {
                val v = if (nchw) data[p] else data[p]
                r = v; g = v; b = v
            } else if (nchw) {
                r = data[p]; g = data[w * h + p]; b = data[2 * w * h + p]
            } else {
                r = data[p * 3]; g = data[p * 3 + 1]; b = data[p * 3 + 2]
            }
            px[p] = Color.argb(255, c8(r), c8(g), c8(b))
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    /** Grayscale/colormap a single-channel field (depth) to a Bitmap. */
    fun depthBitmap(values: FloatArray, w: Int, h: Int): Bitmap {
        val norm = ImageOps.normalize01(values)
        val px = IntArray(w * h)
        for (p in 0 until w * h) px[p] = ImageOps.colormap(norm[p])
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    /** Color a per-pixel class map (segmentation) to a Bitmap. */
    fun segmentationBitmap(classes: IntArray, w: Int, h: Int): Bitmap {
        val px = IntArray(w * h) { ImageOps.classColor(classes[it]) }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun c8(v: Float): Int = (v.coerceIn(0f, 1f) * 255f).toInt()
}
