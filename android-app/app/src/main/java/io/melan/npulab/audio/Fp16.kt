package io.melan.npulab.audio

/**
 * IEEE 754 half-precision conversions in pure Kotlin (no android.util.Half —
 * these run in JVM unit tests too). Round-to-nearest-even on encode.
 *
 * The Whisper context binaries exchange everything as FLOAT_16: encoder
 * features in, cross/self KV caches, attention mask, logits out.
 */
object Fp16 {

    fun floatToHalfBits(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exp = (bits ushr 23) and 0xFF
        var mant = bits and 0x7FFFFF

        if (exp == 0xFF) { // NaN / Inf
            val nan = if (mant != 0) 0x200 else 0
            return (sign or 0x7C00 or nan).toShort()
        }
        // Re-bias 127 → 15
        var e = exp - 127 + 15
        if (e >= 0x1F) return (sign or 0x7C00).toShort()          // overflow → Inf
        if (e <= 0) {
            // Subnormal half (or underflow to zero)
            if (e < -10) return sign.toShort()
            mant = mant or 0x800000
            val shift = 14 - e
            val half = mant ushr shift
            // round to nearest even
            val rem = mant and ((1 shl shift) - 1)
            val halfway = 1 shl (shift - 1)
            val rounded = if (rem > halfway || (rem == halfway && (half and 1) == 1)) half + 1 else half
            return (sign or rounded).toShort()
        }
        var half = (e shl 10) or (mant ushr 13)
        val rem = mant and 0x1FFF
        if (rem > 0x1000 || (rem == 0x1000 && (half and 1) == 1)) half += 1
        return (sign or half).toShort()
    }

    fun halfBitsToFloat(half: Short): Float {
        val h = half.toInt() and 0xFFFF
        val sign = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x3FF
        val bits: Int = when {
            exp == 0 -> {
                if (mant == 0) {
                    sign
                } else {
                    // subnormal: normalize
                    var e = -1
                    var m = mant
                    do { m = m shl 1; e++ } while (m and 0x400 == 0)
                    sign or ((127 - 15 - e) shl 23) or ((m and 0x3FF) shl 13)
                }
            }
            exp == 0x1F -> sign or 0x7F800000 or (mant shl 13)   // Inf / NaN
            else -> sign or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return Float.fromBits(bits)
    }
}
