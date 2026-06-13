package io.melan.npulab.vision

import io.melan.npulab.audio.Fp16
import io.melan.npulab.inference.TensorDtype
import io.melan.npulab.inference.TensorSpec
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Encode/decode model tensors as plain FloatArrays, honoring the tensor's
 * declared dtype and (for fixed-point) its scale/offset. Keeps the Vision
 * pipeline dtype-agnostic; unit-tested against the QNN convention
 * `real = scale * (q + offset)`.
 */
object TensorIo {

    fun directBuffer(byteSize: Int): ByteBuffer =
        ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())

    /** Encode [values] into a fresh direct buffer matching [spec]'s dtype. */
    fun encode(spec: TensorSpec, values: FloatArray): ByteBuffer {
        require(values.size == spec.numel) {
            "encode: ${values.size} values for tensor '${spec.name}' of ${spec.numel}"
        }
        val buf = directBuffer(spec.byteSize)
        when (spec.dtype) {
            TensorDtype.FLOAT32 -> for (v in values) buf.putFloat(v)
            TensorDtype.FLOAT16 -> for (v in values) buf.putShort(Fp16.floatToHalfBits(v))
            TensorDtype.INT32, TensorDtype.UINT32 -> for (v in values) buf.putInt(v.toInt())
            TensorDtype.INT16, TensorDtype.UINT16 -> for (v in values) buf.putShort(v.toInt().toShort())
            TensorDtype.INT8, TensorDtype.UINT8 -> for (v in values) buf.put(v.toInt().toByte())
            TensorDtype.UFIXED8, TensorDtype.SFIXED8 -> {
                val inv = 1f / spec.scale
                for (v in values) buf.put(((v * inv).toInt() - spec.offset).coerceIn(0, 0xFF).toByte())
            }
            TensorDtype.UFIXED16, TensorDtype.SFIXED16 -> {
                val inv = 1f / spec.scale
                for (v in values) buf.putShort(((v * inv).toInt() - spec.offset).coerceIn(0, 0xFFFF).toShort())
            }
        }
        buf.rewind()
        return buf
    }

    /** Decode a model output buffer into FloatArray of length [spec.numel]. */
    fun decode(spec: TensorSpec, buf: ByteBuffer): FloatArray {
        buf.rewind()
        val n = spec.numel
        val out = FloatArray(n)
        when (spec.dtype) {
            TensorDtype.FLOAT32 -> for (i in 0 until n) out[i] = buf.float
            TensorDtype.FLOAT16 -> for (i in 0 until n) out[i] = Fp16.halfBitsToFloat(buf.short)
            TensorDtype.INT32 -> for (i in 0 until n) out[i] = buf.int.toFloat()
            TensorDtype.UINT32 -> for (i in 0 until n) out[i] = (buf.int.toLong() and 0xFFFFFFFFL).toFloat()
            TensorDtype.INT16 -> for (i in 0 until n) out[i] = buf.short.toFloat()
            TensorDtype.UINT16 -> for (i in 0 until n) out[i] = (buf.short.toInt() and 0xFFFF).toFloat()
            TensorDtype.INT8 -> for (i in 0 until n) out[i] = buf.get().toFloat()
            TensorDtype.UINT8 -> for (i in 0 until n) out[i] = (buf.get().toInt() and 0xFF).toFloat()
            TensorDtype.UFIXED8, TensorDtype.SFIXED8 -> {
                for (i in 0 until n) {
                    val q = buf.get().toInt() and 0xFF
                    out[i] = spec.scale * (q + spec.offset)
                }
            }
            TensorDtype.UFIXED16, TensorDtype.SFIXED16 -> {
                for (i in 0 until n) {
                    val q = buf.short.toInt() and 0xFFFF
                    out[i] = spec.scale * (q + spec.offset)
                }
            }
        }
        return out
    }

    /** numerically-stable softmax in place-free form. */
    fun softmax(logits: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        var sum = 0.0
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = kotlin.math.exp((logits[i] - max).toDouble()).toFloat()
            exps[i] = e
            sum += e
        }
        val inv = (1.0 / sum).toFloat()
        for (i in exps.indices) exps[i] *= inv
        return exps
    }

    /** Indices of the [k] largest values, descending. */
    fun topK(values: FloatArray, k: Int): IntArray {
        val idx = (values.indices).sortedByDescending { values[it] }
        return idx.take(k).toIntArray()
    }
}
