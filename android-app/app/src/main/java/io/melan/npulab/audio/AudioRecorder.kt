package io.melan.npulab.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Minimal 16 kHz mono PCM recorder for Whisper. Caller is responsible for the
 * RECORD_AUDIO runtime permission. Up to [maxSeconds] of audio is kept.
 */
class AudioRecorder(private val maxSeconds: Int = 30) {

    private val recording = AtomicBoolean(false)
    @Volatile private var thread: Thread? = null
    @Volatile private var samples = FloatArray(0)

    val isRecording: Boolean get() = recording.get()

    /** Begin capturing. @param onLevel called ~10×/s with RMS level 0..1. */
    @SuppressLint("MissingPermission")
    fun start(onLevel: ((Float) -> Unit)? = null) {
        if (!recording.compareAndSet(false, true)) return
        val sr = LogMelFrontend.SAMPLE_RATE
        val minBuf = AudioRecord.getMinBufferSize(
            sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sr,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, sr / 5 * 2),
        )
        val maxSamples = maxSeconds * sr
        thread = Thread {
            val chunk = ShortArray(sr / 10)
            val acc = ArrayList<FloatArray>(maxSeconds * 10 + 4)
            var total = 0
            rec.startRecording()
            try {
                while (recording.get() && total < maxSamples) {
                    val n = rec.read(chunk, 0, chunk.size)
                    if (n <= 0) break
                    val f = FloatArray(n)
                    var sumSq = 0.0
                    for (i in 0 until n) {
                        val v = chunk[i] / 32768f
                        f[i] = v
                        sumSq += v * v
                    }
                    acc.add(f)
                    total += n
                    onLevel?.invoke(kotlin.math.sqrt(sumSq / n).toFloat())
                }
            } finally {
                runCatching { rec.stop() }
                rec.release()
                recording.set(false)
            }
            val out = FloatArray(total)
            var pos = 0
            for (a in acc) {
                System.arraycopy(a, 0, out, pos, a.size)
                pos += a.size
            }
            samples = out
        }.also { it.start() }
    }

    /** Stop and return the captured PCM (blocks briefly for the writer thread). */
    fun stop(): FloatArray {
        recording.set(false)
        thread?.join(2000)
        thread = null
        return samples
    }
}
