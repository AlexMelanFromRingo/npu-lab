package io.melan.npulab.ui.screens

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.audio.AudioRecorder
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.inference.QnnRuntime
import io.melan.npulab.inference.QnnRuntimeLibs
import io.melan.npulab.inference.WhisperPipeline
import io.melan.npulab.inference.WhisperVariant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TranscribeViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface State {
        data object Idle : State
        data class Recording(val seconds: Int, val level: Float) : State
        data class Loading(val message: String) : State
        data class Transcribing(val partial: String, val chunk: Int = 1, val totalChunks: Int = 1) : State
        data class Done(
            val text: String,
            val tokens: Int,
            val encoderMs: Long,
            val decodeMs: Long,
            val melMs: Long,
            val tokensPerSecond: Float,
            val language: String?,
        ) : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /** null = auto-detect; otherwise an ISO code from WhisperTokenizer.LANGUAGE_IDS. */
    val language = MutableStateFlow<String?>(null)

    /** Selected Whisper flavor. Switching drops the loaded pipeline. */
    val variant = MutableStateFlow(WhisperVariant.BASE)

    private val recorder = AudioRecorder(maxSeconds = MAX_RECORD_SECONDS)
    private var runtime: QnnRuntime? = null
    private var pipeline: WhisperPipeline? = null
    private var pipelineVariant: WhisperVariant? = null
    private var recStartMs = 0L

    fun selectVariant(v: WhisperVariant) {
        if (variant.value == v) return
        variant.value = v
        if (pipelineVariant != v) {
            pipeline?.close()
            pipeline = null
            pipelineVariant = null
        }
    }

    /** True when every file the variant needs is on device. */
    fun isInstalled(v: WhisperVariant): Boolean {
        val store = ModelStore(getApplication())
        return v.requiredFiles().all { store.pathOf(it).isFile }
    }

    private fun missingFiles(v: WhisperVariant): List<String> {
        val store = ModelStore(getApplication())
        return v.requiredFiles().filterNot { store.pathOf(it).isFile }
    }

    fun startRecording() {
        if (recorder.isRecording) return
        recStartMs = System.currentTimeMillis()
        recorder.start { level ->
            val sec = ((System.currentTimeMillis() - recStartMs) / 1000).toInt()
            _state.update { s ->
                if (s is State.Recording || s is State.Idle || s is State.Done || s is State.Failed) {
                    State.Recording(sec, level)
                } else s
            }
            if (sec >= MAX_RECORD_SECONDS) viewModelScope.launch { stopAndTranscribe() }
        }
        _state.value = State.Recording(0, 0f)
    }

    fun stopAndTranscribe() {
        if (!recorder.isRecording && state.value !is State.Recording) return
        viewModelScope.launch {
            val pcm = withContext(Dispatchers.IO) { recorder.stop() }
            if (pcm.size < LogMinSamples) {
                _state.value = State.Failed(
                    "Recording too short (${"%.1f".format(pcm.size / 16000.0)} s)"
                )
                return@launch
            }
            val v = variant.value
            val missing = missingFiles(v)
            if (missing.isNotEmpty()) {
                _state.value = State.Failed(
                    "Missing files for Whisper ${v.displayName}: ${missing.joinToString()}. " +
                        "Open the Models tab and install it."
                )
                return@launch
            }
            _state.value = State.Loading("Initializing NPU…")
            try {
                val lang = language.value
                val result = withContext(Dispatchers.IO) {
                    val pipe = ensurePipeline(v)
                    _state.value = State.Transcribing("")
                    pipe.transcribeLong(pcm, lang) { chunk, total, partial ->
                        _state.value = State.Transcribing(partial, chunk, total)
                    }
                }
                _state.value = State.Done(
                    text = result.text.ifBlank { "(empty transcription)" },
                    tokens = result.tokens,
                    encoderMs = result.encoderMs,
                    decodeMs = result.decodeMs,
                    melMs = result.melMs,
                    tokensPerSecond = result.tokensPerSecond,
                    language = result.language,
                )
            } catch (t: Throwable) {
                _state.value = State.Failed(t.message ?: t::class.java.simpleName)
            }
        }
    }

    private fun ensurePipeline(v: WhisperVariant): WhisperPipeline {
        pipeline?.let { if (pipelineVariant == v) return it }
        pipeline?.close()
        val ctx = getApplication<Application>()
        val rt = runtime ?: QnnRuntime(
            backend = NpuLabNative.Backend.HTP,
            nativeLibDir = QnnRuntimeLibs.runtimeDir(ctx),
        ).also { runtime = it }
        return WhisperPipeline(rt, ModelStore(ctx), ctx, v).also {
            pipeline = it
            pipelineVariant = v
        }
    }

    override fun onCleared() {
        super.onCleared()
        runCatching { recorder.stop() }
        pipeline?.close()
        runtime?.close()
        pipeline = null
        runtime = null
    }

    companion object {
        private const val LogMinSamples = 8000 // 0.5 s
        const val MAX_RECORD_SECONDS = 120
    }
}
