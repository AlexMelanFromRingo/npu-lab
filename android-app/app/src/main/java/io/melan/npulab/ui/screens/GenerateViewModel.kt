package io.melan.npulab.ui.screens

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelKind
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.inference.QnnRuntime
import io.melan.npulab.inference.StableDiffusionPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class GenerateViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface State {
        data object Idle : State
        data class Loading(val message: String) : State
        data class Generating(
            val step: Int,
            val totalSteps: Int,
            val lastStepMs: Long,
        ) : State
        data class Done(
            val bitmap: Bitmap,
            val totalMs: Long,
            val perStepMedianMs: Long,
            val seed: Long,
        ) : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private var runtime: QnnRuntime? = null
    private var pipeline: StableDiffusionPipeline? = null
    private var currentJob: Job? = null

    fun generate(
        prompt: String,
        negativePrompt: String,
        numSteps: Int,
        cfgScale: Float,
        seed: Long,
        sampler: StableDiffusionPipeline.Sampler = StableDiffusionPipeline.Sampler.EULER,
    ) {
        if (currentJob?.isActive == true) return
        currentJob = viewModelScope.launch {
            _state.value = State.Loading("Initializing NPU…")
            try {
                val pipe = withContext(Dispatchers.IO) { ensurePipeline() }
                _state.value = State.Loading("Loading models…")
                val result = withContext(Dispatchers.IO) {
                    val resolvedSeed = if (seed < 0) System.nanoTime() else seed
                    pipe.generate(
                        prompt = prompt,
                        negativePrompt = negativePrompt,
                        numSteps = numSteps,
                        cfgScale = cfgScale,
                        seed = resolvedSeed,
                        sampler = sampler,
                        onProgress = { step ->
                            _state.value = State.Generating(
                                step = step.step + 1,
                                totalSteps = step.totalSteps,
                                lastStepMs = step.unetUs / 1000,
                            )
                        },
                    ).let { r -> r to resolvedSeed }
                }
                val (r, used) = result
                val mid = r.perStep
                    .sortedBy { it.unetUs }
                    .getOrNull(r.perStep.size / 2)?.unetUs ?: 0L
                _state.value = State.Done(
                    bitmap = r.bitmap,
                    totalMs = r.wallUs / 1000,
                    perStepMedianMs = mid / 1000,
                    seed = used,
                )
            } catch (t: Throwable) {
                _state.value = State.Failed(t.message ?: t::class.java.simpleName)
            }
        }
    }

    private fun ensurePipeline(): StableDiffusionPipeline {
        val cached = pipeline
        if (cached != null) return cached
        val ctx = getApplication<Application>()
        val store = ModelStore(ctx)
        val asset = ModelCatalog.byKind(ModelKind.STABLE_DIFFUSION_1_5)
        if (!store.isInstalled(asset)) {
            val missing = store.missingFilesFor(asset).joinToString(", ")
            throw IllegalStateException(
                "Stable Diffusion is not installed. Missing: $missing. " +
                "Install it on the Models tab, or run: python scripts/fetch-models.py && adb push models /sdcard/Android/data/io.melan.npulab/files/"
            )
        }
        val rt = runtime ?: QnnRuntime(
            backend = NpuLabNative.Backend.HTP,
            nativeLibDir = io.melan.npulab.inference.QnnRuntimeLibs.runtimeDir(ctx),
        ).also { runtime = it }
        return StableDiffusionPipeline(rt, store, asset).also { pipeline = it }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline?.close()
        runtime?.close()
        pipeline = null
        runtime = null
    }
}
