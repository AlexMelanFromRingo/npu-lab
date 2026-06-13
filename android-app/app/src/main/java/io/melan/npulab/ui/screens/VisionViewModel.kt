package io.melan.npulab.ui.screens

import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.vision.VisionPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VisionViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface State {
        data object Idle : State
        data class Loading(val message: String) : State
        data class Done(val result: VisionPipeline.Result) : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _input = MutableStateFlow<Bitmap?>(null)
    val input: StateFlow<Bitmap?> = _input.asStateFlow()

    private val _model = MutableStateFlow<ModelAsset?>(null)
    val model: StateFlow<ModelAsset?> = _model.asStateFlow()

    private var pipeline: VisionPipeline? = null
    private var pipelineId: String? = null

    /** Installed image models the Vision screen can render, + custom. */
    fun availableModels(): List<ModelAsset> {
        val store = ModelStore(getApplication())
        val catalog = ModelCatalog.all.filter {
            it.category in VisionPipeline.SUPPORTED && store.isInstalled(it)
        }
        val custom = store.customAssets() // category OTHER → timing/raw only
        return catalog + custom
    }

    fun selectModel(asset: ModelAsset) {
        if (_model.value?.id == asset.id) return
        _model.value = asset
        _state.value = State.Idle
        if (pipelineId != asset.id) {
            pipeline?.close(); pipeline = null; pipelineId = null
        }
    }

    fun setInput(bitmap: Bitmap?) {
        _input.value = bitmap
        if (_state.value is State.Done) _state.value = State.Idle
    }

    fun loadInputFromUri(uri: Uri) {
        viewModelScope.launch {
            val bmp = withContext(Dispatchers.IO) { decodeDownsampled(uri) }
            setInput(bmp)
        }
    }

    fun run() {
        val asset = _model.value ?: run {
            _state.value = State.Failed("Pick a model first"); return
        }
        val bmp = _input.value ?: run {
            _state.value = State.Failed("Pick an image or take a photo first"); return
        }
        _state.value = State.Loading("Initializing NPU…")
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val pipe = ensurePipeline(asset)
                    _state.value = State.Loading("Running ${asset.displayName} on HTP…")
                    pipe.run(bmp)
                }
                _state.value = State.Done(result)
            } catch (t: Throwable) {
                _state.value = State.Failed(t.message ?: t::class.java.simpleName)
            }
        }
    }

    private fun ensurePipeline(asset: ModelAsset): VisionPipeline {
        pipeline?.let { if (pipelineId == asset.id) return it }
        pipeline?.close()
        return VisionPipeline(getApplication(), asset).also {
            pipeline = it; pipelineId = asset.id
        }
    }

    private fun decodeDownsampled(uri: Uri): Bitmap {
        val src = ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
        return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.isMutableRequired = false
            val max = 1280
            val w = info.size.width; val h = info.size.height
            val scale = maxOf(1, maxOf(w, h) / max)
            if (scale > 1) decoder.setTargetSampleSize(scale)
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }.let { if (it.config == Bitmap.Config.ARGB_8888) it else it.copy(Bitmap.Config.ARGB_8888, false) }
    }

    override fun onCleared() {
        super.onCleared()
        pipeline?.close()
        pipeline = null
    }
}
