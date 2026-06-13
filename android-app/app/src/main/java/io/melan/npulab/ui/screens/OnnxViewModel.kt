package io.melan.npulab.ui.screens

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.inference.OrtNpuRunner
import io.melan.npulab.inference.QnnRuntimeLibs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Drives the "ONNX → NPU" screen: import any .onnx from storage, then run it on
 * the Hexagon NPU (or QNN GPU/CPU) via [OrtNpuRunner] — on-device, no PC step.
 */
class OnnxViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface State {
        data object Idle : State
        data class Running(val message: String) : State
        data class Done(val result: OrtNpuRunner.Result, val backend: OrtNpuRunner.Backend) : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _models = MutableStateFlow<List<File>>(emptyList())
    val models: StateFlow<List<File>> = _models.asStateFlow()

    private val _selected = MutableStateFlow<File?>(null)
    val selected: StateFlow<File?> = _selected.asStateFlow()

    val backend = MutableStateFlow(OrtNpuRunner.Backend.HTP)

    private val onnxDir = File(app.filesDir, "onnx").apply { mkdirs() }

    init { refresh() }

    fun refresh() {
        _models.value = onnxDir.listFiles { f -> f.isFile && f.name.endsWith(".onnx") }
            ?.sortedBy { it.name.lowercase() } ?: emptyList()
        if (_selected.value?.exists() != true) _selected.value = _models.value.firstOrNull()
    }

    fun select(f: File) { _selected.value = f; _state.value = State.Idle }
    fun setBackend(b: OrtNpuRunner.Backend) { backend.value = b }

    fun importOnnx(uri: Uri) {
        viewModelScope.launch {
            _state.value = State.Running("Importing…")
            val ok = withContext(Dispatchers.IO) { copyIn(uri) }
            refresh()
            _state.value = if (ok != null) State.Idle
            else State.Failed("Import failed — pick a .onnx file")
            if (ok != null) _selected.value = ok
        }
    }

    fun run() {
        val model = _selected.value ?: run {
            _state.value = State.Failed("Import and pick an .onnx first"); return
        }
        val b = backend.value
        _state.value = State.Running("Compiling & running on ${b.label}…")
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                OrtNpuRunner.benchmark(
                    modelPath = model.absolutePath,
                    nativeLibDir = QnnRuntimeLibs.runtimeDir(getApplication()),
                    backend = b,
                )
            }
            _state.value = if (result.ok) State.Done(result, b)
            else State.Failed(result.error ?: "run failed")
        }
    }

    private fun copyIn(uri: Uri): File? {
        val name = displayName(uri)?.takeIf { it.endsWith(".onnx", true) } ?: return null
        val safe = name.substringAfterLast('/').replace(Regex("[^A-Za-z0-9._-]"), "_")
        val dest = File(onnxDir, safe)
        return try {
            getApplication<Application>().contentResolver.openInputStream(uri).use { input ->
                dest.outputStream().use { out -> input!!.copyTo(out) }
            }
            dest
        } catch (t: Throwable) { null }
    }

    private fun displayName(uri: Uri): String? {
        getApplication<Application>().contentResolver.query(uri, null, null, null, null)?.use { c ->
            val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (i >= 0 && c.moveToFirst()) return c.getString(i)
        }
        return uri.lastPathSegment
    }
}
