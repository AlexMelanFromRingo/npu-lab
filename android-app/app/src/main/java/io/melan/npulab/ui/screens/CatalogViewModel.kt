package io.melan.npulab.ui.screens

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelKind
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.install.ModelInstaller
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class CatalogViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface CardState {
        data object Idle : CardState
        data class Downloading(val bytesDone: Long, val bytesTotal: Long, val phase: String) : CardState
        data object Extracting : CardState
        data class Installed(val totalBytes: Long) : CardState
        data class Failed(val message: String) : CardState
    }

    data class UiState(
        val installed: Set<ModelKind> = emptySet(),
        val states: Map<ModelKind, CardState> = emptyMap(),
    )

    private val installer = ModelInstaller(app.applicationContext)
    private val store = ModelStore(app.applicationContext)
    private val jobs = mutableMapOf<ModelKind, Job>()

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    init {
        refreshInstalled()
    }

    fun refreshInstalled() {
        _state.update { it.copy(installed = store.installedKinds()) }
    }

    fun install(asset: ModelAsset) {
        if (jobs[asset.kind]?.isActive == true) return
        jobs[asset.kind] = viewModelScope.launch {
            installer.install(asset).collect { p ->
                val cardState = when (p) {
                    is ModelInstaller.Progress.Downloading ->
                        CardState.Downloading(p.bytesDone, p.bytesTotal, p.phase)
                    is ModelInstaller.Progress.Extracting -> CardState.Extracting
                    is ModelInstaller.Progress.Installed -> {
                        refreshInstalled()
                        CardState.Installed(p.totalBytes)
                    }
                    is ModelInstaller.Progress.Failed -> CardState.Failed(p.message)
                }
                _state.update { it.copy(states = it.states + (asset.kind to cardState)) }
            }
        }
    }

    fun cancel(asset: ModelAsset) {
        jobs[asset.kind]?.cancel()
        jobs.remove(asset.kind)
        _state.update { it.copy(states = it.states - asset.kind) }
    }

    fun uninstall(asset: ModelAsset) {
        installer.uninstall(asset)
        refreshInstalled()
        _state.update { it.copy(states = it.states - asset.kind) }
    }
}
