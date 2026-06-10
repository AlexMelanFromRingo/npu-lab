package io.melan.npulab.ui.screens

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.benchmark.BenchmarkRow
import io.melan.npulab.benchmark.BenchmarkRunner
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.NpuLabNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkViewModel(app: Application) : AndroidViewModel(app) {

    data class UiState(
        val running: Boolean = false,
        val progressLabel: String = "",
        val rows: List<BenchmarkRow> = emptyList(),
        val error: String? = null,
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var currentJob: Job? = null

    fun run(
        models: List<ModelAsset>,
        backends: List<NpuLabNative.Backend>,
        iterations: Int = 8,
        warmup: Int = 2,
    ) {
        if (currentJob?.isActive == true) return
        if (models.isEmpty() || backends.isEmpty()) {
            _state.value = UiState(error = "Pick at least one model and one backend")
            return
        }
        _state.value = UiState(running = true, progressLabel = "0 / ${models.size * backends.size}")
        currentJob = viewModelScope.launch {
            val ctx = getApplication<Application>()
            val runner = BenchmarkRunner(ctx)
            val totalCells = models.size * backends.size
            var done = 0
            try {
                withContext(Dispatchers.IO) {
                    runner.run(
                        models = models,
                        backends = backends,
                        iterationsPerCombination = iterations,
                        warmupIterations = warmup,
                        onProgress = { row ->
                            done += 1
                            _state.update {
                                it.copy(
                                    progressLabel = "$done / $totalCells",
                                    rows = it.rows + row,
                                )
                            }
                        },
                    )
                }
                _state.update { it.copy(running = false) }
            } catch (t: Throwable) {
                _state.update {
                    it.copy(running = false, error = t.message ?: t::class.java.simpleName)
                }
            }
        }
    }

    fun cancel() {
        currentJob?.cancel()
        _state.update { it.copy(running = false) }
    }
}
