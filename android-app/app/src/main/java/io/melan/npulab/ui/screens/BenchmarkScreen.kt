package io.melan.npulab.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.benchmark.BenchmarkRow
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun BenchmarkScreen(vm: BenchmarkViewModel = viewModel()) {
    val ui by vm.state.collectAsStateWithLifecycle()
    val ctx = LocalContext.current

    // Catalog models + user-compiled .bin files from models/custom/.
    val allModels by produceState(initialValue = ModelCatalog.all) {
        value = withContext(Dispatchers.IO) {
            ModelCatalog.all + ModelStore(ctx).customAssets()
        }
    }
    val selectedModels = remember {
        mutableStateListOf<String>().apply { addAll(ModelCatalog.all.map { it.displayName }) }
    }
    val selectedBackends = remember {
        mutableStateListOf<NpuLabNative.Backend>().apply {
            addAll(listOf(NpuLabNative.Backend.HTP, NpuLabNative.Backend.GPU, NpuLabNative.Backend.CPU))
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        TopAppBar(
            title = { Text("Benchmark") },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item { VSpace(4); SectionTitle("Models") }
            item {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    allModels.forEach { asset ->
                        val isOn = asset.displayName in selectedModels
                        FilterChip(
                            selected = isOn,
                            onClick = {
                                if (isOn) selectedModels.remove(asset.displayName)
                                else selectedModels.add(asset.displayName)
                            },
                            label = { Text(asset.displayName) },
                        )
                    }
                }
            }
            item { SectionTitle("Backends") }
            item {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    NpuLabNative.Backend.values().forEach { b ->
                        val isOn = b in selectedBackends
                        FilterChip(
                            selected = isOn,
                            onClick = {
                                if (isOn) selectedBackends.remove(b)
                                else selectedBackends.add(b)
                            },
                            label = { Text(b.name) },
                        )
                    }
                }
            }
            item {
                if (ui.running) {
                    Button(
                        onClick = { vm.cancel() },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(20.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer,
                            contentColor = MaterialTheme.colorScheme.onErrorContainer,
                        ),
                    ) {
                        Icon(Icons.Outlined.Stop, contentDescription = null)
                        Spacer(Modifier.width(10.dp))
                        Text("Cancel  ·  ${ui.progressLabel}",
                            style = MaterialTheme.typography.titleMedium)
                    }
                } else {
                    Button(
                        onClick = {
                            val models = allModels.filter { it.displayName in selectedModels }
                            vm.run(models, selectedBackends.toList())
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(20.dp),
                        enabled = selectedModels.isNotEmpty() && selectedBackends.isNotEmpty(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.secondary,
                            contentColor = MaterialTheme.colorScheme.onSecondary,
                        ),
                    ) {
                        Icon(Icons.Outlined.PlayArrow, contentDescription = null)
                        Spacer(Modifier.width(10.dp))
                        Text("Run benchmark", style = MaterialTheme.typography.titleMedium)
                    }
                }
            }
            if (ui.running) {
                item {
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
            ui.error?.let {
                item {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = MaterialTheme.colorScheme.errorContainer,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(it,
                            modifier = Modifier.padding(16.dp),
                            color = MaterialTheme.colorScheme.onErrorContainer)
                    }
                }
            }
            if (ui.rows.isNotEmpty()) {
                item { SectionTitle("Results (${ui.rows.size})") }
                items(ui.rows) { row -> BenchmarkRowCard(row) }
            }
            item { VSpace(16) }
        }
    }
}

@Composable
private fun BenchmarkRowCard(row: BenchmarkRow) {
    val accent = when (row.backend) {
        NpuLabNative.Backend.HTP -> MaterialTheme.colorScheme.primary
        NpuLabNative.Backend.GPU -> MaterialTheme.colorScheme.tertiary
        NpuLabNative.Backend.CPU -> MaterialTheme.colorScheme.secondary
    }
    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .background(accent, shape = RoundedCornerShape(50)),
                )
                Spacer(Modifier.width(10.dp))
                Text(row.modelName, style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.weight(1f))
                AssistChip(
                    onClick = {},
                    label = { Text(row.backend.name) },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = accent.copy(alpha = 0.18f),
                        labelColor = accent,
                    ),
                )
            }
            if (row.error != null) {
                Spacer(Modifier.height(8.dp))
                Text(row.error, color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium)
            } else {
                Spacer(Modifier.height(8.dp))
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        StatLine("median ", row.medianUs)
                        StatLine("p95    ", row.p95Us)
                        StatLine("mean   ", row.meanUs)
                        StatLine("min    ", row.minUs)
                        StatLine("max    ", row.maxUs)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatLine(label: String, us: Long) {
    Text(
        text = "$label = ${"%.2f".format(us / 1000.0)} ms",
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}
