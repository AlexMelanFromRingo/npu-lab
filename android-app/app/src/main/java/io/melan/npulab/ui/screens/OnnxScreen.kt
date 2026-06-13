package io.melan.npulab.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.DriveFolderUpload
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.inference.OrtNpuRunner
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace

/**
 * "Any ONNX → NPU" — import a standard .onnx and run it on the Hexagon NPU via
 * ONNX Runtime + the QNN EP, on-device. The second engine, next to the
 * hand-built QNN pipeline.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun OnnxScreen(vm: OnnxViewModel = viewModel(), embedded: Boolean = false) {
    val state by vm.state.collectAsStateWithLifecycle()
    val models by vm.models.collectAsStateWithLifecycle()
    val selected by vm.selected.collectAsStateWithLifecycle()
    val backend by vm.backend.collectAsStateWithLifecycle()

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> if (uri != null) vm.importOnnx(uri) }

    Column(
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
    ) {
        if (!embedded) TopAppBar(
            title = { Text("ONNX → NPU") },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                VSpace(4)
                Card(
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            "Run any standard .onnx on the Hexagon NPU via ONNX Runtime " +
                                "+ QNN EP — compiled on-device, no PC step. Unsupported ops " +
                                "fall back to CPU automatically.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                        VSpace(12)
                        FilledTonalButton(
                            onClick = { picker.launch("*/*") },
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Outlined.DriveFolderUpload, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("Import .onnx from storage")
                        }
                    }
                }
            }

            if (models.isNotEmpty()) {
                item { SectionTitle("Model") }
                item {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        models.forEach { f ->
                            FilterChip(
                                selected = selected?.absolutePath == f.absolutePath,
                                onClick = { vm.select(f) },
                                label = { Text(f.name.removeSuffix(".onnx")) },
                            )
                        }
                    }
                }
                item { SectionTitle("Backend") }
                item {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OrtNpuRunner.Backend.values().forEach { b ->
                            FilterChip(
                                selected = backend == b,
                                onClick = { vm.setBackend(b) },
                                label = { Text(b.label) },
                            )
                        }
                    }
                }
                item {
                    val busy = state is OnnxViewModel.State.Running
                    Button(
                        onClick = { vm.run() },
                        enabled = !busy && selected != null,
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                        shape = RoundedCornerShape(20.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary,
                            contentColor = MaterialTheme.colorScheme.onPrimary,
                        ),
                    ) {
                        Icon(Icons.Outlined.PlayArrow, contentDescription = null)
                        Spacer(Modifier.width(10.dp))
                        Text("Compile & run on NPU", style = MaterialTheme.typography.titleMedium)
                    }
                }
            }

            item { ResultBlock(state) }
            item { VSpace(16) }
        }
    }
}

@Composable
private fun ResultBlock(state: OnnxViewModel.State) {
    when (state) {
        is OnnxViewModel.State.Running -> Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(12.dp))
                Text(state.message, style = MaterialTheme.typography.bodyLarge)
            }
        }
        is OnnxViewModel.State.Failed -> Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.errorContainer,
            modifier = Modifier.fillMaxWidth(),
        ) {
            SelectionContainer {
                Text(state.message, modifier = Modifier.padding(16.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }
        is OnnxViewModel.State.Done -> {
            val r = state.result
            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text("${state.backend.label}", style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary)
                    VSpace(8)
                    Surface(color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(12.dp)) {
                        Column(Modifier.padding(12.dp)) {
                            mono("median = ${"%.2f".format(r.medianMs)} ms")
                            mono("p95    = ${"%.2f".format(r.p95Ms)} ms")
                            mono("mean   = ${"%.2f".format(r.meanMs)} ms")
                            mono("min    = ${"%.2f".format(r.minMs)} ms")
                            mono("max    = ${"%.2f".format(r.maxMs)} ms")
                            mono("load   = ${r.loadMs} ms (compile+session)")
                        }
                    }
                    VSpace(8)
                    r.inputs.forEach { mono("IN  ${it.name}: ${it.type} ${it.shape}") }
                    r.outputs.forEach { mono("OUT ${it.name}: ${it.type} ${it.shape}") }
                }
            }
        }
        OnnxViewModel.State.Idle -> {}
    }
}

@Composable
private fun mono(text: String) = Text(
    text,
    fontFamily = FontFamily.Monospace,
    style = MaterialTheme.typography.labelMedium,
    color = MaterialTheme.colorScheme.onSurfaceVariant,
)
