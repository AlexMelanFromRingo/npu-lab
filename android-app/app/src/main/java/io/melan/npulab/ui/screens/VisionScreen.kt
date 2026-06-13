package io.melan.npulab.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace
import io.melan.npulab.vision.VisionPipeline

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun VisionScreen(vm: VisionViewModel = viewModel(), embedded: Boolean = false) {
    val state by vm.state.collectAsStateWithLifecycle()
    val input by vm.input.collectAsStateWithLifecycle()
    val selected by vm.model.collectAsStateWithLifecycle()
    val ctx = LocalContext.current

    // Recompute available models whenever the screen recomposes (cheap file check).
    val models by produceState(initialValue = emptyList<ModelAsset>(), state) {
        value = vm.availableModels()
    }

    val gallery = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> if (uri != null) vm.loadInputFromUri(uri) }
    val camera = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bmp -> if (bmp != null) vm.setInput(bmp) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        if (!embedded) TopAppBar(
            title = { Text("Vision") },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )
        if (models.isEmpty()) {
            Column(modifier = Modifier.padding(24.dp)) {
                Text(
                    "No vision models installed yet.\n\nOpen the Models tab and install a " +
                        "classification (MobileNet, ResNet…), depth (MiDaS), segmentation " +
                        "(DeepLabV3+) or super-resolution model — they are small DLCs.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            return@Column
        }

        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item { VSpace(4); SectionTitle("Model") }
            item {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    models.forEach { m ->
                        FilterChip(
                            selected = selected?.id == m.id,
                            onClick = { vm.selectModel(m) },
                            label = { Text(m.displayName) },
                        )
                    }
                }
            }

            item { SectionTitle("Input") }
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilledTonalButton(onClick = { gallery.launch("image/*") },
                        modifier = Modifier.weight(1f)) {
                        Icon(Icons.Outlined.PhotoLibrary, contentDescription = null)
                        Spacer(Modifier.width(8.dp)); Text("Gallery")
                    }
                    FilledTonalButton(onClick = { camera.launch(null) },
                        modifier = Modifier.weight(1f)) {
                        Icon(Icons.Outlined.PhotoCamera, contentDescription = null)
                        Spacer(Modifier.width(8.dp)); Text("Camera")
                    }
                }
            }
            input?.let { bmp ->
                item {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = "Input",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(bmp.width.toFloat() / bmp.height)
                            .clip(RoundedCornerShape(16.dp)),
                    )
                }
            }

            item {
                val busy = state is VisionViewModel.State.Loading
                Button(
                    onClick = { vm.run() },
                    enabled = !busy && selected != null && input != null,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(20.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    ),
                ) {
                    Icon(Icons.Outlined.PlayArrow, contentDescription = null)
                    Spacer(Modifier.width(10.dp))
                    Text("Run on NPU", style = MaterialTheme.typography.titleMedium)
                }
            }

            item { ResultView(state) }
            item { VSpace(16) }
        }
    }
}

@Composable
private fun ResultView(state: VisionViewModel.State) {
    when (state) {
        is VisionViewModel.State.Loading -> Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Row(modifier = Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(22.dp))
                Spacer(Modifier.width(12.dp))
                Text(state.message, style = MaterialTheme.typography.bodyLarge)
            }
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
        is VisionViewModel.State.Failed -> Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.errorContainer,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(state.message, modifier = Modifier.padding(16.dp),
                color = MaterialTheme.colorScheme.onErrorContainer)
        }
        is VisionViewModel.State.Done -> ResultCard(state.result)
        VisionViewModel.State.Idle -> {}
    }
}

@Composable
private fun ResultCard(result: VisionPipeline.Result) {
    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            when (result) {
                is VisionPipeline.Result.Labels -> {
                    result.top.forEachIndexed { i, (label, prob) ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                label,
                                style = if (i == 0) MaterialTheme.typography.titleMedium
                                else MaterialTheme.typography.bodyLarge,
                                modifier = Modifier.weight(1f),
                            )
                            Text("${"%.1f".format(prob * 100)}%",
                                fontFamily = FontFamily.Monospace,
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary)
                        }
                        LinearProgressIndicator(
                            progress = { prob.coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }
                is VisionPipeline.Result.Image -> {
                    Image(
                        bitmap = result.bitmap.asImageBitmap(),
                        contentDescription = result.caption,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(result.bitmap.width.toFloat() / result.bitmap.height)
                            .clip(RoundedCornerShape(12.dp)),
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(result.caption, style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                is VisionPipeline.Result.Raw -> SelectionContainer {
                    Text(result.text, style = MaterialTheme.typography.bodyMedium)
                }
            }
            Spacer(Modifier.height(10.dp))
            Surface(shape = RoundedCornerShape(10.dp),
                color = MaterialTheme.colorScheme.surfaceVariant) {
                Text("inference ${result.inferenceMs} ms on HTP",
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    fontFamily = FontFamily.Monospace,
                    style = MaterialTheme.typography.labelMedium)
            }
        }
    }
}
