package io.melan.npulab.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Casino
import androidx.compose.material.icons.outlined.HourglassEmpty
import androidx.compose.material.icons.outlined.SaveAlt
import androidx.compose.material.icons.outlined.Timer
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace
import io.melan.npulab.ui.util.GalleryWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GenerateScreen(vm: GenerateViewModel = viewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }
    val coroutineScope = rememberCoroutineScope()
    val context = LocalContext.current

    var prompt by remember { mutableStateOf("a moody cyberpunk fox in neon rain, cinematic, sharp focus") }
    var negPrompt by remember { mutableStateOf("blurry, low quality, distorted") }
    var steps by remember { mutableStateOf(20f) }
    var cfg by remember { mutableStateOf(7.5f) }
    var seed by remember { mutableLongStateOf(-1L) }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = { Text("Generate") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            )
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                VSpace(4)
                ImagePanel(
                    state = state,
                    onSave = { bitmap ->
                        coroutineScope.launch {
                            val result = withContext(Dispatchers.IO) {
                                GalleryWriter.save(context, bitmap, prompt)
                            }
                            snackbarHostState.showSnackbar(
                                if (result.isSuccess) "Saved to Pictures/NpuLab/"
                                else "Save failed: ${result.exceptionOrNull()?.message}"
                            )
                        }
                    },
                )
            }
            item {
                SectionTitle("Prompt")
                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    label = { Text("Prompt") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 2,
                    maxLines = 4,
                    shape = RoundedCornerShape(16.dp),
                    enabled = state !is GenerateViewModel.State.Generating &&
                              state !is GenerateViewModel.State.Loading,
                )
            }
            item {
                OutlinedTextField(
                    value = negPrompt,
                    onValueChange = { negPrompt = it },
                    label = { Text("Negative prompt") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 1,
                    maxLines = 3,
                    shape = RoundedCornerShape(16.dp),
                    enabled = state !is GenerateViewModel.State.Generating &&
                              state !is GenerateViewModel.State.Loading,
                )
            }
            item {
                SectionTitle("Steps  •  ${steps.toInt()}")
                Slider(
                    value = steps,
                    onValueChange = { steps = it },
                    valueRange = 4f..30f,
                    steps = 25,
                )
            }
            item {
                SectionTitle("CFG  •  ${(cfg * 10).roundToInt() / 10f}")
                Slider(
                    value = cfg,
                    onValueChange = { cfg = it },
                    valueRange = 1f..15f,
                    steps = 13,
                )
            }
            item {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = if (seed < 0) "random" else seed.toString(),
                        onValueChange = { v ->
                            seed = if (v.equals("random", ignoreCase = true)) -1L
                            else v.toLongOrNull() ?: -1L
                        },
                        label = { Text("Seed") },
                        modifier = Modifier.weight(1f),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
                        shape = RoundedCornerShape(16.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    FilledTonalIconButton(onClick = { seed = -1L }) {
                        Icon(Icons.Outlined.Casino, contentDescription = "Random seed")
                    }
                }
            }
            if (state is GenerateViewModel.State.Failed) {
                item {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = MaterialTheme.colorScheme.errorContainer,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            (state as GenerateViewModel.State.Failed).message,
                            modifier = Modifier.padding(16.dp),
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                    }
                }
            }
            item {
                val busy = state is GenerateViewModel.State.Generating ||
                           state is GenerateViewModel.State.Loading
                Button(
                    onClick = {
                        vm.generate(
                            prompt = prompt,
                            negativePrompt = negPrompt,
                            numSteps = steps.toInt(),
                            cfgScale = cfg,
                            seed = seed,
                        )
                    },
                    enabled = !busy && prompt.isNotBlank(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(20.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    ),
                ) {
                    if (busy) {
                        CircularProgressIndicator(
                            modifier = Modifier.width(20.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp,
                        )
                    } else {
                        Icon(Icons.Outlined.AutoAwesome, contentDescription = null)
                    }
                    Spacer(Modifier.width(12.dp))
                    Text(if (busy) "Generating…" else "Generate",
                        style = MaterialTheme.typography.titleMedium)
                }
                VSpace(16)
            }
        }
    }
}

@Composable
private fun ImagePanel(
    state: GenerateViewModel.State,
    onSave: (android.graphics.Bitmap) -> Unit,
) {
    Card(
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f),
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            when (state) {
                is GenerateViewModel.State.Done -> {
                    Image(
                        bitmap = state.bitmap.asImageBitmap(),
                        contentDescription = "Generated image",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(24.dp)),
                    )
                    Row(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Surface(
                            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                            shape = RoundedCornerShape(12.dp),
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Icon(Icons.Outlined.HourglassEmpty, contentDescription = null,
                                    tint = MaterialTheme.colorScheme.tertiary)
                                Spacer(Modifier.width(6.dp))
                                Text(
                                    "${state.totalMs} ms · ${state.perStepMedianMs}ms/step",
                                    fontFamily = FontFamily.Monospace,
                                    style = MaterialTheme.typography.labelMedium,
                                )
                            }
                        }
                        FilledTonalIconButton(
                            onClick = { onSave(state.bitmap) },
                            modifier = Modifier.size(40.dp),
                            colors = IconButtonDefaults.filledTonalIconButtonColors(
                                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                                contentColor = MaterialTheme.colorScheme.primary,
                            ),
                        ) {
                            Icon(Icons.Outlined.SaveAlt, contentDescription = "Save to gallery")
                        }
                    }
                }
                is GenerateViewModel.State.Generating -> {
                    PlaceholderIcon()
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .align(Alignment.BottomCenter)
                            .background(MaterialTheme.colorScheme.scrim.copy(alpha = 0.6f))
                            .padding(12.dp),
                    ) {
                        LinearProgressIndicator(
                            progress = { state.step.toFloat() / state.totalSteps },
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.primary,
                        )
                        Spacer(Modifier.height(6.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Outlined.Timer, contentDescription = null,
                                tint = MaterialTheme.colorScheme.onPrimary)
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "step ${state.step}/${state.totalSteps} · ${state.lastStepMs} ms",
                                color = MaterialTheme.colorScheme.onPrimary,
                                style = MaterialTheme.typography.labelMedium,
                                fontFamily = FontFamily.Monospace,
                            )
                        }
                    }
                }
                is GenerateViewModel.State.Loading -> {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.height(8.dp))
                        Text(state.message,
                            style = MaterialTheme.typography.labelMedium,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                else -> PlaceholderIcon()
            }
        }
    }
}

@Composable
private fun PlaceholderIcon() {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(
            Icons.Outlined.AutoAwesome,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "tap Generate to draw something",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontFamily = FontFamily.Monospace,
        )
    }
}
