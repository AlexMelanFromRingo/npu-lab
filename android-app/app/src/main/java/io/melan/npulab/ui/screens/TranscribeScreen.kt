package io.melan.npulab.ui.screens

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.GraphicEq
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.inference.WhisperVariant
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace

/**
 * Speech-to-text on the Hexagon NPU: up to 2 min of mic audio, split into
 * Whisper's 30 s windows, each run through mel → encoder → autoregressive
 * decoder with the forced prompt [SOT, lang, transcribe, notimestamps].
 * Model flavor (Base / Small / Large-v3 Turbo) and language are selectable.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun TranscribeScreen(vm: TranscribeViewModel = viewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val selectedLang by vm.language.collectAsStateWithLifecycle()
    val selectedVariant by vm.variant.collectAsStateWithLifecycle()
    val ctx = LocalContext.current

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) vm.startRecording() }

    fun startWithPermission() {
        val has = ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (has) vm.startRecording() else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        TopAppBar(
            title = { Text("Speech → Text") },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            VSpace(4)
            SectionTitle("Model")
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Re-checked on every recomposition — cheap file existence test.
                WhisperVariant.ALL.forEach { v ->
                    val installed = remember(v, state) { vm.isInstalled(v) }
                    FilterChip(
                        selected = selectedVariant == v,
                        onClick = { vm.selectVariant(v) },
                        label = {
                            Text(if (installed) v.displayName else "${v.displayName} ↓")
                        },
                    )
                }
            }
            (WhisperVariant.ALL.firstOrNull { it == selectedVariant && !vm.isInstalled(it) })?.let {
                VSpace(6)
                Text(
                    "Whisper ${it.displayName} is not installed — grab it on the Models tab.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            VSpace(12)
            SectionTitle("Language")
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val langs = listOf(
                    null to "Auto", "ru" to "Русский", "en" to "English",
                    "uk" to "Українська", "de" to "Deutsch",
                )
                langs.forEach { (code, label) ->
                    FilterChip(
                        selected = selectedLang == code,
                        onClick = { vm.language.value = code },
                        label = { Text(label) },
                    )
                }
            }
            VSpace(12)
            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    when (val s = state) {
                        is TranscribeViewModel.State.Recording -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Outlined.GraphicEq,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Spacer(Modifier.width(10.dp))
                                Text(
                                    "Recording… ${s.seconds} s / ${TranscribeViewModel.MAX_RECORD_SECONDS} s",
                                    style = MaterialTheme.typography.titleMedium,
                                )
                            }
                            VSpace(12)
                            LinearProgressIndicator(
                                progress = { (s.level * 4f).coerceIn(0f, 1f) },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                        is TranscribeViewModel.State.Loading -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(modifier = Modifier.size(22.dp))
                                Spacer(Modifier.width(12.dp))
                                Text(s.message, style = MaterialTheme.typography.bodyLarge)
                            }
                        }
                        is TranscribeViewModel.State.Transcribing -> {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(modifier = Modifier.size(22.dp))
                                Spacer(Modifier.width(12.dp))
                                Text(
                                    if (s.totalChunks > 1)
                                        "Transcribing on NPU… window ${s.chunk}/${s.totalChunks}"
                                    else "Transcribing on NPU…",
                                    style = MaterialTheme.typography.bodyLarge,
                                )
                            }
                            if (s.partial.isNotBlank()) {
                                VSpace(12)
                                Text(
                                    s.partial,
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                        is TranscribeViewModel.State.Done -> {
                            SelectionContainer {
                                Text(s.text, style = MaterialTheme.typography.headlineSmall)
                            }
                            VSpace(16)
                            Surface(
                                shape = RoundedCornerShape(12.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant,
                            ) {
                                Text(
                                    "lang ${s.language ?: "?"} · encoder ${s.encoderMs} ms · " +
                                        "decode ${s.decodeMs} ms " +
                                        "(${"%.1f".format(s.tokensPerSecond)} tok/s) · " +
                                        "mel ${s.melMs} ms · ${s.tokens} tokens",
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                    fontFamily = FontFamily.Monospace,
                                    style = MaterialTheme.typography.labelMedium,
                                )
                            }
                        }
                        is TranscribeViewModel.State.Failed -> {
                            Text(
                                s.message,
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                        TranscribeViewModel.State.Idle -> {
                            Text(
                                "Tap the button and speak — up to 2 minutes; longer " +
                                    "recordings are transcribed in 30-second windows. " +
                                    "Pinning a language above is more reliable than Auto " +
                                    "on short phrases.",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
            VSpace(16)
            val recording = state is TranscribeViewModel.State.Recording
            val busy = state is TranscribeViewModel.State.Loading ||
                state is TranscribeViewModel.State.Transcribing
            Button(
                onClick = { if (recording) vm.stopAndTranscribe() else startWithPermission() },
                enabled = !busy,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp),
                shape = RoundedCornerShape(22.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (recording) MaterialTheme.colorScheme.errorContainer
                    else MaterialTheme.colorScheme.primary,
                    contentColor = if (recording) MaterialTheme.colorScheme.onErrorContainer
                    else MaterialTheme.colorScheme.onPrimary,
                ),
            ) {
                Icon(
                    if (recording) Icons.Outlined.Stop else Icons.Outlined.Mic,
                    contentDescription = null,
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    if (recording) "Stop & transcribe" else "Record",
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            VSpace(24)
        }
    }
}
