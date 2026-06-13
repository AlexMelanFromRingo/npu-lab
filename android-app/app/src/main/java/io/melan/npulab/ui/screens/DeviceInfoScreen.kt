package io.melan.npulab.ui.screens

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
import androidx.compose.material.icons.outlined.Bolt
import androidx.compose.material.icons.outlined.BugReport
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.SdStorage
import androidx.compose.material.icons.outlined.Numbers
import androidx.compose.material.icons.outlined.Speed
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.melan.npulab.device.DeviceInfo
import io.melan.npulab.device.DeviceInfoCollector
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelStore
import io.melan.npulab.inference.NpuLabNative
import io.melan.npulab.inference.QnnRuntimeLibs
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.StatChip
import io.melan.npulab.ui.components.VSpace
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun DeviceInfoScreen() {
    val ctx = LocalContext.current
    val info by produceState<DeviceInfo?>(initialValue = null) {
        value = withContext(Dispatchers.IO) { DeviceInfoCollector.collect(ctx) }
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        TopAppBar(
            title = { Text("Device") },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
        )

        val data = info
        if (data == null) {
            Row(modifier = Modifier.padding(24.dp)) { Text("Reading device info…") }
            return
        }
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item { VSpace(4); SectionTitle("Device") }
            item {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Text(data.deviceModel, style = MaterialTheme.typography.headlineMedium)
                        Spacer(Modifier.height(4.dp))
                        Text(data.androidVersion, style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(12.dp))
                        Text(
                            text = "${data.socManufacturer} • ${data.socName}",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.primary,
                            fontFamily = FontFamily.Monospace,
                        )
                        if (data.htpArchGuess > 0) {
                            Text(
                                text = "Hexagon HTP v${data.htpArchGuess}",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.secondary,
                            )
                        } else {
                            Text(
                                text = "HTP arch not recognized — possibly not a Snapdragon SoC",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        val arches = remember { QnnRuntimeLibs.bundledArches(ctx) }
                        val supported = data.htpArchGuess <= 0 || data.htpArchGuess in arches
                        Text(
                            text = "Bundled HTP arches: " +
                                (arches.joinToString(", ") { "v$it" }.ifEmpty { "none" }),
                            style = MaterialTheme.typography.labelMedium,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            text = if (supported)
                                "✓ This chip's NPU is supported by this build"
                            else
                                "⚠ No skel for v${data.htpArchGuess} — rebuild with " +
                                    "HTP_ARCHES=v${data.htpArchGuess} (DLC models won't run on NPU)",
                            style = MaterialTheme.typography.labelMedium,
                            color = if (supported) MaterialTheme.colorScheme.tertiary
                            else MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }
            item { SectionTitle("Resources") }
            item {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    StatChip("Total RAM", "${data.totalRamMb} MB", icon = Icons.Outlined.Memory)
                    StatChip("Free RAM", "${data.availRamMb} MB", icon = Icons.Outlined.SdStorage)
                    StatChip("CPU cores", "${data.cpuCores}", icon = Icons.Outlined.Numbers,
                        accent = MaterialTheme.colorScheme.secondary)
                    StatChip("CPU max", "${"%.2f".format(data.cpuMaxGhz)} GHz",
                        icon = Icons.Outlined.Speed,
                        accent = MaterialTheme.colorScheme.tertiary)
                }
            }
            item { SectionTitle("ABIs") }
            item {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    data.supportedAbis.forEach { abi ->
                        Surface(
                            shape = RoundedCornerShape(12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant,
                        ) {
                            Text(
                                abi,
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                fontFamily = FontFamily.Monospace,
                                style = MaterialTheme.typography.labelMedium,
                            )
                        }
                    }
                }
            }
            item { SectionTitle("Native libs in APK") }
            item {
                Card(
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        if (data.qnnLibrariesPresent.isEmpty()) {
                            Text(
                                text = "No libQnn*.so found. " +
                                        "Copy them into app/src/main/jniLibs/arm64-v8a/ — see docs/04-qnn-sdk-setup.md",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.error,
                            )
                        } else {
                            data.qnnLibrariesPresent.forEach { name ->
                                Row(modifier = Modifier.padding(vertical = 2.dp)) {
                                    Text(
                                        text = "✓",
                                        color = MaterialTheme.colorScheme.tertiary,
                                        fontFamily = FontFamily.Monospace,
                                    )
                                    Spacer(Modifier.padding(horizontal = 6.dp))
                                    Text(
                                        text = name,
                                        fontFamily = FontFamily.Monospace,
                                        style = MaterialTheme.typography.bodyMedium,
                                    )
                                }
                            }
                        }
                    }
                }
            }
            item { SectionTitle("On-device compile") }
            item { OnDeviceCompileCard() }
            item { SectionTitle("NPU self-test") }
            item { NpuSelfTestCard() }
            data.runtimeJson?.let { json ->
                item { SectionTitle("Backend reports") }
                item {
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            text = json,
                            modifier = Modifier.padding(16.dp),
                            fontFamily = FontFamily.Monospace,
                            style = MaterialTheme.typography.labelMedium,
                        )
                    }
                }
            }
            item { VSpace(16) }
        }
    }
}

private sealed interface CompileState {
    data object Idle : CompileState
    data class Running(val done: Int, val total: Int, val name: String) : CompileState
    data class Done(val results: List<io.melan.npulab.inference.OnDeviceCompiler.CompileResult>) : CompileState
}

/**
 * On-device compiler: composes every installed DLC for this exact chip and
 * serializes it to a fast-loading context binary in models/custom/, where the
 * Benchmark/Vision tabs pick it up. Shows per-model compile time — the cost you
 * pay once so later loads skip the prepare.
 */
@Composable
private fun OnDeviceCompileCard() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<CompileState>(CompileState.Idle) }
    var cacheBytes by remember { mutableStateOf(0L) }
    val dlcCount = remember { io.melan.npulab.inference.OnDeviceCompiler.installedDlcs(ctx).size }
    LaunchedEffect(state) {
        cacheBytes = withContext(Dispatchers.IO) {
            io.melan.npulab.inference.OnDeviceCompiler.cacheBytes(ctx)
        }
    }

    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Compiles your installed DLC models for this exact chip and " +
                    "saves them as fast-loading context binaries. The compiled " +
                    "models appear in Benchmark / Vision. ONNX→DLC stays on PC; " +
                    "this is the on-device half.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            VSpace(8)
            Text(
                "$dlcCount installed DLC model(s) · cache ${cacheBytes / (1024 * 1024)} MB " +
                    "· target ${io.melan.npulab.inference.OnDeviceCompiler.archTag()}",
                style = MaterialTheme.typography.labelMedium,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            VSpace(12)
            when (val s = state) {
                is CompileState.Running -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(12.dp))
                        Text("Compiling ${s.done}/${s.total}: ${s.name}…",
                            style = MaterialTheme.typography.bodyMedium)
                    }
                }
                else -> {
                    Row {
                        FilledTonalButton(
                            onClick = {
                                scope.launch {
                                    val results = withContext(Dispatchers.IO) {
                                        io.melan.npulab.inference.OnDeviceCompiler.compileAll(ctx) { i, n, name ->
                                            state = CompileState.Running(i, n, name)
                                        }
                                    }
                                    state = CompileState.Done(results)
                                }
                            },
                            enabled = dlcCount > 0,
                        ) {
                            Icon(Icons.Outlined.Bolt, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text(if (s is CompileState.Done) "Recompile" else "Compile for this chip")
                        }
                        if (cacheBytes > 0) {
                            Spacer(Modifier.width(4.dp))
                            TextButton(onClick = {
                                io.melan.npulab.inference.OnDeviceCompiler.clearCache(ctx)
                                state = CompileState.Idle
                                cacheBytes = 0
                            }) { Text("Clear") }
                        }
                    }
                }
            }
            (state as? CompileState.Done)?.let { done ->
                VSpace(10)
                done.results.forEach { r ->
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                        Text(if (r.ok) "✓" else "✗",
                            color = if (r.ok) MaterialTheme.colorScheme.tertiary
                            else MaterialTheme.colorScheme.error,
                            fontFamily = FontFamily.Monospace)
                        Spacer(Modifier.width(8.dp))
                        Text(r.name, modifier = Modifier.weight(1f),
                            style = MaterialTheme.typography.bodyMedium)
                        Text(
                            if (r.ok) "${r.composeMs} ms · ${r.outBytes / (1024 * 1024)} MB"
                            else (r.error ?: "failed"),
                            style = MaterialTheme.typography.labelMedium,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

private sealed interface SelfTestState {
    data object Idle : SelfTestState
    data object Running : SelfTestState
    data class Done(val report: String) : SelfTestState
}

/**
 * One-tap NPU diagnostics. Runs the native bring-up matrix (unsigned-PD /
 * default / SOC+ARCH deviceCreate, each × contextCreateFromBinary) against the
 * smallest installed model and shows the full report with a copy button — so
 * the root cause can be shared without adb access.
 */
@Composable
private fun NpuSelfTestCard() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val clipboard = LocalClipboardManager.current
    var state by remember { mutableStateOf<SelfTestState>(SelfTestState.Idle) }

    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Runs every HTP init flavor and tries to load a model under " +
                    "each one, capturing the full internal QNN log. " +
                    "The report is copyable — no adb needed.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            VSpace(12)
            when (val s = state) {
                SelfTestState.Idle, is SelfTestState.Done -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        FilledTonalButton(onClick = {
                            state = SelfTestState.Running
                            scope.launch {
                                val report = withContext(Dispatchers.IO) {
                                    runCatching {
                                        NpuLabNative.npuSelfTest(
                                            QnnRuntimeLibs.runtimeDir(ctx),
                                            smallestInstalledModel(ctx) ?: "",
                                        )
                                    }.getOrElse { t ->
                                        "self-test crashed: ${t.message ?: t::class.java.name}"
                                    }
                                }
                                state = SelfTestState.Done(report)
                            }
                        }) {
                            Icon(Icons.Outlined.BugReport, contentDescription = null)
                            Spacer(Modifier.padding(horizontal = 4.dp))
                            Text(if (s is SelfTestState.Done) "Run again" else "Run self-test")
                        }
                        if (s is SelfTestState.Done) {
                            Spacer(Modifier.padding(horizontal = 4.dp))
                            TextButton(onClick = {
                                clipboard.setText(AnnotatedString(s.report))
                            }) {
                                Icon(Icons.Outlined.ContentCopy, contentDescription = null)
                                Spacer(Modifier.padding(horizontal = 4.dp))
                                Text("Copy")
                            }
                        }
                    }
                }
                SelfTestState.Running -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.padding(end = 12.dp))
                        Text(
                            "Running the init matrix… (~10–30 s)",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
            (state as? SelfTestState.Done)?.let { done ->
                VSpace(12)
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    SelectionContainer {
                        Text(
                            text = done.report,
                            modifier = Modifier.padding(12.dp),
                            fontFamily = FontFamily.Monospace,
                            style = MaterialTheme.typography.labelSmall,
                            fontSize = 10.sp,
                            lineHeight = 14.sp,
                        )
                    }
                }
            }
        }
    }
}

/** Smallest installed .bin — fastest to (de)serialize during the self-test. */
private fun smallestInstalledModel(ctx: android.content.Context): String? {
    val store = ModelStore(ctx)
    return ModelCatalog.all
        .flatMap { it.expectedFiles }
        .filter { it.endsWith(".bin") }
        .map { store.pathOf(it) }
        .filter { it.isFile && it.length() > 0 }
        .minByOrNull { it.length() }
        ?.absolutePath
}
