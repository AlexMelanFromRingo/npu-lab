package io.melan.npulab.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.CloudOff
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material.icons.outlined.DriveFolderUpload
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.melan.npulab.inference.ModelAsset
import io.melan.npulab.inference.ModelCatalog
import io.melan.npulab.inference.ModelCategory
import io.melan.npulab.ui.components.SectionTitle
import io.melan.npulab.ui.components.VSpace
import android.widget.Toast

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CatalogScreen(vm: CatalogViewModel = viewModel()) {
    val ui by vm.state.collectAsStateWithLifecycle()
    val ctx = LocalContext.current

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> if (uri != null) vm.importFromStorage(uri) }

    LaunchedEffect(ui.importMessage) {
        ui.importMessage?.let {
            if (it != "Importing…") {
                Toast.makeText(ctx, it, Toast.LENGTH_LONG).show()
                vm.clearImportMessage()
            }
        }
    }

    // Group catalog by category, keeping declaration order within each group.
    val grouped = remember { ModelCatalog.all.groupBy { it.category } }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        TopAppBar(
            title = { Text("Models") },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.surface,
            ),
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                VSpace(4)
                IntroBanner(
                    installedCount = ui.installed.size,
                    totalCount = ModelCatalog.all.size,
                    importing = ui.importMessage == "Importing…",
                    onImport = { picker.launch("*/*") },
                )
            }
            for (cat in CATEGORY_ORDER) {
                val assets = grouped[cat] ?: continue
                item(key = "hdr_$cat") { SectionTitle(categoryLabel(cat)) }
                items(assets, key = { it.id }) { asset ->
                    ModelCard(
                        asset = asset,
                        isInstalled = asset.id in ui.installed,
                        state = ui.states[asset.id] ?: CatalogViewModel.CardState.Idle,
                        onInstall = { vm.install(asset) },
                        onCancel = { vm.cancel(asset) },
                        onUninstall = { vm.uninstall(asset) },
                    )
                }
            }
            item { VSpace(16) }
        }
    }
}

private val CATEGORY_ORDER = listOf(
    ModelCategory.TEXT_TO_IMAGE,
    ModelCategory.SPEECH,
    ModelCategory.CLASSIFICATION,
    ModelCategory.DETECTION,
    ModelCategory.POSE,
    ModelCategory.DEPTH,
    ModelCategory.SEGMENTATION,
    ModelCategory.SUPER_RESOLUTION,
    ModelCategory.INPAINTING,
    ModelCategory.OTHER,
)

private fun categoryLabel(c: ModelCategory): String = when (c) {
    ModelCategory.TEXT_TO_IMAGE -> "Text → Image"
    ModelCategory.SPEECH -> "Speech"
    ModelCategory.CLASSIFICATION -> "Image classification"
    ModelCategory.DETECTION -> "Detection"
    ModelCategory.POSE -> "Pose"
    ModelCategory.DEPTH -> "Depth"
    ModelCategory.SEGMENTATION -> "Segmentation"
    ModelCategory.SUPER_RESOLUTION -> "Super-resolution"
    ModelCategory.INPAINTING -> "Inpainting"
    ModelCategory.OTHER -> "Other"
}

@Composable
private fun IntroBanner(
    installedCount: Int,
    totalCount: Int,
    importing: Boolean,
    onImport: () -> Unit,
) {
    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.CloudDownload, contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer)
                Spacer(Modifier.width(10.dp))
                Text(
                    "$installedCount / $totalCount installed",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(
                "Models download straight from the public Qualcomm S3 mirror, no AI Hub " +
                    "account needed. Or import your own .bin / .dlc / .zip from this phone.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
            Spacer(Modifier.height(12.dp))
            FilledTonalButton(
                onClick = onImport,
                enabled = !importing,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Outlined.DriveFolderUpload, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(if (importing) "Importing…" else "Import from storage")
            }
        }
    }
}

@Composable
private fun ModelCard(
    asset: ModelAsset,
    isInstalled: Boolean,
    state: CatalogViewModel.CardState,
    onInstall: () -> Unit,
    onCancel: () -> Unit,
    onUninstall: () -> Unit,
) {
    val busy = state is CatalogViewModel.CardState.Downloading ||
        state is CatalogViewModel.CardState.Extracting
    val source = asset.installSource

    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isInstalled)
                MaterialTheme.colorScheme.secondaryContainer
            else
                MaterialTheme.colorScheme.surface,
        ),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusDot(isInstalled = isInstalled, busy = busy)
                Spacer(Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(asset.displayName, style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = asset.description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (source != null) {
                    AssistChip(
                        onClick = {},
                        label = { Text("~${source.approxMib} MiB") },
                        colors = AssistChipDefaults.assistChipColors(
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                            labelColor = MaterialTheme.colorScheme.onTertiaryContainer,
                        ),
                    )
                }
            }

            when (state) {
                is CatalogViewModel.CardState.Downloading -> {
                    Spacer(Modifier.height(12.dp))
                    val progress = if (state.bytesTotal > 0)
                        (state.bytesDone.toFloat() / state.bytesTotal).coerceIn(0f, 1f)
                    else null
                    if (progress != null) {
                        LinearProgressIndicator(
                            progress = { progress },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    } else {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = "${state.phase}  ·  " +
                            "${state.bytesDone / (1024 * 1024)} / " +
                            "${state.bytesTotal / (1024 * 1024)} MiB",
                        style = MaterialTheme.typography.labelMedium,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                is CatalogViewModel.CardState.Extracting -> {
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Spacer(Modifier.height(6.dp))
                    Text("Extracting…", style = MaterialTheme.typography.labelMedium,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                is CatalogViewModel.CardState.Failed -> {
                    Spacer(Modifier.height(12.dp))
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.errorContainer,
                    ) {
                        Row(modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Outlined.ErrorOutline, contentDescription = null,
                                tint = MaterialTheme.colorScheme.onErrorContainer)
                            Spacer(Modifier.width(8.dp))
                            Text(state.message,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onErrorContainer)
                        }
                    }
                }
                else -> { /* nothing extra */ }
            }

            Spacer(Modifier.height(12.dp))
            Row(modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {

                if (source == null) {
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        modifier = Modifier.weight(1f),
                    ) {
                        Row(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Outlined.CloudOff, contentDescription = null,
                                tint = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "Not on public S3 — needs an AI Hub account",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                } else if (busy) {
                    Button(
                        onClick = onCancel,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer,
                            contentColor = MaterialTheme.colorScheme.onErrorContainer,
                        ),
                    ) {
                        Icon(Icons.Outlined.Stop, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Cancel", style = MaterialTheme.typography.titleMedium)
                    }
                } else if (isInstalled) {
                    Button(
                        onClick = onInstall,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.tertiary,
                            contentColor = MaterialTheme.colorScheme.onTertiary,
                        ),
                    ) {
                        Icon(Icons.Outlined.CheckCircle, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Reinstall")
                    }
                    FilledTonalIconButton(
                        onClick = onUninstall,
                        modifier = Modifier.size(48.dp),
                        colors = IconButtonDefaults.filledTonalIconButtonColors(
                            contentColor = MaterialTheme.colorScheme.error,
                        ),
                    ) {
                        Icon(Icons.Outlined.DeleteOutline, contentDescription = "Delete")
                    }
                } else {
                    Button(
                        onClick = onInstall,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary,
                            contentColor = MaterialTheme.colorScheme.onPrimary,
                        ),
                    ) {
                        Icon(Icons.Outlined.CloudDownload, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Install", style = MaterialTheme.typography.titleMedium)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusDot(isInstalled: Boolean, busy: Boolean) {
    val color = when {
        busy -> MaterialTheme.colorScheme.primary
        isInstalled -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.outline
    }
    Box(
        modifier = Modifier
            .size(12.dp)
            .background(color, shape = RoundedCornerShape(50)),
    )
}
