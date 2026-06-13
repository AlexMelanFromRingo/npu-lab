package io.melan.npulab.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.Visibility
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Hosts the three "creative" tools — Generate (SD), Speech (Whisper) and
 * Vision — under one bottom-nav destination with a top segmented switcher.
 * Collapses three tabs into one so the bottom bar has breathing room.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StudioScreen() {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    val tabs = listOf(
        Triple("Generate", Icons.Outlined.AutoAwesome, "Generate"),
        Triple("Speech", Icons.Outlined.Mic, "Speech"),
        Triple("Vision", Icons.Outlined.Visibility, "Vision"),
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        TopAppBar(
            title = { Text("Studio") },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.surface,
            ),
        )
        SingleChoiceSegmentedButtonRow(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            tabs.forEachIndexed { i, (label, icon, _) ->
                SegmentedButton(
                    selected = tab == i,
                    onClick = { tab = i },
                    shape = SegmentedButtonDefaults.itemShape(i, tabs.size),
                    icon = {
                        SegmentedButtonDefaults.Icon(active = tab == i) {
                            Icon(icon, contentDescription = null,
                                modifier = Modifier.padding(0.dp))
                        }
                    },
                    label = { Text(label) },
                )
            }
        }
        when (tab) {
            0 -> GenerateScreen(embedded = true)
            1 -> TranscribeScreen(embedded = true)
            else -> VisionScreen(embedded = true)
        }
    }
}
