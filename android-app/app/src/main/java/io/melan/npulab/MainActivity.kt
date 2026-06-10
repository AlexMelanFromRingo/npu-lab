package io.melan.npulab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.Key
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material.icons.outlined.Speed
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import io.melan.npulab.ui.screens.BenchmarkScreen
import io.melan.npulab.ui.screens.CatalogScreen
import io.melan.npulab.ui.screens.DeviceInfoScreen
import io.melan.npulab.ui.screens.GenerateScreen
import io.melan.npulab.ui.screens.SettingsScreen
import io.melan.npulab.ui.screens.TranscribeScreen
import io.melan.npulab.ui.theme.NpuLabTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.Transparent.value.toInt()),
            navigationBarStyle = SystemBarStyle.dark(Color.Transparent.value.toInt()),
        )
        setContent {
            NpuLabTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    NpuLabApp()
                }
            }
        }
    }
}

private data class TabDestination(
    val route: String,
    val label: String,
    val icon: ImageVector,
)

@Composable
private fun NpuLabApp() {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    val tabs = listOf(
        TabDestination("generate", "Generate", Icons.Outlined.AutoAwesome),
        TabDestination("speech", "Speech", Icons.Outlined.Mic),
        TabDestination("benchmark", "Bench", Icons.Outlined.Speed),
        TabDestination("catalog", "Models", Icons.Outlined.CloudDownload),
        TabDestination("device", "Device", Icons.Outlined.Memory),
        TabDestination("settings", "Account", Icons.Outlined.Key),
    )

    Scaffold(
        bottomBar = {
            NavigationBar {
                tabs.forEach { tab ->
                    NavigationBarItem(
                        selected = backStackEntry?.destination?.hierarchy?.any {
                            it.route == tab.route
                        } == true,
                        onClick = {
                            if (currentRoute != tab.route) {
                                navController.navigate(tab.route) {
                                    popUpTo(navController.graph.startDestinationId) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            }
                        },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label) },
                    )
                }
            }
        }
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = "generate",
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            composable("generate") { GenerateScreen() }
            composable("speech") { TranscribeScreen() }
            composable("benchmark") { BenchmarkScreen() }
            composable("catalog") { CatalogScreen() }
            composable("device") { DeviceInfoScreen() }
            composable("settings") { SettingsScreen() }
        }
    }
}
