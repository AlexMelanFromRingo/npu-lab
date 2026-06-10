package io.melan.npulab.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val DarkColors = darkColorScheme(
    primary = NeuralBlue,
    onPrimary = OnDeepSpace,
    primaryContainer = NeuralBlueContainer,
    onPrimaryContainer = OnNeuralBlueContainer,
    secondary = HexagonAmber,
    onSecondary = OnDeepSpace,
    secondaryContainer = HexagonAmberContainer,
    onSecondaryContainer = OnHexagonAmberContainer,
    tertiary = TensorTeal,
    onTertiary = OnDeepSpace,
    tertiaryContainer = TensorTealContainer,
    onTertiaryContainer = OnTensorTealContainer,
    background = DeepSpace,
    onBackground = OnDeepSpace,
    surface = DeepSpaceSurface,
    onSurface = OnDeepSpace,
    surfaceVariant = DeepSpaceSurfaceVariant,
    onSurfaceVariant = OnDeepSpaceSurface,
    error = ErrorRose,
    onError = OnErrorRose,
    errorContainer = ErrorRoseContainer,
    onErrorContainer = OnErrorRoseContainer,
    outline = Outline,
    outlineVariant = OutlineVariant,
)

private val LightColors = lightColorScheme(
    primary = NeuralBlueDark,
    secondary = HexagonAmberContainer,
    tertiary = TensorTealContainer,
)

@Composable
fun NpuLabTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = NpuLabTypography,
        content = content,
    )
}
