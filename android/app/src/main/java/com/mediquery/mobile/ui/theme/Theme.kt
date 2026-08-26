package com.mediquery.mobile.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

// Desktop forces a dark palette regardless of system preference
// (frontend-react/app/globals.css: "Force a consistent shadcn-style dark
// palette across the app") — match that here for brand consistency.
private val MqDarkColorScheme = darkColorScheme(
    primary = MqPrimary,
    onPrimary = MqForeground,
    secondary = MqPrimaryLight,
    background = MqBackground,
    onBackground = MqForeground,
    surface = MqSurface,
    onSurface = MqForeground,
    surfaceVariant = MqSurfaceVariant,
    onSurfaceVariant = MqMutedForeground,
    outline = MqBorder,
    error = MqError,
)

@Composable
fun MediQueryTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = MqDarkColorScheme, content = content)
}
