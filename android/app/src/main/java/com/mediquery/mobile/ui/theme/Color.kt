package com.mediquery.mobile.ui.theme

import androidx.compose.ui.graphics.Color

// Ported directly from the desktop app's brand palette
// (frontend-react/app/globals.css + public/mediquery-icon.svg), so the mobile
// app looks like the same product, not a generic Material default.
val MqPrimary = Color(0xFF1D4ED8) // brand blue, from mediquery-icon.svg
val MqPrimaryLight = Color(0xFF93C5FD) // pulse-line accent, from mediquery-icon.svg
val MqBackground = Color(0xFF0F172A) // dark navy, matches --background
val MqSurface = Color(0xFF0F172A)
val MqSurfaceVariant = Color(0xFF1E293B) // matches --accent
val MqForeground = Color(0xFFF8FAFC) // matches --foreground
val MqMutedForeground = Color(0xFF94A3B8)
val MqBorder = Color(0xFF26344A)
val MqError = Color(0xFFB91C1C)

// Confidence-level semantic colors, matching the emerald/amber/red family the
// desktop frontend already uses for confidence/factual-consistency state
// (frontend-react/components/mediquery/answer-card.tsx).
val MqConfidenceHigh = Color(0xFF34D399) // emerald-400
val MqConfidenceMedium = Color(0xFFFBBF24) // amber-400
val MqConfidenceLow = Color(0xFFFCA5A5) // red-300
