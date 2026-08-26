/*
 * Adapted from Google AI Edge Gallery
 * (ui/common/chat/MessageBubbleShape.kt), Apache License 2.0:
 * https://github.com/google-ai-edge/gallery
 * Copyright 2025 Google LLC. Ported verbatim (package renamed only) — small,
 * self-contained, brand-agnostic utility.
 */
package com.mediquery.mobile.ui

import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.LayoutDirection

/**
 * Rounded-rectangle outline for chat bubbles with one sharp ("tail") corner —
 * top-right when [hardCornerAtLeftOrRight] is false (right-aligned/user
 * bubbles), top-left when true (left-aligned bubbles).
 */
class MessageBubbleShape(
    private val radius: Dp,
    private val hardCornerAtLeftOrRight: Boolean = false,
) : Shape {
    override fun createOutline(
        size: Size,
        layoutDirection: LayoutDirection,
        density: Density,
    ): Outline {
        val radiusPx = with(density) { radius.toPx() }
        val path = Path().apply {
            addRoundRect(
                RoundRect(
                    left = 0f,
                    top = 0f,
                    right = size.width,
                    bottom = size.height,
                    topLeftCornerRadius = if (hardCornerAtLeftOrRight) CornerRadius(0f, 0f)
                        else CornerRadius(radiusPx, radiusPx),
                    topRightCornerRadius = if (hardCornerAtLeftOrRight) CornerRadius(radiusPx, radiusPx)
                        else CornerRadius(0f, 0f),
                    bottomLeftCornerRadius = CornerRadius(radiusPx, radiusPx),
                    bottomRightCornerRadius = CornerRadius(radiusPx, radiusPx),
                )
            )
        }
        return Outline.Generic(path)
    }
}
