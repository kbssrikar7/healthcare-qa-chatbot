package com.mediquery.mobile

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ProvideTextStyle
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp
import com.halilibo.richtext.commonmark.Markdown
import com.halilibo.richtext.ui.RichTextStyle
import com.halilibo.richtext.ui.material3.RichText

/**
 * Renders LLM output as Markdown instead of raw text (so "**bold**" and "* bullet"
 * actually render instead of showing literal asterisks). Same library
 * (halilibo compose-richtext) Google AI Edge Gallery uses for the same purpose.
 */
@Composable
fun MarkdownText(text: String, modifier: Modifier = Modifier) {
    ProvideTextStyle(
        value = TextStyle(
            fontSize = MaterialTheme.typography.bodyLarge.fontSize,
            lineHeight = MaterialTheme.typography.bodyLarge.fontSize * 1.5f,
            letterSpacing = 0.2.sp,
        )
    ) {
        RichText(modifier = modifier, style = RichTextStyle()) {
            Markdown(content = text)
        }
    }
}
