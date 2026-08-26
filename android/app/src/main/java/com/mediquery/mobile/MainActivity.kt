package com.mediquery.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.mediquery.mobile.retrieval.CalibratedConfidence
import com.mediquery.mobile.retrieval.ConfidenceScorer
import com.mediquery.mobile.retrieval.HybridRetriever
import com.mediquery.mobile.retrieval.KnowledgeBase
import com.mediquery.mobile.retrieval.OnnxEmbedder
import com.mediquery.mobile.ui.LongPressCopyContainer
import com.mediquery.mobile.ui.MessageBubbleShape
import com.mediquery.mobile.ui.theme.MediQueryTheme
import com.mediquery.mobile.ui.theme.MqConfidenceHigh
import com.mediquery.mobile.ui.theme.MqConfidenceLow
import com.mediquery.mobile.ui.theme.MqConfidenceMedium
import com.mediquery.mobile.ui.theme.MqPrimary
import com.mediquery.mobile.ui.theme.MqPrimaryLight
import com.mediquery.mobile.ui.theme.MqSurfaceVariant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * On-device RAG demo: Gemma3-1B-IT (LiteRT-LM, GPU backend) for generation,
 * all-MiniLM-L6-v2 (ONNX Runtime) for query embedding, hybrid BM25+dense
 * retrieval (RRF-fused) over a curated KB subset. See
 * project_paperwork/scratch/mobile_port_notes.md for the full build history.
 *
 * Required pushed files (adb push, /data/local/tmp):
 *   adb push gemma3-1b-it.task /data/local/tmp/gemma3-1b-it.task
 *   adb push mobile_kb_subset.jsonl /data/local/tmp/mobile_kb.jsonl
 * Embedder model + vocab are bundled as app assets (no push needed).
 *
 * Each turn uses a *fresh* LiteRT-LM Conversation (not one accumulated across
 * the whole session) — the model's 2048-token context is easily exceeded by
 * a few turns' worth of RAG context otherwise. The visual chat transcript is
 * independent of this and always shows full history; there's just no true
 * cross-turn LLM memory yet. See mobile_port_notes.md for this tradeoff.
 */
private const val MODEL_ASSET = "gemma3-1b-it.task"
private const val KB_ASSET = "mobile_kb.jsonl"
private const val RETRIEVE_K = 3
private const val MAX_CONTEXT_CHARS_PER_CHUNK = 600

private fun truncateAtSentence(text: String, maxChars: Int): String {
    if (text.length <= maxChars) return text
    val window = text.take(maxChars)
    val lastSentenceEnd = window.lastIndexOfAny(charArrayOf('.', '!', '?'))
    return if (lastSentenceEnd > maxChars / 2) {
        window.take(lastSentenceEnd + 1)
    } else {
        window.substringBeforeLast(' ') + "..."
    }
}

// Plain `var`s here would NOT trigger recomposition when mutated from the
// background generation coroutine — mutableStateListOf only observes
// structural changes to the list itself, not field mutations on the items it
// holds. Each field must be its own Compose State for streamed token updates
// (turn.answer += ...) to actually repaint the UI.
class ChatTurn(val question: String) {
    var answer by mutableStateOf("")
    var sources by mutableStateOf<List<HybridRetriever.Result>>(emptyList())
    var isGenerating by mutableStateOf(true)
    var errorMessage by mutableStateOf<String?>(null)
    var confidenceLevel by mutableStateOf<String?>(null)
    var confidenceScore by mutableStateOf<Double?>(null)
}

class MainActivity : ComponentActivity() {

    private var engine: Engine? = null
    private var embedder: OnnxEmbedder? = null
    private var retriever: HybridRetriever? = null

    private val modelStatus = mutableStateOf("Not loaded")
    private val modelReady = mutableStateOf(false)
    private val kbChunkCount = mutableStateOf(0)

    // Calibration-data-collection mode — see CalibrationRunner.kt doc comment.
    private val isCalibrationMode by lazy { intent.getStringExtra("mode") == "calibrate" }
    private val calibrationStatus = mutableStateOf("Waiting for model...")
    private val calibrationDone = mutableStateOf(0)
    private val calibrationTotal = mutableStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        loadModel() // start loading immediately — no manual "Load model" step
        setContent {
            MediQueryTheme {
                if (isCalibrationMode) {
                    CalibrationScreen(
                        status = calibrationStatus.value,
                        done = calibrationDone.value,
                        total = calibrationTotal.value,
                    )
                } else {
                    ChatScreen(
                        modelStatus = modelStatus.value,
                        modelReady = modelReady.value,
                        kbChunkCount = kbChunkCount.value,
                        onSend = { question, turn -> sendMessage(question, turn) },
                    )
                }
            }
        }
    }

    private fun runCalibration() {
        val eng = engine
        val ret = retriever
        if (eng == null || ret == null) {
            calibrationStatus.value = "Error: engine or retriever not ready"
            return
        }
        lifecycleScope.launch(Dispatchers.Default) {
            withContext(Dispatchers.Main) { calibrationStatus.value = "Running calibration set..." }
            CalibrationRunner.run(this@MainActivity, eng, ret) { result ->
                lifecycleScope.launch(Dispatchers.Main) {
                    calibrationDone.value = result.index
                    calibrationTotal.value = result.total
                    calibrationStatus.value = if (result.error != null) {
                        "[${result.index}/${result.total}] ${result.id} — error: ${result.error}"
                    } else {
                        "[${result.index}/${result.total}] ${result.id} — ok"
                    }
                }
            }
            val outFile = File(getExternalFilesDir(null), CalibrationRunner.OUTPUT_FILENAME)
            withContext(Dispatchers.Main) {
                calibrationStatus.value = "Done. Results at ${outFile.absolutePath}"
            }
        }
    }

    private fun loadModel() {
        modelStatus.value = "Setting up (first run)..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                var lastReportedPct = -1
                val modelFile = AssetInstaller.ensureInstalled(
                    this@MainActivity, MODEL_ASSET, File(filesDir, MODEL_ASSET),
                ) { copied, total ->
                    val pct = if (total > 0) (copied * 100 / total).toInt() else 0
                    if (pct != lastReportedPct) {
                        lastReportedPct = pct
                        modelStatus.value = "Setting up model... $pct%"
                    }
                }
                val kbFile = AssetInstaller.ensureInstalled(this@MainActivity, KB_ASSET, File(filesDir, KB_ASSET))

                modelStatus.value = "Loading embedder + knowledge base..."
                val newEmbedder = OnnxEmbedder(this@MainActivity)
                val chunks = KnowledgeBase.load(kbFile.absolutePath)
                embedder = newEmbedder
                retriever = HybridRetriever(chunks, newEmbedder)
                withContext(Dispatchers.Main) {
                    kbChunkCount.value = chunks.size
                    modelStatus.value = "Loading language model..."
                }

                val engineConfig = EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = Backend.GPU(),
                    maxNumTokens = 2048,
                    cacheDir = getExternalFilesDir(null)?.absolutePath,
                )
                val newEngine = Engine(engineConfig)
                newEngine.initialize()
                engine = newEngine

                withContext(Dispatchers.Main) {
                    modelStatus.value = "Ready"
                    modelReady.value = true
                }
                if (isCalibrationMode) runCalibration()
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { modelStatus.value = "Failed to load: ${e.message}" }
            }
        }
    }

    private fun buildRagPrompt(question: String, retrieved: List<HybridRetriever.Result>): String {
        if (retrieved.isEmpty()) return question
        val context = retrieved.withIndex().joinToString("\n\n") { (i, r) ->
            "[${i + 1}] ${truncateAtSentence(r.chunk.text, MAX_CONTEXT_CHARS_PER_CHUNK)}"
        }
        return "Answer the medical question using ONLY the context passages below. " +
            "If the context does not contain the answer, say so explicitly rather than guessing.\n\n" +
            "Context:\n$context\n\nQuestion: $question"
    }

    private fun sendMessage(question: String, turn: ChatTurn) {
        val eng = engine
        if (eng == null) {
            turn.errorMessage = "Model not ready yet."
            turn.isGenerating = false
            return
        }
        lifecycleScope.launch(Dispatchers.Default) {
            val retrieved = retriever?.retrieve(question, k = RETRIEVE_K) ?: emptyList()
            withContext(Dispatchers.Main) { turn.sources = retrieved }
            val prompt = buildRagPrompt(question, retrieved)

            // Fresh conversation per turn — see class doc for why.
            val conversation: Conversation = eng.createConversation(
                ConversationConfig(
                    samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 1.0),
                )
            )
            conversation.sendMessageAsync(
                Contents.of(prompt),
                object : MessageCallback {
                    override fun onMessage(message: Message) {
                        turn.answer += message.toString()
                    }
                    override fun onDone() {
                        val retrievalConf = ConfidenceScorer.retrievalConfidence(retrieved)
                        val sourceAgreement = ConfidenceScorer.sourceAgreement(turn.answer, retrieved)
                        val raw = ConfidenceScorer.rawScore(retrievalConf, sourceAgreement)
                        val calibrated = CalibratedConfidence.default.calibrate(raw)
                        turn.confidenceScore = calibrated
                        turn.confidenceLevel = CalibratedConfidence.level(calibrated)
                        turn.isGenerating = false
                        conversation.close()
                    }
                    override fun onError(throwable: Throwable) {
                        turn.errorMessage = "Error: ${throwable.message}"
                        turn.isGenerating = false
                        conversation.close()
                    }
                },
            )
        }
    }

    override fun onDestroy() {
        try {
            engine?.close()
            embedder?.close()
        } catch (_: Exception) {
        }
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    modelStatus: String,
    modelReady: Boolean,
    kbChunkCount: Int,
    onSend: (String, ChatTurn) -> Unit,
) {
    val messages = remember { mutableStateListOf<ChatTurn>() }
    var query by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    fun submit() {
        val q = query.trim()
        if (q.isEmpty() || !modelReady) return
        val turn = ChatTurn(question = q)
        messages.add(turn)
        query = ""
        onSend(q, turn)
    }

    LaunchedEffect(messages.size, messages.lastOrNull()?.answer) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("MediQuery", fontWeight = FontWeight.Bold)
                        Text(
                            text = if (modelReady) "$kbChunkCount sources loaded" else modelStatus,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
        bottomBar = {
            Column {
                if (!modelReady) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.padding(4.dp))
                        Text(modelStatus, style = MaterialTheme.typography.labelSmall)
                    }
                }
                Text(
                    text = "For educational use only — not a substitute for professional medical advice.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
                )
                InputBar(
                    value = query,
                    onValueChange = { query = it },
                    enabled = modelReady,
                    onSend = { submit() },
                )
            }
        },
    ) { padding ->
        if (messages.isEmpty()) {
            EmptyState(modifier = Modifier.padding(padding))
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding),
                contentPadding = PaddingValues(vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                itemsIndexed(messages) { _, turn -> ChatTurnView(turn) }
            }
        }
    }
}

@Composable
private fun EmptyState(modifier: Modifier = Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "Ask a medical question",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                "Answers are grounded in retrieved sources, shown below each response.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp, start = 32.dp, end = 32.dp),
            )
        }
    }
}

@Composable
private fun ChatTurnView(turn: ChatTurn) {
    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp)) {
        // User message — a bubble, right-aligned. Sharp top-right corner
        // ("tail" toward the screen edge) matches the convention used by
        // Google AI Edge Gallery's chat UI (MessageBubbleShape).
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            LongPressCopyContainer(copyText = turn.question) {
                Surface(
                    color = MqPrimary,
                    shape = MessageBubbleShape(radius = 18.dp, hardCornerAtLeftOrRight = false),
                    modifier = Modifier.widthIn(max = 300.dp),
                ) {
                    Text(
                        text = turn.question,
                        color = Color.White,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 11.dp),
                    )
                }
            }
        }
        Spacer(Modifier.height(10.dp))
        // Assistant response — deliberately NOT in a bubble (same choice
        // Gallery makes for agent text messages: bubbles cramp long
        // markdown-formatted answers; plain full-width text reads better,
        // closer to how ChatGPT/Claude render responses).
        LongPressCopyContainer(
            copyText = turn.answer,
            modifier = Modifier.fillMaxWidth(0.94f),
        ) {
            Column {
                when {
                    turn.errorMessage != null -> Text(
                        turn.errorMessage!!,
                        color = MaterialTheme.colorScheme.error,
                    )
                    turn.isGenerating && turn.answer.isEmpty() -> ThinkingIndicator()
                    else -> MarkdownText(text = turn.answer)
                }
                if (turn.confidenceLevel != null) {
                    Spacer(Modifier.height(8.dp))
                    ConfidenceBadge(turn.confidenceLevel!!, turn.confidenceScore!!)
                }
                if (turn.sources.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    SourceChips(turn.sources)
                }
            }
        }
    }
}

@Composable
private fun ConfidenceBadge(level: String, score: Double) {
    val color = when (level) {
        "high" -> MqConfidenceHigh
        "medium" -> MqConfidenceMedium
        else -> MqConfidenceLow
    }
    Surface(color = color.copy(alpha = 0.15f), shape = RoundedCornerShape(8.dp)) {
        Text(
            text = "Confidence: ${level.replaceFirstChar { it.uppercase() }} (${(score * 100).toInt()}%)",
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun ThinkingIndicator() {
    val infiniteTransition = rememberInfiniteTransition(label = "thinking")
    Row(verticalAlignment = Alignment.CenterVertically) {
        repeat(3) { index ->
            val alpha by infiniteTransition.animateFloat(
                initialValue = 0.25f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(600, delayMillis = index * 150, easing = LinearEasing),
                    repeatMode = RepeatMode.Reverse,
                ),
                label = "dot$index",
            )
            Box(
                modifier = Modifier
                    .padding(end = 4.dp)
                    .size(7.dp)
                    .graphicsLayer { this.alpha = alpha }
                    .background(MqPrimaryLight, shape = CircleShape),
            )
        }
    }
}

@Composable
private fun SourceChips(sources: List<HybridRetriever.Result>) {
    Column {
        Text(
            "Sources",
            style = MaterialTheme.typography.labelSmall,
            color = MqPrimaryLight,
            fontWeight = FontWeight.Bold,
        )
        Row(
            modifier = Modifier.padding(top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            sources.distinctBy { it.chunk.source }.forEach { r ->
                Surface(
                    color = MqPrimary.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Text(
                        r.chunk.source,
                        style = MaterialTheme.typography.labelSmall,
                        color = MqPrimaryLight,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun InputBar(
    value: String,
    onValueChange: (String) -> Unit,
    enabled: Boolean,
    onSend: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        TextField(
            value = value,
            onValueChange = onValueChange,
            enabled = enabled,
            placeholder = { Text(if (enabled) "Ask a question" else "Loading...") },
            modifier = Modifier.weight(1f),
            shape = RoundedCornerShape(24.dp),
            colors = TextFieldDefaults.colors(
                unfocusedContainerColor = MqSurfaceVariant,
                focusedContainerColor = MqSurfaceVariant,
                unfocusedIndicatorColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
            ),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text, imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { onSend() }),
            textStyle = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.padding(4.dp))
        IconButton(
            onClick = onSend,
            enabled = enabled && value.isNotBlank(),
        ) {
            Surface(
                color = if (enabled && value.isNotBlank()) MqPrimary else MqSurfaceVariant,
                shape = RoundedCornerShape(50),
            ) {
                Icon(
                    imageVector = Icons.Filled.Send,
                    contentDescription = "Send",
                    tint = Color.White,
                    modifier = Modifier.padding(10.dp),
                )
            }
        }
    }
}

@Composable
private fun CalibrationScreen(status: String, done: Int, total: Int) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Calibration run", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(16.dp))
        Text(status, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(16.dp))
        if (total > 0) {
            Text("$done / $total", style = MaterialTheme.typography.headlineSmall)
        }
    }
}
