"""
Streamlit frontend for Healthcare QA Chatbot.
Professional, modern UI design without emoji clutter.
"""
import os
import json
import logging
import streamlit as st
import requests
import time
import html
import tempfile
import threading
from typing import Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speech-to-text (faster-whisper, CPU-only, tiny model)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock = threading.Lock()

def _load_whisper():
    """Lazy-load Whisper tiny model (cached after first call)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                    # int8 quantisation keeps it fast on CPU; tiny model ~75 MB
                    _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                except Exception as e:
                    logger.warning(f"Whisper model load failed, audio disabled: {e}")
                    _whisper_model = None
    return _whisper_model


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe raw WAV/WebM audio bytes to text using Whisper tiny."""
    model = _load_whisper()
    if model is None:
        return ""
    try:
        import numpy as np, io
        # Write to a temp file — faster_whisper needs a file path or numpy array
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        segments, _ = model.transcribe(tmp_path, language="en", beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        Path(tmp_path).unlink(missing_ok=True)
        return text
    except Exception as e:
        logger.warning(f"Audio transcription failed: {e}")
        return ""

def sanitize_html(text: str) -> str:
    """Escape HTML content to prevent XSS attacks."""
    return html.escape(str(text)) if text else ""

# Configuration - Load from environment with fallback
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="MediQuery | Advanced Medical AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%234f46e5'/><path d='M50 25v50M25 50h50' stroke='white' stroke-width='8' stroke-linecap='round'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_css():
    """Generate CSS variables based on current theme."""
    if st.session_state.dark_mode:
        return """
    :root {
        --primary: #818cf8;
        --primary-dark: #6366f1;
        --primary-light: #a5b4fc;
        --secondary: #38bdf8;
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f87171;
        --surface: #1e1e2e;
        --surface-elevated: #262637;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border: #334155;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4), 0 2px 4px -2px rgba(0,0,0,0.3);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5), 0 4px 6px -4px rgba(0,0,0,0.4);
        --radius-sm: 0px;
        --radius-md: 0px;
        --radius-lg: 0px;
        --user-card-bg: #1e293b;
        --user-card-border: #334155;
        --user-card-label: #60a5fa;
        --disclaimer-bg: #2d2305;
        --disclaimer-border: #a16207;
        --disclaimer-title: #fbbf24;
        --disclaimer-text: #fde68a;
        --disclaimer-meta: #d97706;
        --reasoning-bg: #052e16;
        --reasoning-text: #6ee7b7;
        --attr-bg: #1e293b;
        --source-badge-bg: #312e81;
        --source-badge-text: #a5b4fc;
        --info-bg: #0c2d48;
        --info-border: #1e5f8a;
        --info-text: #38bdf8;
    }

    /* ===== Dark Mode: Override ALL Streamlit elements ===== */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #1e1e2e !important;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    .main .block-container {
        background-color: #1e1e2e !important;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #262637 0%, #1e1e2e 100%) !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stHeader"] {
        background-color: #1e1e2e !important;
    }
    
    /* Text & headings */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
    .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    /* Widgets, sliders, toggles */
    [data-testid="stSlider"] label, [data-testid="stSlider"] div,
    .stSlider p, .stToggle label span {
        color: #94a3b8 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #e2e8f0 !important;
        background-color: #262637 !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #262637 !important;
        border-color: #334155 !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"], [data-testid="stChatInput"] textarea {
        background-color: #262637 !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #262637 !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
    .stButton > button:hover {
        border-color: #818cf8 !important;
        color: #818cf8 !important;
    }
    
    /* Status widget */
    [data-testid="stStatusWidget"], [data-testid="stStatus"] {
        background-color: #262637 !important;
        border-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    /* Horizontal rule */
    .stApp hr {
        border-color: #334155 !important;
    }
    
    /* ===== Bottom chat input bar ===== */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    .stChatInput, .stChatInputContainer,
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"] {
        background-color: #1e1e2e !important;
        border-color: #334155 !important;
    }
    
    /* Chat input textarea specifically */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    .stChatInput textarea {
        background-color: #262637 !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
        caret-color: #818cf8 !important;
    }
    
    /* Chat input placeholder */
    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Chat input submit button */
    [data-testid="stChatInput"] button,
    .stChatInput button {
        background-color: #262637 !important;
        color: #818cf8 !important;
        border-color: #334155 !important;
    }
    
    /* Bottom content container (fixed position area) */
    .stBottom, div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div:first-child {
        background-color: #1e1e2e !important;
        border-top: 1px solid #334155 !important;
    }
    
    /* Any remaining white containers */
    .element-container, .stMarkdown,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    
    /* Tooltip and popover */
    [data-testid="stTooltipContent"] {
        background-color: #262637 !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }"""
    else:
        return """
    :root {
        --primary: #4f46e5;
        --primary-dark: #3730a3;
        --primary-light: #818cf8;
        --secondary: #0ea5e9;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --surface: #ffffff;
        --surface-elevated: #f8fafc;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.05);
        --radius-sm: 0px;
        --radius-md: 0px;
        --radius-lg: 0px;
        --user-card-bg: #eff6ff;
        --user-card-border: #bfdbfe;
        --user-card-label: #2563eb;
        --disclaimer-bg: #fffbeb;
        --disclaimer-border: var(--warning);
        --disclaimer-title: #b45309;
        --disclaimer-text: #78350f;
        --disclaimer-meta: #a16207;
        --reasoning-bg: #f0fdf4;
        --reasoning-text: #166534;
        --attr-bg: #f1f5f9;
        --source-badge-bg: #eef2ff;
        --source-badge-text: var(--primary);
        --info-bg: #f0f9ff;
        --info-border: #bae6fd;
        --info-text: #0369a1;
    }"""  

# Custom CSS - Premium Design System (loaded from external file + dynamic theme vars)
import pathlib as _pathlib
_css_path = _pathlib.Path(__file__).parent / "static" / "style.css"
_static_css = _css_path.read_text() if _css_path.exists() else ""
st.markdown(
    "<style>\n    /* ========== CSS Variables & Theme ========== */\n    "
    + get_theme_css()
    + "\n"
    + _static_css
    + "\n</style>",
    unsafe_allow_html=True,
)


def fetch_available_models() -> dict:
    """Fetch available models from API."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch models from API: {e}")
    # Fallback — include BioMistral if the GGUF file exists locally
    from pathlib import Path as _Path
    models = {
        "tinyllama": {"display_name": "TinyLlama 1.1B", "description": "Lightweight, fast responses", "parameters": "1.1B", "requires_gpu": False, "loaded": False},
    }
    biomistral_path = _Path(__file__).parent.parent / "models" / "biomistral" / "ggml-model-Q4_K_M.gguf"
    if biomistral_path.exists():
        models["biomistral"] = {"display_name": "BioMistral 7B (Q4_K_M)", "description": "Medical-domain Mistral 7B, CPU inference", "parameters": "7B", "requires_gpu": False, "loaded": False}
    return models


def ask_question(
    question: str,
    num_sources: int = 5,
    model_choice: str = None,
    use_langchain: bool = False,
    use_langgraph: bool = False,
    session_id: str = None,
) -> Optional[dict]:
    """Send question to API and get response."""
    try:
        payload = {
            "question": question,
            "include_explanation": True,
            "num_sources": num_sources,
        }
        if model_choice:
            payload["model_choice"] = model_choice
        if use_langchain:
            payload["use_langchain"] = True
        if use_langgraph:
            payload["use_langgraph"] = True
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(
            f"{API_URL}/ask",
            json=payload,
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.error("System is initializing or busy. Please try again in a moment.")
            return None
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API server is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded or still initializing. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def clear_backend_cache():
    """Clear the backend response cache."""
    try:
        response = requests.post(f"{API_URL}/clear-cache", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Cache clear request failed: {e}")
        return False


def submit_feedback(
    response_id: str,
    rating: Optional[int] = None,
    was_helpful: Optional[bool] = None,
    was_accurate: Optional[bool] = None,
    was_safe: Optional[bool] = None,
    feedback_text: Optional[str] = None,
) -> bool:
    """Submit answer feedback to backend."""
    payload = {
        "response_id": response_id,
        "session_id": st.session_state.get("session_id"),
    }
    if rating is not None:
        payload["rating"] = rating
    if was_helpful is not None:
        payload["was_helpful"] = was_helpful
    if was_accurate is not None:
        payload["was_accurate"] = was_accurate
    if was_safe is not None:
        payload["was_safe"] = was_safe
    if feedback_text:
        payload["feedback_text"] = feedback_text

    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json=payload,
            timeout=15,
        )
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Failed to submit feedback: {e}")
        return False


def display_confidence(confidence: dict, breakdown: dict = None):
    """Display animated confidence meter with badge and optional 5-signal breakdown."""
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0)
    explanation = confidence.get("explanation", "")

    width_percent = int(score * 100)

    badge_colors = {"high": "#34d399", "medium": "#fbbf24", "low": "#f87171"}
    badge_labels = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}
    badge_color = badge_colors.get(level, "#fbbf24")
    badge_label = badge_labels.get(level, "Medium confidence")

    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-header">
            <span class="confidence-label">AI confidence score</span>
            <span class="confidence-badge" style="background: {badge_color}; color: #1e1e2e; padding: 2px 10px; border-radius: 0; font-size: 0.75rem; font-weight: 600;">{badge_label} &middot; {width_percent}%</span>
        </div>
        <div class="confidence-track">
            <div class="confidence-fill {level}" style="width: {width_percent}%"></div>
        </div>
        <div class="confidence-desc">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    # 5-signal XAI breakdown (only shown when multi-signal scorer ran)
    if breakdown:
        with st.expander("XAI Signal Breakdown", expanded=False):
            signal_labels = {
                "retrieval_confidence": "Retrieval Quality",
                "generation_confidence": "Generation Certainty",
                "consistency_score": "Self-Consistency",
                "source_agreement": "Source Agreement",
                "medical_entity_coverage": "Entity Coverage",
            }
            weights = breakdown.get("signal_weights", {})
            signal_map = {
                "retrieval_confidence": "retrieval",
                "generation_confidence": "generation",
                "consistency_score": "consistency",
                "source_agreement": "source_agreement",
                "medical_entity_coverage": "entity_coverage",
            }

            rows = []
            for key, label in signal_labels.items():
                val = breakdown.get(key, 0.0)
                weight_key = signal_map.get(key, "")
                weight = weights.get(weight_key, 0.0)
                rows.append((label, val, weight))

            # Render as styled horizontal bars
            st.markdown(
                "<div style='font-size:0.78rem; font-weight:600; color:var(--text-secondary,#888); "
                "margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;'>"
                "Signal &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                "Score &nbsp;&nbsp;&nbsp; Weight</div>",
                unsafe_allow_html=True,
            )
            bar_colors = {"high": "#34d399", "medium": "#fbbf24", "low": "#f87171"}
            for label, val, weight in rows:
                pct = int(val * 100)
                lvl = "high" if val >= 0.75 else ("medium" if val >= 0.45 else "low")
                bar_color = bar_colors[lvl]
                st.markdown(
                    f"""<div style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span style="font-size:0.85rem; color:var(--text-primary,#222);">{label}</span>
                        <span style="font-size:0.8rem; color:var(--text-secondary,#666);">
                          <b style="color:{bar_color};">{pct}%</b> &nbsp; w={weight:.0%}
                        </span>
                      </div>
                      <div style="height:6px; background:var(--border,#e2e8f0); border-radius:3px; overflow:hidden;">
                        <div style="height:100%; width:{pct}%; background:{bar_color}; border-radius:3px;"></div>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def display_sources(sources: list):
    """Display source cards in a grid."""
    if not sources:
        st.markdown("""
        <div class="info-box">
            No specific medical sources were used for this answer. The AI used its general medical knowledge.
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">Verified Sources</div>', unsafe_allow_html=True)
    
    # Create grid of sources
    cols = st.columns(min(3, len(sources)))
    
    for i, source in enumerate(sources):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="source-card">
                <div class="source-badge">Match: {source['score']:.0%}</div>
                <div class="source-text">{sanitize_html(source['content'][:180])}...</div>
                <div class="source-meta">{sanitize_html(source['source'])}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Full Content"):
                st.write(source['content'])
                if source.get('url'):
                    st.markdown(f"[View Original Source]({source['url']})")


def display_attributions(attributions: list):
    """Display attributions for specific claims."""
    if not attributions:
        return
        
    with st.expander("Claim Verification Details"):
        for attr in attributions:
            if attr['source'] != "Unsupported":
                st.markdown(f"""
                <div class="attribution-card">
                    <div class="attribution-claim">"{sanitize_html(attr['claim'])}"</div>
                    <div class="attribution-source">Verified by: {sanitize_html(attr['source'])} ({attr['similarity']:.0%} match)</div>
                </div>
                """, unsafe_allow_html=True)


def render_answer(result):
    """Render the full answer card with professional styling."""

    # ── Emergency / Safety Warning ──
    safety = result.get('safety')
    if safety and safety.get('is_emergency'):
        st.markdown(f"""
        <div style="background:#dc2626; color:white; padding:16px 20px; border-radius:0; margin-bottom:16px; font-weight:600;">
            {sanitize_html(safety.get('emergency_message', 'Please seek immediate medical attention.'))}
        </div>
        """, unsafe_allow_html=True)

    if safety and safety.get('drug_warnings'):
        for w in safety['drug_warnings']:
            st.warning(w)

    # ── Bot Answer Card ──
    st.markdown(f"""
    <div class="chat-card bot-card">
        <div class="card-label">MediQuery AI</div>
        <div class="card-content">{sanitize_html(result['answer'])}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Response metadata badges ──
    meta_parts = []
    if result.get('model_used'):
        meta_parts.append(f"Model: {result['model_used']}")
    if result.get('pipeline_used'):
        meta_parts.append(f"Pipeline: {result['pipeline_used']}")
    if result.get('latency_ms'):
        meta_parts.append(f"Latency: {result['latency_ms']:.0f}ms")
    elif result.get('elapsed_time'):
        meta_parts.append(f"Latency: {result['elapsed_time']:.2f}s")
    if meta_parts:
        badges_html = " &middot; ".join(meta_parts)
        st.markdown(f"""
        <div style="color: var(--text-secondary, #888); font-size: 0.78rem; margin: -8px 0 12px 4px;">
            {badges_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Safety level badge ──
    if safety and safety.get('level') and safety['level'] != 'safe':
        level_colors = {'caution': '#fbbf24', 'blocked': '#f87171', 'emergency': '#dc2626'}
        badge_color = level_colors.get(safety['level'], '#888')
        st.markdown(f"""
        <span style="background:{badge_color}; color:#1e1e2e; padding:2px 10px; border-radius:0; font-size:0.75rem; font-weight:600;">
            Safety: {safety['level'].title()}
        </span>
        """, unsafe_allow_html=True)

    # Display Rationale if available
    if result.get('rationale'):
        with st.expander("AI Reasoning", expanded=False):
            st.markdown(f"""
            <div class="reasoning-box">
                {sanitize_html(result['rationale'])}
            </div>
            """, unsafe_allow_html=True)
    
    # Key Terms Analysis
    if result.get('sources'):
        with st.expander("Key Terms Analysis", expanded=False):
            question_words = result.get('question', '').lower().split()
            answer_text = result.get('answer', '').lower()
            
            medical_terms = ['diabetes', 'hypertension', 'symptoms', 'treatment', 'diagnosis', 
                           'pain', 'blood', 'heart', 'disease', 'medication', 'chronic', 
                           'acute', 'fever', 'infection', 'pressure', 'sugar', 'insulin']
            
            found_terms = []
            for term in medical_terms:
                if term in answer_text or term in ' '.join(question_words):
                    found_terms.append(term)
            
            if found_terms:
                st.markdown("**Key Medical Terms Detected:**")
                term_html = " ".join([
                    f'<span class="key-term">{term}</span>'
                    for term in found_terms[:8]
                ])
                st.markdown(term_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    No specific medical terms highlighted for this query.
                </div>
                """, unsafe_allow_html=True)
    
    # Metrics & Sources — pass 5-signal breakdown if available
    display_confidence(result['confidence'], result.get('confidence_breakdown'))
    display_sources(result['sources'])
    display_attributions(result.get('attributions', []))

    # Hallucination detection result
    hal = result.get('hallucination')
    if hal:
        with st.expander("Hallucination Analysis", expanded=False):
            score = hal.get('score', 0.0)
            has_hal = hal.get('has_hallucination', False)
            hal_type = hal.get('type', 'none')
            explanation = hal.get('explanation', '')
            med_flags = hal.get('medical_accuracy_flags', [])

            status_color = "#f87171" if has_hal else "#34d399"
            status_label = f"Detected ({hal_type})" if has_hal else "Clean"
            st.markdown(
                f"""<div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                  <span style="background:{status_color}; color:#1e1e2e; padding:2px 12px;
                    font-size:0.75rem; font-weight:700; border-radius:0;">
                    {status_label}
                  </span>
                  <span style="font-size:0.85rem; color:var(--text-secondary,#666);">
                    Hallucination risk: <b>{int(score*100)}%</b>
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )
            if explanation:
                st.caption(explanation)
            if med_flags:
                st.warning("Medical accuracy flags: " + "; ".join(med_flags))
    
    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-box">
        <div class="disclaimer-title">Medical Disclaimer</div>
        <div class="disclaimer-text">{result['disclaimer']}</div>
    </div>
    """, unsafe_allow_html=True)

    response_id = result.get("response_id")
    if response_id:
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = {}

        submitted = st.session_state.feedback_submitted.get(response_id)
        if submitted:
            st.caption(f"Feedback recorded: {submitted}")
        else:
            st.caption("Please rate the medical accuracy:")
            fb_col1, fb_col2, _ = st.columns([1, 1, 6])

            if fb_col1.button("Accurate", key=f"feedback_helpful_{response_id}", use_container_width=True):
                ok = submit_feedback(
                    response_id=response_id,
                    rating=5,
                    was_helpful=True,
                    was_accurate=True,
                    was_safe=True,
                )
                if ok:
                    st.session_state.feedback_submitted[response_id] = "Accurate"
                    st.rerun()
                else:
                    st.warning("Could not submit feedback. Check backend /feedback endpoint.")

            if fb_col2.button("Inaccurate", key=f"feedback_negative_{response_id}", use_container_width=True):
                ok = submit_feedback(
                    response_id=response_id,
                    rating=1,
                    was_helpful=False,
                    was_accurate=False,
                    was_safe=True,
                )
                if ok:
                    st.session_state.feedback_submitted[response_id] = "Inaccurate"
                    st.rerun()
                else:
                    st.warning("Could not submit feedback. Check backend /feedback endpoint.")


# ========== Question History Persistence ==========
HISTORY_FILE = Path("data/question_history.json")

def load_question_history():
    """Load question history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load question history: {e}")
            return []
    return []

def save_question_history(history):
    """Save question history to disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save question history: {e}")

def add_to_history(question):
    """Add a question to persistent history (no duplicates)."""
    history = load_question_history()
    # Remove duplicate if exists
    history = [h for h in history if h['question'] != question]
    # Add to front
    history.insert(0, {
        'question': question,
        'timestamp': datetime.now().strftime('%b %d, %I:%M %p')
    })
    # Keep last 20 questions
    history = history[:20]
    save_question_history(history)
    return history

def clear_question_history():
    """Clear all question history."""
    save_question_history([])


def processing_chain(question, num_sources, model_choice=None, use_langchain=False, use_langgraph=False):
    """Process a question and add to history."""
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.status("Analyzing medical literature...", expanded=True) as status:
        st.markdown("""
        <div class="loading-indicator">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <span>Searching knowledge base...</span>
        </div>
        """, unsafe_allow_html=True)
        
        start_time = time.time()
        result = ask_question(
            question,
            num_sources,
            model_choice=model_choice,
            use_langchain=use_langchain,
            use_langgraph=use_langgraph,
            session_id=st.session_state.get("session_id"),
        )
        elapsed = time.time() - start_time
        
        if result:
            result['elapsed_time'] = elapsed
            # Track session_id from response
            if result.get('session_id'):
                st.session_state.session_id = result['session_id']
                # Persist session_id to URL so it survives page reloads
                try:
                    st.query_params["sid"] = result['session_id']
                except Exception as e:
                    logger.debug(f"Could not persist session_id to URL params: {e}")
            st.write("Response generated successfully")
            status.update(label="Response ready", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            # Save question to persistent history
            add_to_history(question)
            st.rerun()
        else:
            status.update(label="Error generating response", state="error")
            st.toast("Failed to get a response. Is the API server running?", icon="\u26a0\ufe0f")


def main():
    # Initialize session state
    if "session_id" not in st.session_state:
        # Restore session_id from URL query params (survives page reloads)
        saved_sid = st.query_params.get("sid")
        st.session_state.session_id = saved_sid if saved_sid else None

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M12 4v16M4 12h16" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
        <h1 class="sidebar-title">MediQuery</h1>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # New chat button
        if st.button("New chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()
        
        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)

        # Model selector (TinyLlama + BioMistral when available)
        available_models = fetch_available_models()
        model_display_names = {
            k: f"{v.get('display_name', k)} ({v.get('parameters', '?')})"
            for k, v in available_models.items()
        }
        if len(available_models) > 1:
            selected_label = st.selectbox(
                "LLM Model",
                options=list(model_display_names.values()),
                index=0,
                help="TinyLlama is fast; BioMistral-7B is slower but medically specialised",
            )
            model_choice = next(
                k for k, v in model_display_names.items() if v == selected_label
            )
        else:
            model_choice = list(available_models.keys())[0]
            model_info = available_models.get(model_choice, {})
            st.caption(f"{model_info.get('display_name', 'TinyLlama 1.1B')} ({model_info.get('parameters', '1.1B')})")

        # Pipeline selector
        pipeline_choice = st.radio(
            "Pipeline",
            ["Standard", "LangChain (LCEL)", "LangGraph (Self-Correcting)"],
            help="Standard is recommended for most use cases",
        )
        use_langchain = pipeline_choice == "LangChain (LCEL)"
        use_langgraph = pipeline_choice == "LangGraph (Self-Correcting)"

        st.markdown('<div class="sidebar-section">Analysis settings</div>', unsafe_allow_html=True)
        num_sources = st.slider(
            "Number of references", 
            min_value=2, 
            max_value=10, 
            value=3, 
            help="More sources = slower but more thorough analysis"
        )
        
        st.markdown('---')
        st.markdown('<div class="sidebar-section">Evaluation results</div>', unsafe_allow_html=True)
        _eval_dir = Path(__file__).parent.parent / "evaluation" / "results"
        _calib_file = _eval_dir / "calibration.json"
        _model_files = [
            ("TinyLlama", _eval_dir / "metrics_full_tinyllama.json"),
            ("BioMistral", _eval_dir / "metrics_full_biomistral.json"),
            ("QLoRA", _eval_dir / "metrics_full_qlora.json"),
        ]
        _any_shown = False
        for _label, _mfile in _model_files:
            if _mfile.exists():
                try:
                    _m = json.loads(_mfile.read_text())
                    with st.expander(_label):
                        st.write(f"**KW Coverage:** {_m.get('keyword_coverage_mean', _m.get('kw_coverage_mean', 0)):.1%}")
                        st.write(f"**Answerable:** {_m.get('answerable_pct', 0):.1%}")
                        if _m.get('rougeL_mean'):
                            st.write(f"**ROUGE-L:** {_m['rougeL_mean']:.3f}")
                    _any_shown = True
                except Exception as e:
                    logger.warning(f"Failed to load eval metrics {_mfile}: {e}")
        if _calib_file.exists():
            try:
                _calib = json.loads(_calib_file.read_text())
                st.caption(f"Confidence ECE: {_calib.get('ece', 'N/A')}")
            except Exception as e:
                logger.warning(f"Failed to load calibration data: {e}")
        if not _any_shown:
            st.caption("Run evaluation/run_paper_eval.py to populate")

        st.markdown('---')
        st.markdown('<div class="sidebar-section">Appearance</div>', unsafe_allow_html=True)
        st.toggle(
            "Dark mode",
            key="dark_mode"
        )
        
        st.markdown('---')
        st.markdown('<div class="sidebar-section">Question history</div>', unsafe_allow_html=True)
        
        history = load_question_history()
        if history:
            for i, item in enumerate(history[:10]):
                q_display = item['question'][:50] + ('...' if len(item['question']) > 50 else '')
                if st.button(
                    q_display,
                    key=f"hist_{i}",
                    use_container_width=True,
                    help=f"{item['question']}\n\n{item['timestamp']}"
                ):
                    st.session_state.messages = []
                    processing_chain(item['question'], num_sources, model_choice, use_langchain, use_langgraph)
            
            if st.button("Clear history", use_container_width=True):
                clear_question_history()
                st.rerun()
        else:
            st.caption("No questions asked yet.")
        
        st.markdown('---')
        st.markdown('<div class="sidebar-section">Cache</div>', unsafe_allow_html=True)
        if st.button("Clear response cache", use_container_width=True):
            if clear_backend_cache():
                st.success("Cache cleared!")
            else:
                st.warning("Could not clear cache (API may not support this endpoint).")
        
    # Main Layout - Header
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                <path d="M12 4v16M4 12h16" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
        <h1 class="main-title">MediQuery AI</h1>
    </div>
    <p class="sub-header">Advanced Explainable Medical Intelligence</p>
    """, unsafe_allow_html=True)
    
    # Initialize session state for history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-card user-card">
                <div class="card-label">You</div>
                <div class="card-content">{sanitize_html(message["content"])}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_answer(message["content"])

    # Suggestion chips — shown only when no messages yet, above the input bar
    if not st.session_state.messages:
        SUGGESTIONS = {
            "Symptoms of Type 2 Diabetes": "Symptoms of Type 2 Diabetes",
            "Treatment for Hypertension": "Treatment for Hypertension",
            "Side effects of Dolo 650": "What are the side effects of Dolo 650",
            "Causes of acute migraine": "Causes of acute migraine",
        }
        st.caption("Try one of these questions to get started:")
        cols = st.columns(4)
        for i, (label, chip_q) in enumerate(SUGGESTIONS.items()):
            if cols[i].button(label, type="secondary", use_container_width=True):
                processing_chain(chip_q, num_sources, model_choice, use_langchain, use_langgraph)

    # Spacer so the last message is never hidden under the fixed bar
    st.markdown("<div style='height: 90px'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Fixed chat input bar — mic + text input + send, all in one pill
    # ---------------------------------------------------------------------------
    # Show transcription toast if we just converted audio
    if st.session_state.get("_voice_toast"):
        st.markdown(
            f'<div class="voice-toast">'
            f'<b>Heard:</b> {sanitize_html(st.session_state._voice_toast)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.container(key="chat_input_bar"):
        col_mic, col_form = st.columns([1, 13], gap="small")

        # ── Mic button ──────────────────────────────────────────────────────
        with col_mic:
            try:
                from audio_recorder_streamlit import audio_recorder
                audio_bytes = audio_recorder(
                    text="",
                    recording_color="#ef4444",
                    neutral_color="#94a3b8",
                    icon_name="microphone",
                    icon_size="lg",
                    pause_threshold=2.5,
                    key="mic_recorder",
                )
            except Exception as e:
                logger.warning(f"Audio recorder widget unavailable: {e}")
                audio_bytes = None

        # ── Text input + send button ────────────────────────────────────────
        with col_form:
            with st.form("chat_form", clear_on_submit=True, border=False):
                col_input, col_send = st.columns([12, 1], gap="small")
                with col_input:
                    question_typed = st.text_input(
                        "",
                        placeholder="Ask a medical question...",
                        label_visibility="collapsed",
                        key="chat_text_input",
                    )
                with col_send:
                    form_submitted = st.form_submit_button(
                        "↑",
                        use_container_width=True,
                    )

    # ── Handle audio transcription ──────────────────────────────────────────
    if audio_bytes and audio_bytes != st.session_state.get("_last_audio"):
        st.session_state["_last_audio"] = audio_bytes
        st.session_state.pop("_voice_toast", None)
        with st.spinner("Transcribing..."):
            transcribed = transcribe_audio(audio_bytes)
        if transcribed:
            st.session_state["_voice_toast"] = transcribed
            processing_chain(transcribed, num_sources, model_choice, use_langchain, use_langgraph)
        else:
            st.toast("Could not transcribe audio. Please speak clearly and try again.")

    # ── Handle typed / Enter / send ─────────────────────────────────────────
    elif form_submitted and question_typed.strip():
        st.session_state.pop("_voice_toast", None)
        processing_chain(question_typed.strip(), num_sources, model_choice, use_langchain, use_langgraph)


if __name__ == "__main__":
    main()
