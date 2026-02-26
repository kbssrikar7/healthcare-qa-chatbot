"""
Streamlit frontend for Healthcare QA Chatbot.
Professional, modern UI design without emoji clutter.
"""
import os
import json
import streamlit as st
import requests
import time
import html
from typing import Optional
from pathlib import Path
from datetime import datetime

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
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --user-card-bg: linear-gradient(135deg, #1e293b 0%, #1e2738 100%);
        --user-card-border: #334155;
        --user-card-label: #60a5fa;
        --disclaimer-bg: linear-gradient(135deg, #2d2305 0%, #3d3010 100%);
        --disclaimer-border: #a16207;
        --disclaimer-title: #fbbf24;
        --disclaimer-text: #fde68a;
        --disclaimer-meta: #d97706;
        --reasoning-bg: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
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
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --user-card-bg: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        --user-card-border: #bfdbfe;
        --user-card-label: #2563eb;
        --disclaimer-bg: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        --disclaimer-border: var(--warning);
        --disclaimer-title: #b45309;
        --disclaimer-text: #78350f;
        --disclaimer-meta: #a16207;
        --reasoning-bg: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        --reasoning-text: #166534;
        --attr-bg: #f1f5f9;
        --source-badge-bg: #eef2ff;
        --source-badge-text: var(--primary);
        --info-bg: #f0f9ff;
        --info-border: #bae6fd;
        --info-text: #0369a1;
    }"""  

# Custom CSS - Premium Design System
st.markdown("<style>\n    /* ========== CSS Variables & Theme ========== */\n    " + get_theme_css() + """

    /* ========== Typography ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* ========== Header Styles ========== */
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }
    
    .app-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-md);
    }
    
    .app-logo svg {
        width: 28px;
        height: 28px;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
        padding-left: 64px;
    }

    /* ========== Chat Cards ========== */
    .chat-card {
        padding: 1.25rem 1.5rem;
        border-radius: var(--radius-lg);
        margin-bottom: 1rem;
        position: relative;
    }
    
    .user-card {
        background: var(--user-card-bg);
        border: 1px solid var(--user-card-border);
        margin-left: 48px;
    }
    
    .user-card::before {
        content: '';
        position: absolute;
        left: -40px;
        top: 12px;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .user-card::after {
        content: 'U';
        position: absolute;
        left: -40px;
        top: 12px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .bot-card {
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md);
        margin-left: 48px;
    }
    
    .bot-card::before {
        content: '';
        position: absolute;
        left: -40px;
        top: 12px;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        border-radius: var(--radius-sm);
    }
    
    .bot-card::after {
        content: 'M';
        position: absolute;
        left: -40px;
        top: 12px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.875rem;
    }
    
    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .user-card .card-label {
        color: var(--user-card-label);
    }
    
    .bot-card .card-label {
        color: var(--primary);
    }
    
    .card-content {
        color: var(--text-primary);
        line-height: 1.7;
    }

    /* ========== Confidence Meter ========== */
    .confidence-container {
        background: var(--surface-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    .confidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .confidence-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .confidence-value {
        font-size: 1.25rem;
        font-weight: 700;
    }
    
    .confidence-value.high { color: var(--success); }
    .confidence-value.medium { color: var(--warning); }
    .confidence-value.low { color: var(--danger); }
    
    .confidence-track {
        width: 100%;
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .confidence-fill.high { background: linear-gradient(90deg, #10b981, #059669); }
    .confidence-fill.medium { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .confidence-fill.low { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    .confidence-desc {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.75rem;
    }

    /* ========== Source Cards ========== */
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        display: inline-block;
    }
    
    .source-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .source-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        transition: all 0.2s ease;
        height: 100%;
    }
    
    .source-card:hover {
        border-color: var(--primary-light);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .source-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--source-badge-text);
        background: var(--source-badge-bg);
        padding: 4px 10px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.75rem;
    }
    
    .source-text {
        font-size: 0.875rem;
        color: var(--text-primary);
        line-height: 1.6;
        margin-bottom: 0.75rem;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .source-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-style: italic;
    }

    /* ========== Disclaimer Box ========== */
    .disclaimer-box {
        background: var(--disclaimer-bg);
        border-left: 4px solid var(--disclaimer-border);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 1rem 1.25rem;
        margin-top: 1.5rem;
    }
    
    .disclaimer-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--disclaimer-title);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .disclaimer-text {
        font-size: 0.85rem;
        color: var(--disclaimer-text);
        line-height: 1.6;
    }
    
    .disclaimer-meta {
        font-size: 0.75rem;
        color: var(--disclaimer-meta);
        margin-top: 0.5rem;
    }

    /* ========== Attribution Cards ========== */
    .attribution-card {
        background: var(--attr-bg);
        border-left: 3px solid var(--primary);
        padding: 0.875rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }
    
    .attribution-claim {
        font-size: 0.9rem;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .attribution-source {
        font-size: 0.8rem;
        color: var(--primary);
        font-weight: 500;
    }

    /* ========== Reasoning Box ========== */
    .reasoning-box {
        background: var(--reasoning-bg);
        border-left: 4px solid var(--success);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 1rem 1.25rem;
        color: var(--reasoning-text);
        line-height: 1.7;
    }

    /* ========== Key Terms ========== */
    .key-term {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.25);
    }

    /* ========== Loading States ========== */
    .loading-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        background: var(--primary);
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* ========== Example Buttons ========== */
    .stButton > button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stButton > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }

    /* ========== Sidebar Styles ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--surface-elevated) 0%, var(--surface) 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary);
    }
    
    .sidebar-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 8px;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .sidebar-section {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* ========== Info Box ========== */
    .info-box {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-radius: var(--radius-md);
        padding: 1rem;
        color: var(--info-text);
        font-size: 0.9rem;
    }

    /* ========== Hide Streamlit Branding ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ========== Custom Expander ========== */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)


def fetch_available_models() -> dict:
    """Fetch available models from API."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # Fallback
    return {
        "tinyllama": {"display_name": "TinyLlama 1.1B", "description": "Lightweight, fast responses", "parameters": "1.1B", "requires_gpu": False, "loaded": False},
        "biomistral-7b": {"display_name": "BioMistral 7B", "description": "Medical-specialized, higher accuracy", "parameters": "7B", "requires_gpu": True, "loaded": False},
        "airllm-mistral-7b": {"display_name": "AirLLM Mistral 7B", "description": "AirLLM sharded inference", "parameters": "7B", "requires_gpu": True, "loaded": False},
    }


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
        st.error("Cannot connect to API. Make sure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def clear_backend_cache():
    """Clear the backend response cache."""
    try:
        response = requests.post(f"{API_URL}/clear-cache", timeout=10)
        return response.status_code == 200
    except Exception:
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
    except Exception:
        return False


def display_confidence(confidence: dict):
    """Display animated confidence meter with badge."""
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0)
    explanation = confidence.get("explanation", "")
    
    width_percent = int(score * 100)
    
    # Color-coded badge based on confidence level (from improving-streamlit-design skill)
    badge_colors = {"high": "#34d399", "medium": "#fbbf24", "low": "#f87171"}
    badge_labels = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}
    badge_color = badge_colors.get(level, "#fbbf24")
    badge_label = badge_labels.get(level, "Medium confidence")
    
    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-header">
            <span class="confidence-label">AI confidence score</span>
            <span class="confidence-badge" style="background: {badge_color}; color: #1e1e2e; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">{badge_label} &middot; {width_percent}%</span>
        </div>
        <div class="confidence-track">
            <div class="confidence-fill {level}" style="width: {width_percent}%"></div>
        </div>
        <div class="confidence-desc">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)


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
        <div style="background:#dc2626; color:white; padding:16px 20px; border-radius:10px; margin-bottom:16px; font-weight:600;">
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
        <span style="background:{badge_color}; color:#1e1e2e; padding:2px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;">
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
    
    # Metrics & Sources
    display_confidence(result['confidence'])
    display_sources(result['sources'])
    display_attributions(result.get('attributions', []))
    
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
            st.caption("Was this answer helpful?")
            fb_col1, fb_col2, _ = st.columns([1, 1, 6])

            if fb_col1.button("Helpful", key=f"feedback_helpful_{response_id}", use_container_width=True):
                ok = submit_feedback(
                    response_id=response_id,
                    rating=5,
                    was_helpful=True,
                    was_accurate=True,
                    was_safe=True,
                )
                if ok:
                    st.session_state.feedback_submitted[response_id] = "Helpful"
                    st.rerun()
                st.warning("Could not submit feedback. Check backend /feedback endpoint.")

            if fb_col2.button("Needs Improvement", key=f"feedback_negative_{response_id}", use_container_width=True):
                ok = submit_feedback(
                    response_id=response_id,
                    rating=1,
                    was_helpful=False,
                    was_accurate=False,
                    was_safe=True,
                )
                if ok:
                    st.session_state.feedback_submitted[response_id] = "Needs Improvement"
                    st.rerun()
                st.warning("Could not submit feedback. Check backend /feedback endpoint.")


# ========== Question History Persistence ==========
HISTORY_FILE = Path("data/question_history.json")

def load_question_history():
    """Load question history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_question_history(history):
    """Save question history to disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

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
            st.write("Response generated successfully")
            status.update(label="Response ready", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            # Save question to persistent history
            add_to_history(question)
            st.rerun()
        else:
            status.update(label="Error generating response", state="error")


def main():
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

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
        
        st.markdown('<div class="sidebar-section">Model settings</div>', unsafe_allow_html=True)

        # Model selector
        available_models = fetch_available_models()
        model_keys = list(available_models.keys())
        model_choice = st.selectbox(
            "LLM Model",
            options=model_keys,
            format_func=lambda k: f"{available_models[k]['display_name']} ({available_models[k]['parameters']})",
            help="Select the language model for answering questions",
        )

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

    # Chat Input 
    if question := st.chat_input("Ask a medical question..."):
        processing_chain(question, num_sources, model_choice, use_langchain, use_langgraph)

    # Suggestion chips (from building-streamlit-chat-ui skill)
    if not st.session_state.messages:
        SUGGESTIONS = {
            "Symptoms of Type 2 Diabetes": "Symptoms of Type 2 Diabetes",
            "Treatment for Hypertension": "Treatment for Hypertension",
            "Side effects of Dolo 650": "What are the side effects of Dolo 650",
            "Causes of acute migraine": "Causes of acute migraine",
        }
        st.caption("Try one of these questions to get started:")
        cols = st.columns(4)
        for i, (label, question) in enumerate(SUGGESTIONS.items()):
            if cols[i].button(label, type="secondary", use_container_width=True):
                processing_chain(question, num_sources, model_choice, use_langchain, use_langgraph)


if __name__ == "__main__":
    main()
