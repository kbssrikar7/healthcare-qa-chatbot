"""
Streamlit frontend for Healthcare QA Chatbot.
Professional, modern UI design without emoji clutter.
"""
import os
import streamlit as st
import requests
import time
from typing import Optional

# Configuration - Load from environment with fallback
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="MediQuery | Advanced Medical AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%234f46e5'/><path d='M50 25v50M25 50h50' stroke='white' stroke-width='8' stroke-linecap='round'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Premium Design System
st.markdown("""
<style>
    /* ========== CSS Variables & Theme ========== */
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
    }

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
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border: 1px solid #bfdbfe;
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
        color: #2563eb;
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
        color: var(--primary);
        background: #eef2ff;
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
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid var(--warning);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 1rem 1.25rem;
        margin-top: 1.5rem;
    }
    
    .disclaimer-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #b45309;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .disclaimer-text {
        font-size: 0.85rem;
        color: #78350f;
        line-height: 1.6;
    }
    
    .disclaimer-meta {
        font-size: 0.75rem;
        color: #a16207;
        margin-top: 0.5rem;
    }

    /* ========== Attribution Cards ========== */
    .attribution-card {
        background: #f1f5f9;
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
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid var(--success);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 1rem 1.25rem;
        color: #166534;
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
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
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
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: var(--radius-md);
        padding: 1rem;
        color: #0369a1;
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


def ask_question(question: str, num_sources: int = 5) -> Optional[dict]:
    """Send question to API and get response."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "include_explanation": True,
                "num_sources": num_sources
            },
            timeout=300
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


def display_confidence(confidence: dict):
    """Display animated confidence meter."""
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0)
    explanation = confidence.get("explanation", "")
    
    width_percent = int(score * 100)
    
    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-header">
            <span class="confidence-label">AI Confidence Score</span>
            <span class="confidence-value {level}">{width_percent}%</span>
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
                <div class="source-text">{source['content'][:180]}...</div>
                <div class="source-meta">{source['source']}</div>
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
                    <div class="attribution-claim">"{attr['claim']}"</div>
                    <div class="attribution-source">Verified by: {attr['source']} ({attr['similarity']:.0%} match)</div>
                </div>
                """, unsafe_allow_html=True)


def render_answer(result):
    """Render the full answer card with professional styling."""
    # Bot Answer Card
    st.markdown(f"""
    <div class="chat-card bot-card">
        <div class="card-label">MediQuery AI</div>
        <div class="card-content">{result['answer']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Rationale if available
    if result.get('rationale'):
        with st.expander("AI Reasoning", expanded=False):
            st.markdown(f"""
            <div class="reasoning-box">
                {result['rationale']}
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
    elapsed_html = ""
    if result.get('elapsed_time'):
        elapsed_html = f"<div class='disclaimer-meta'>Generated in {result['elapsed_time']:.2f}s</div>"
        
    st.markdown(f"""
    <div class="disclaimer-box">
        <div class="disclaimer-title">Medical Disclaimer</div>
        <div class="disclaimer-text">{result['disclaimer']}</div>
        {elapsed_html}
    </div>
    """, unsafe_allow_html=True)


def processing_chain(question, num_sources):
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
        result = ask_question(question, num_sources)
        elapsed = time.time() - start_time
        
        if result:
            result['elapsed_time'] = elapsed
            st.write("Response generated successfully")
            status.update(label="Response ready", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()
        else:
            status.update(label="Error generating response", state="error")


def main():
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
        
        st.markdown('<div class="sidebar-section">Analysis Settings</div>', unsafe_allow_html=True)
        num_sources = st.slider(
            "Number of References", 
            min_value=2, 
            max_value=10, 
            value=3, 
            help="More sources = slower but more thorough analysis"
        )
        
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
                <div class="card-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_answer(message["content"])

    # Chat Input 
    if question := st.chat_input("Ask a medical question..."):
        processing_chain(question, num_sources)

    # Example Questions
    if not st.session_state.messages:
        st.markdown("### Try asking:")
        cols = st.columns(4)
        examples = [
            "Symptoms of Type 2 Diabetes", 
            "Treatment for Hypertension",
            "Side effects of lisinopril",
            "Causes of acute migraine"
        ]
        for i, ex in enumerate(examples):
            if cols[i].button(ex, type="secondary", use_container_width=True):
                processing_chain(ex, num_sources)


if __name__ == "__main__":
    main()
