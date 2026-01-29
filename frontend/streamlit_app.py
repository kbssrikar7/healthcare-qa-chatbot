"""
Streamlit frontend for Healthcare QA Chatbot.
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
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Chat Message Cards */
    .user-card {
        background-color: #eff6ff;
        color: #1e293b;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .bot-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Confidence Meters */
    .confidence-wrapper {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .meter-bg {
        width: 100%;
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .meter-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-in-out;
    }
    
    .confidence-high .meter-fill { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .confidence-medium .meter-fill { background: linear-gradient(90deg, #eab308, #ca8a04); }
    .confidence-low .meter-fill { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    /* Source Cards */
    .source-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .source-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.2s;
        height: 100%;
    }
    
    .source-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1);
    }
    
    .source-score {
        font-size: 0.75rem;
        font-weight: bold;
        color: #3b82f6;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    /* Disclaimer */
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #f59e0b;
        color: #64748b;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    
    /* Attribution */
    .attribution-card {
        background: #f1f5f9;
        border-left: 3px solid #6366f1;
        padding: 0.75rem;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        color: #1e293b !important;
    }
    
    .attribution-card strong, .attribution-card span {
        color: #0f172a !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.8rem;
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
            st.error("🚧 System is initializing or busy. Please try again in a moment.")
            return None
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def display_confidence(confidence: dict):
    """Display animated confidence meter."""
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0)
    explanation = confidence.get("explanation", "")
    
    color_class = f"confidence-{level}"
    width_percent = int(score * 100)
    
    st.markdown(f"""
    <div class="confidence-wrapper {color_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 0.9rem; color: #334155;">AI Confidence</span>
            <span style="font-weight: 700; color: #334155;">{width_percent}%</span>
        </div>
        <div class="meter-bg">
            <div class="meter-fill" style="width: {width_percent}%"></div>
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources(sources: list):
    """Display interactive source cards."""
    if not sources:
        st.info("ℹ️ No specific medical sources were used for this answer. The AI used its general medical knowledge.")
        return

    st.markdown("### 📚 Verified Sources")
    
    # Create grid of sources
    cols = st.columns(min(3, len(sources)))
    
    for i, source in enumerate(sources):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="source-card">
                <div class="source-score">Match: {source['score']:.0%}</div>
                <div style="font-size: 0.9rem; color: #334155; margin-bottom: 0.5rem; height: 80px; overflow: hidden; text-overflow: ellipsis;">
                    {source['content'][:150]}...
                </div>
                <div style="font-size: 0.75rem; color: #64748b; font-style: italic;">
                    {source['source']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Details"):
                st.write(source['content'])
                if source.get('url'):
                    st.markdown(f"[🔗 Original Source]({source['url']})")

def display_attributions(attributions: list):
    """Display attributions for specific claims."""
    if not attributions:
        return
        
    with st.expander("🔍 Integrity Check: Claim Verification"):
        for attr in attributions:
            if attr['source'] != "Unsupported":
                st.markdown(f"""
                <div class="attribution-card">
                    <strong>Claim:</strong> "{attr['claim']}"<br/>
                    <span style="color: #6366f1">Verified by:</span> {attr['source']} ({attr['similarity']:.0%} match)
                </div>
                """, unsafe_allow_html=True)

def render_answer(result):
    """Render the full answer card."""
    # Bot Answer
    st.markdown(f"""
    <div class="bot-card">
        🤖 <strong>MediQuery:</strong><br>
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Display Rationale if available
    if result.get('rationale'):
        with st.expander("🧠 Why this answer? (AI Reasoning)", expanded=False):
            st.markdown(f"""
            <div style="background: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; border-radius: 4px; color: #166534;">
                {result['rationale']}
            </div>
            """, unsafe_allow_html=True)
    
    # Token Importance Visualization (XAI Feature)
    if result.get('sources'):
        with st.expander("🔬 Key Terms Analysis", expanded=False):
            # Extract important terms from question and answer
            question_words = result.get('question', '').lower().split()
            answer_text = result.get('answer', '').lower()
            
            # Highlight medical terms that appear in both
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
                    f'<span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 4px 12px; border-radius: 16px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500; box-shadow: 0 2px 4px rgba(99,102,241,0.3);">{term}</span>'
                    for term in found_terms[:8]
                ])
                st.markdown(term_html, unsafe_allow_html=True)
            else:
                st.info("No specific medical terms highlighted for this query.")
    
    # Metrics & Sources
    display_confidence(result['confidence'])
    display_sources(result['sources'])
    display_attributions(result.get('attributions', []))
    
    # Disclaimer
    elapsed_html = ""
    if result.get('elapsed_time'):
        elapsed_html = f"<br/><span style='font-size:0.75rem; color:#94a3b8'>Generated in {result['elapsed_time']:.2f}s</span>"
        
    st.markdown(f"""<div class="disclaimer-box">
        <span>⚠️ <strong>Medical Disclaimer:</strong> {result['disclaimer']}</span>
        {elapsed_html}
    </div>""", unsafe_allow_html=True)

def processing_chain(question, num_sources):
    """Process a question and add logic to history."""
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Use st.status for better UX
    with st.status("🧠 Analyzing medical literature...", expanded=True) as status:
        st.write("🔍 Searching knowledge base...")
        start_time = time.time()
        
        # Real API call
        result = ask_question(question, num_sources)
        elapsed = time.time() - start_time
        
        if result:
            result['elapsed_time'] = elapsed
            st.write("✅ Response generated!")
            status.update(label="Response generated!", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()
        else:
            status.update(label="Error generating response", state="error")

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=60)
        st.title("MediQuery")
        st.markdown("---")
        
        st.subheader("⚙️ Analysis Depth")
        num_sources = st.slider("References to Analyze", 2, 10, 3, help="More sources = slower but more thorough.")
        
        st.markdown("---")
        st.info("""
        **System Status**
        🟢 AI Core: Online
        🟢 Knowledge Base: Active
        """)
        
    # Main Layout
    st.markdown('<div class="main-header">MediQuery AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Explainable Medical Intelligence</div>', unsafe_allow_html=True)
    
    # Initialize session state for history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-card">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            render_answer(message["content"])

    # Chat Input 
    if question := st.chat_input("Ask a medical question (e.g., 'What are type 2 diabetes symptoms?')"):
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
    
    # Footer
    st.markdown('<div class="footer">Built with ❤️ for Healthcare AI • Using TinyLlama & BioMistral</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
