"""
Streamlit frontend for Healthcare QA Chatbot.
Clean chat UI using native st.chat_message + st.chat_input.
"""
import os
import json
import logging
import streamlit as st
import requests
import time
import html
from typing import Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def sanitize_html(text: str) -> str:
    return html.escape(str(text)) if text else ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MediQuery | Medical AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external stylesheet
import pathlib as _pathlib
_css_path = _pathlib.Path(__file__).parent / "static" / "style.css"
_css = _css_path.read_text() if _css_path.exists() else ""
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def fetch_available_models() -> dict:
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch models: {e}")
    return {
        "tinyllama": {
            "display_name": "TinyLlama 1.1B",
            "description": "Lightweight, fast responses",
            "parameters": "1.1B",
            "requires_gpu": False,
            "loaded": False,
        },
    }


def ask_question(
    question: str,
    num_sources: int = 5,
    model_choice: str = None,
    use_langchain: bool = False,
    use_langgraph: bool = False,
    session_id: str = None,
) -> Optional[dict]:
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

        response = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.error("System is initializing or busy. Please try again in a moment.")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Make sure it is running on port 8000.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be busy — please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def clear_backend_cache() -> bool:
    try:
        resp = requests.post(f"{API_URL}/clear-cache", timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")
        return False


def submit_feedback(
    response_id: str,
    rating: Optional[int] = None,
    was_helpful: Optional[bool] = None,
    was_accurate: Optional[bool] = None,
    was_safe: Optional[bool] = None,
    feedback_text: Optional[str] = None,
) -> bool:
    payload = {"response_id": response_id, "session_id": st.session_state.get("session_id")}
    if rating is not None:       payload["rating"] = rating
    if was_helpful is not None:  payload["was_helpful"] = was_helpful
    if was_accurate is not None: payload["was_accurate"] = was_accurate
    if was_safe is not None:     payload["was_safe"] = was_safe
    if feedback_text:            payload["feedback_text"] = feedback_text
    try:
        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=15)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Feedback submission failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Answer rendering helpers
# ---------------------------------------------------------------------------
def display_confidence(confidence: dict, breakdown: dict = None):
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0)
    explanation = confidence.get("explanation", "")
    pct = int(score * 100)

    badge_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
    badge_labels = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}
    color = badge_colors.get(level, "#f59e0b")
    label = badge_labels.get(level, "Medium confidence")

    st.markdown(
        f"""
        <div class="conf-wrap">
          <div class="conf-header">
            <span class="conf-label">Confidence</span>
            <span class="conf-badge" style="background:{color}18;color:{color};">
              {label}&nbsp;&nbsp;{pct}%
            </span>
          </div>
          <div class="conf-track">
            <div class="conf-fill {level}" style="width:{pct}%"></div>
          </div>
          <div class="conf-desc">{sanitize_html(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if breakdown:
        with st.expander("XAI Signal Breakdown"):
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
            bar_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}
            for key, lbl in signal_labels.items():
                val = breakdown.get(key, 0.0)
                w = weights.get(signal_map.get(key, ""), 0.0)
                p = int(val * 100)
                lvl = "high" if val >= 0.75 else ("medium" if val >= 0.45 else "low")
                bc = bar_colors[lvl]
                st.markdown(
                    f"""
                    <div style="margin-bottom:11px;">
                      <div style="display:flex;justify-content:space-between;
                                  font-size:0.85rem;margin-bottom:4px;">
                        <span>{lbl}</span>
                        <span style="color:{bc};font-weight:600;font-family:'JetBrains Mono',monospace;">
                          {p}%<span style="opacity:.5;font-weight:400;"> w={w:.0%}</span>
                        </span>
                      </div>
                      <div style="height:5px;background:rgba(128,128,128,.16);
                                  border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{p}%;background:{bc};border-radius:3px;"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def display_sources(sources: list):
    if not sources:
        st.info("No specific sources retrieved — response based on general medical knowledge.")
        return

    st.markdown(f"**{len(sources)} verified source{'s' if len(sources) != 1 else ''}**")
    for source in sources:
        st.markdown(
            f"""
            <div class="src-card">
              <span class="src-score">Match&nbsp;{source['score']:.0%}</span>
              <div class="src-text">{sanitize_html(source['content'][:200])}…</div>
              <div class="src-meta">{sanitize_html(source['source'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View full content"):
            st.write(source["content"])
            if source.get("url"):
                st.markdown(f"[View original source]({source['url']})")


def display_attributions(attributions: list):
    verified = [a for a in (attributions or []) if a.get("source") != "Unsupported"]
    if not verified:
        return
    with st.expander("Claim Verification"):
        for attr in verified:
            st.markdown(
                f"""
                <div class="attr-card">
                  <div class="attr-claim">"{sanitize_html(attr['claim'])}"</div>
                  <div class="attr-source">
                    Verified by {sanitize_html(attr['source'])} &middot; {attr['similarity']:.0%} match
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_answer(result: dict):
    """Render a full API result inside a st.chat_message('assistant') block."""
    safety = result.get("safety")

    # Emergency banner
    if safety and safety.get("is_emergency"):
        st.markdown(
            f'<div class="emergency">'
            f'{sanitize_html(safety.get("emergency_message", "Please seek immediate medical attention."))}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if safety and safety.get("drug_warnings"):
        for w in safety["drug_warnings"]:
            st.warning(w)

    # Main answer (rendered as markdown — preserves formatting)
    st.markdown(result.get("answer", "No answer was generated. Please try again."))

    # Meta line
    meta_parts = []
    if result.get("model_used"):
        meta_parts.append(result["model_used"])
    if result.get("pipeline_used"):
        meta_parts.append(result["pipeline_used"])
    if result.get("latency_ms"):
        meta_parts.append(f"{result['latency_ms']:.0f} ms")
    elif result.get("elapsed_time"):
        meta_parts.append(f"{result['elapsed_time']:.2f} s")
    if meta_parts:
        st.markdown(
            f'<div class="meta-info">{" · ".join(sanitize_html(p) for p in meta_parts)}</div>',
            unsafe_allow_html=True,
        )

    # Non-safe level badge
    if safety and safety.get("level") and safety["level"] != "safe":
        level_colors = {"caution": "#f59e0b", "blocked": "#ef4444", "emergency": "#dc2626"}
        bc = level_colors.get(safety["level"], "#888")
        st.markdown(
            f'<span style="background:{bc}18;color:{bc};padding:2px 10px;'
            f'border-radius:4px;font-size:0.75rem;font-weight:600;">'
            f'Safety: {safety["level"].title()}</span>',
            unsafe_allow_html=True,
        )

    # Confidence bar (always visible)
    display_confidence(
        result.get("confidence", {"score": 0.0, "level": "unknown"}),
        result.get("confidence_breakdown"),
    )

    # Key medical terms
    answer_text = result.get("answer", "").lower()
    q_text = result.get("question", "").lower()
    medical_terms = [
        "diabetes", "hypertension", "symptoms", "treatment", "diagnosis",
        "pain", "blood", "heart", "disease", "medication", "chronic",
        "acute", "fever", "infection", "pressure", "sugar", "insulin",
    ]
    found_terms = [t for t in medical_terms if t in answer_text or t in q_text]
    if found_terms:
        with st.expander("Key Medical Terms"):
            st.markdown(
                "".join(f'<span class="kterm">{t}</span>' for t in found_terms[:8]),
                unsafe_allow_html=True,
            )

    # AI reasoning
    if result.get("rationale"):
        with st.expander("AI Reasoning"):
            st.markdown(result["rationale"])

    # Sources
    if result.get("sources") is not None:
        with st.expander(f"Sources & References ({len(result['sources'])})"):
            display_sources(result["sources"])

    # Claim attribution
    display_attributions(result.get("attributions", []))

    # Hallucination analysis
    hal = result.get("hallucination")
    if hal:
        with st.expander("Hallucination Analysis"):
            score = hal.get("score", 0.0)
            has_hal = hal.get("has_hallucination", False)
            hal_type = hal.get("type", "none")
            explanation = hal.get("explanation", "")
            med_flags = hal.get("medical_accuracy_flags", [])
            sc = "#ef4444" if has_hal else "#10b981"
            sl = f"Detected ({hal_type})" if has_hal else "Clean"
            st.markdown(
                f'<span class="hal-badge" style="background:{sc}18;color:{sc};">{sl}</span>'
                f'<span style="font-size:0.85rem;opacity:0.6;">'
                f'Hallucination risk: <b>{int(score * 100)}%</b></span>',
                unsafe_allow_html=True,
            )
            if explanation:
                st.caption(explanation)
            if med_flags:
                st.warning("Medical accuracy flags: " + "; ".join(med_flags))

    # Disclaimer
    disclaimer_text = result.get(
        "disclaimer",
        "This information is for educational purposes only and is NOT a substitute "
        "for professional medical advice. Always consult a qualified healthcare provider.",
    )
    st.markdown(
        f'<div class="disclaimer">'
        f'<div class="disclaimer-title">Medical Disclaimer</div>'
        f'<div class="disclaimer-text">{sanitize_html(disclaimer_text)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Feedback
    response_id = result.get("response_id")
    if response_id:
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = {}
        submitted = st.session_state.feedback_submitted.get(response_id)
        if submitted:
            st.caption(f"✓ Feedback recorded: {submitted}")
        else:
            st.caption("Was this answer medically accurate?")
            c1, c2, _ = st.columns([1, 1, 5])
            if c1.button("👍 Accurate", key=f"fb_yes_{response_id}", use_container_width=True):
                if submit_feedback(response_id, rating=5, was_helpful=True, was_accurate=True, was_safe=True):
                    st.session_state.feedback_submitted[response_id] = "Accurate"
                    st.rerun()
                else:
                    st.warning("Could not submit feedback — check the /feedback endpoint.")
            if c2.button("👎 Inaccurate", key=f"fb_no_{response_id}", use_container_width=True):
                if submit_feedback(response_id, rating=1, was_helpful=False, was_accurate=False, was_safe=True):
                    st.session_state.feedback_submitted[response_id] = "Inaccurate"
                    st.rerun()
                else:
                    st.warning("Could not submit feedback — check the /feedback endpoint.")


# ---------------------------------------------------------------------------
# Question history persistence
# ---------------------------------------------------------------------------
HISTORY_FILE = Path("data/question_history.json")


def load_question_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_question_history(history: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")


def add_to_history(question: str):
    history = load_question_history()
    history = [h for h in history if h["question"] != question]
    history.insert(0, {"question": question, "timestamp": datetime.now().strftime("%b %d, %I:%M %p")})
    save_question_history(history[:20])


def clear_question_history():
    save_question_history([])


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def processing_chain(
    question: str,
    num_sources: int,
    model_choice: str = None,
    use_langchain: bool = False,
    use_langgraph: bool = False,
):
    """Submit question, show loading state, append result to session state, rerun."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="⚕️"):
        with st.status("Searching medical knowledge base…", expanded=True) as status:
            start = time.time()
            result = ask_question(
                question,
                num_sources,
                model_choice=model_choice,
                use_langchain=use_langchain,
                use_langgraph=use_langgraph,
                session_id=st.session_state.get("session_id"),
            )
            elapsed = time.time() - start

        if result:
            result["elapsed_time"] = elapsed
            if result.get("session_id"):
                st.session_state.session_id = result["session_id"]
                try:
                    st.query_params["sid"] = result["session_id"]
                except Exception:
                    pass
            status.update(label="Response ready", state="complete", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": result})
            add_to_history(question)
            st.rerun()
        else:
            status.update(label="Error generating response", state="error", expanded=False)
            st.toast("Failed to get a response. Is the API server running?", icon="⚠️")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── Session state init ──────────────────────────────────────────────────
    if "session_id" not in st.session_state:
        saved_sid = st.query_params.get("sid")
        st.session_state.session_id = saved_sid if saved_sid else None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚕️ MediQuery")
        st.caption("Explainable Medical AI · RAG + LLM")
        st.divider()

        if st.button("＋  New conversation", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

        # Model
        st.markdown('<span class="sidebar-sep">Model</span>', unsafe_allow_html=True)
        available_models = fetch_available_models()
        model_display_names = {
            k: f"{v.get('display_name', k)} ({v.get('parameters', '?')})"
            for k, v in available_models.items()
        }
        if len(available_models) > 1:
            selected_label = st.selectbox(
                "LLM model",
                options=list(model_display_names.values()),
                index=0,
                label_visibility="collapsed",
                help="Select the LLM model for generating answers",
            )
            model_choice = next(k for k, v in model_display_names.items() if v == selected_label)
        else:
            model_choice = list(available_models.keys())[0]
            info = available_models.get(model_choice, {})
            st.caption(f"{info.get('display_name', 'TinyLlama 1.1B')} · {info.get('parameters', '1.1B')}")

        # Pipeline
        st.markdown('<span class="sidebar-sep">Pipeline</span>', unsafe_allow_html=True)
        pipeline_choice = st.radio(
            "Pipeline",
            ["Standard", "LangChain (LCEL)", "LangGraph (Self-Correcting)"],
            label_visibility="collapsed",
            help="Standard is recommended for most queries",
        )
        use_langchain = pipeline_choice == "LangChain (LCEL)"
        use_langgraph = pipeline_choice == "LangGraph (Self-Correcting)"

        # Sources slider
        st.markdown('<span class="sidebar-sep">References</span>', unsafe_allow_html=True)
        num_sources = st.slider(
            "Sources",
            min_value=2,
            max_value=10,
            value=3,
            label_visibility="collapsed",
            help="More sources = slower but more thorough",
        )


        # Question history
        st.divider()
        st.markdown('<span class="sidebar-sep">Recent Questions</span>', unsafe_allow_html=True)
        history = load_question_history()
        if history:
            for i, item in enumerate(history[:10]):
                q_display = item["question"][:48] + ("…" if len(item["question"]) > 48 else "")
                if st.button(
                    q_display,
                    key=f"hist_{i}",
                    use_container_width=True,
                    help=f"{item['question']}\n\n{item['timestamp']}",
                ):
                    st.session_state.messages = []
                    processing_chain(item["question"], num_sources, model_choice, use_langchain, use_langgraph)
            if st.button("Clear history", use_container_width=True):
                clear_question_history()
                st.rerun()
        else:
            st.caption("No questions yet.")

        st.divider()
        if st.button("Clear response cache", use_container_width=True):
            if clear_backend_cache():
                st.success("Cache cleared!")
            else:
                st.warning("Could not clear cache (API may not support /clear-cache).")

    # ── Main chat area ───────────────────────────────────────────────────────
    if not st.session_state.messages:
        # Welcome screen
        st.markdown(
            """
            <div class="welcome-wrap">
              <div class="welcome-logo">⚕️</div>
              <div class="welcome-title">MediQuery AI</div>
              <div class="welcome-sub">
                Ask me about symptoms, conditions, treatments, or medications.
                I search verified medical literature to answer your questions with
                explainability and source attribution.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggestion chips (2-column grid, centred)
        SUGGESTIONS = [
            ("Symptoms of Type 2 Diabetes",    "What are the symptoms of Type 2 Diabetes?"),
            ("Treatment for Hypertension",      "What is the treatment for Hypertension?"),
            ("Side effects of Dolo 650",        "What are the side effects of Dolo 650?"),
            ("Causes of acute migraine",        "What causes acute migraine?"),
        ]
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            c1, c2 = st.columns(2)
            for i, (label, prompt) in enumerate(SUGGESTIONS):
                col = c1 if i % 2 == 0 else c2
                if col.button(label, key=f"chip_{i}", use_container_width=True):
                    processing_chain(prompt, num_sources, model_choice, use_langchain, use_langgraph)

    else:
        # Render full chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="⚕️"):
                    render_answer(message["content"])

    # ── Input bar: mic + text + send ────────────────────────────────────────
    # Spacer so last message isn't hidden behind the input bar
    st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)

    # ── Chat input (pinned to bottom by Streamlit) ───────────────────────────
    if prompt := st.chat_input("Ask a medical question…"):
        processing_chain(prompt.strip(), num_sources, model_choice, use_langchain, use_langgraph)


if __name__ == "__main__":
    main()
