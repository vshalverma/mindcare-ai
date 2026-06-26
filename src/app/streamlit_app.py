"""Streamlit front-end for mindcare-ai.

Run:
    streamlit run src/app/streamlit_app.py

What this app shows:
  - A simple chat box that calls ChatEngine.reply()
  - The model's predicted emotion + confidence (for transparency)
  - A crisis-safety banner that appears whenever the classifier OR the
    keyword gate flags the input. This is the most important UI element.

Why Streamlit: fastest path to a usable demo, fully self-hosted, no JS
front-end to maintain. For production we'd replace this with a real
chat framework, but Streamlit is fine for a portfolio-grade demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src.` imports work when Streamlit is launched from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.inference.chat_engine import (  # noqa: E402
    CRISIS_RESOURCES,
    ChatEngine,
)


# ---------------------------------------------------------------------------
# Streamlit page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="mindcare-ai",
    page_icon="💙",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Resource loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading mindcare-ai model...")
def _load_engine() -> ChatEngine:
    """Load the ChatEngine once and cache it for the lifetime of the session."""
    return ChatEngine()


# ---------------------------------------------------------------------------
# Sidebar — model info + disclaimers
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💙 mindcare-ai")
    st.markdown(
        "A demo mental-health companion. **Not a substitute for professional "
        "care.** If you're in crisis, please reach out to a hotline below."
    )
    st.divider()
    st.subheader("Crisis resources")
    for code, info in CRISIS_RESOURCES.items():
        with st.expander(f"{code} — {info['name']}"):
            st.markdown(f"- 📞 **Call:** {info['call']}")
            st.markdown(f"- 💬 **Text:** {info['text']}")
            st.markdown(f"- 🌐 **Site:** [{info['site']}]({info['site']})")
    st.divider()
    st.caption(
        "Built with DistilBERT (multi-task emotion + crisis classifier) "
        "trained on GoEmotions, EmpatheticDialogues, and a Reddit "
        "suicide/depression corpus."
    )


# ---------------------------------------------------------------------------
# Main chat panel
# ---------------------------------------------------------------------------

st.header("How are you feeling today?")

# Initialise chat history on first load.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm here to listen. Tell me what's on your mind, "
                "and I'll do my best to understand."
            ),
        }
    ]


def _render_history() -> None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("crisis_flag"):
                _render_crisis_banner()


def _render_crisis_banner() -> None:
    st.error(
        "**It sounds like you might be going through a really difficult "
        "time.** Please consider reaching out to a crisis line:\n\n"
        "- 🇺🇸 **988 Suicide & Crisis Lifeline:** call or text **988**\n"
        "- 🇮🇳 **iCall (India):** call **+91-9152987821**\n\n"
        "You don't have to face this alone.",
        icon="🆘",
    )


_render_history()

# User input -> classify -> reply.
if prompt := st.chat_input("Type what's on your mind..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            engine = _load_engine()
            reply = engine.reply(prompt)

        st.markdown(reply.text)
        with st.expander("Why this response? (model details)"):
            col1, col2 = st.columns(2)
            col1.metric(
                "Predicted emotion",
                reply.emotion,
                f"{reply.emotion_confidence:.0%} confidence",
            )
            col2.metric(
                "Crisis probability",
                f"{reply.crisis_prob:.0%}",
                f"trigger: {reply.crisis_reason}",
            )

        if reply.crisis_flag:
            _render_crisis_banner()

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": reply.text,
            "crisis_flag": reply.crisis_flag,
        }
    )