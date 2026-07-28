"""
app.py
Stage 8 - Final Streamlit UI: merges the Base version (TF-IDF + Cosine
Similarity) and the RAG upgrade (Embeddings + local AI generation) into
one app, with a sidebar toggle to switch between them live.

Quick-win additions (see README for details):
  - 👍 / 👎 feedback buttons under every answer -> data/feedback_log.csv
  - Unmatched queries auto-logged                -> data/unmatched_log.csv
  - Color-coded confidence badge + progress bar for every match

Run this with:
    streamlit run src/app.py

Note: the first time you switch to RAG mode, two small free models will
download automatically (~90MB + ~300MB). After that they're cached and
load instantly.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from match_engine import FAQMatcher
from rag_pipeline import RAGChatbot
from logging_utils import log_unmatched, log_feedback

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"

st.set_page_config(page_title="IT Helpdesk FAQ Chatbot", page_icon="🤖")


@st.cache_resource
def load_basic_matcher():
    return FAQMatcher(str(DATA_PATH), similarity_threshold=0.3)


@st.cache_resource
def load_rag_chatbot():
    return RAGChatbot(str(DATA_PATH))


def confidence_badge(score: float, threshold: float) -> str:
    """Returns a short color-coded label for a similarity/confidence score."""
    if score >= threshold + 0.3:
        return "🟢 High confidence"
    elif score >= threshold:
        return "🟡 Medium confidence"
    else:
        return "🔴 Low confidence"


def render_confidence(score: float, threshold: float):
    st.caption(confidence_badge(score, threshold))
    st.progress(min(max(score, 0.0), 1.0))


def render_feedback_controls(idx: int, meta: dict):
    """Shows 👍/👎 once, then a thank-you note after the user picks one."""
    if meta.get("feedback"):
        chosen = "👍" if meta["feedback"] == "up" else "👎"
        st.caption(f"Thanks for the feedback! ({chosen})")
        return

    col1, col2, _ = st.columns([1, 1, 6])
    with col1:
        if st.button("👍", key=f"fb_up_{idx}"):
            meta["feedback"] = "up"
            log_feedback(meta["query"], meta["mode"], meta.get("matched_question"),
                         meta["answer"], meta["score"], "up")
            st.rerun()
    with col2:
        if st.button("👎", key=f"fb_down_{idx}"):
            meta["feedback"] = "down"
            log_feedback(meta["query"], meta["mode"], meta.get("matched_question"),
                         meta["answer"], meta["score"], "down")
            st.rerun()


st.title("🤖 IT Helpdesk FAQ Chatbot")
st.caption("InternGrow AI Track — Task 2 (Base version + RAG Upgrade)")

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Choose matching mode:",
        [
            "Basic (TF-IDF + Cosine Similarity)",
            "RAG (Embeddings + AI Generation)",
        ],
        index=0,
    )
    if mode.startswith("RAG"):
        st.info("First use downloads two small free models (~400MB total). "
                 "This may take a minute; it only happens once.")

    st.markdown("---")
    st.header("About")
    st.write(
        "**Basic mode** matches your question to the closest FAQ using "
        "keyword-based TF-IDF + Cosine Similarity, and returns the stored "
        "answer as-is."
    )
    st.write(
        "**RAG mode** uses AI embeddings to understand the *meaning* of "
        "your question (catches paraphrases basic mode misses), then a "
        "small local AI model rewrites the answer in a natural, "
        "conversational way."
    )
    st.markdown("---")
    st.write("Try asking:")
    st.markdown(
        "- *How do I reset my password?*\n"
        "- *my laptop battery drains fast*\n"
        "- *internet is dead at my desk*\n"
        "- *vpn is not connecting*"
    )
    st.markdown("---")
    st.caption(
        "👍/👎 feedback and unmatched questions are logged to `data/feedback_log.csv` "
        "and `data/unmatched_log.csv` — run `python3 src/review_unmatched.py` anytime "
        "to see which new FAQs to add."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your IT Helpdesk assistant. Ask me about passwords, "
                "VPN, WiFi, software installs, hardware requests, tickets, "
                "security, or printers."
            ),
            "meta": None,
        }
    ]

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            render_confidence(meta["score"], meta["threshold"])
            if meta["matched"]:
                render_feedback_controls(idx, meta)

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "meta": None})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode.startswith("Basic"):
                mode_label = "basic"
                matcher = load_basic_matcher()
                result = matcher.get_best_match(user_input)
                answer = result["answer"]
                threshold = matcher.similarity_threshold
                st.write(answer)
                if result["matched"]:
                    st.caption(f"Matched FAQ category: {result['category']}")
                else:
                    st.caption("No confident match found")
                render_confidence(result["score"], threshold)
                meta = {
                    "query": user_input,
                    "mode": mode_label,
                    "answer": answer,
                    "matched": result["matched"],
                    "matched_question": result.get("matched_question"),
                    "score": result["score"],
                    "threshold": threshold,
                    "feedback": None,
                }
                if result["matched"]:
                    render_feedback_controls(len(st.session_state.messages), meta)
                else:
                    log_unmatched(user_input, mode_label, result["score"])
            else:
                mode_label = "rag"
                chatbot = load_rag_chatbot()
                result = chatbot.answer(user_input)
                answer = result["answer"]
                threshold = chatbot.similarity_threshold
                best_score = max((s["score"] for s in result["sources"]), default=0.0)
                st.write(answer)
                if result["matched"]:
                    sources = ", ".join(s["question"] for s in result["sources"])
                    st.caption(f"Based on: {sources}")
                else:
                    st.caption("No confident match found")
                render_confidence(best_score, threshold)
                meta = {
                    "query": user_input,
                    "mode": mode_label,
                    "answer": answer,
                    "matched": result["matched"],
                    "matched_question": result["sources"][0]["question"] if result["sources"] else None,
                    "score": best_score,
                    "threshold": threshold,
                    "feedback": None,
                }
                if result["matched"]:
                    render_feedback_controls(len(st.session_state.messages), meta)
                else:
                    log_unmatched(user_input, mode_label, best_score)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})