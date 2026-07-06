"""
app.py
Stage 4 - Streamlit Chat UI (InternGrow Task 2, base version)

Run this with:
    streamlit run src/app.py

This provides a simple chat interface on top of the TF-IDF + Cosine
Similarity matching engine built in match_engine.py.
"""

import sys
from pathlib import Path

import streamlit as st

# Make sure sibling modules (match_engine.py, preprocess.py) are importable
# regardless of which folder you run this command from.
sys.path.append(str(Path(__file__).resolve().parent))
from match_engine import FAQMatcher

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"

st.set_page_config(page_title="IT Helpdesk FAQ Chatbot", page_icon="🤖")


@st.cache_resource
def load_matcher():
    return FAQMatcher(str(DATA_PATH), similarity_threshold=0.3)


matcher = load_matcher()

st.title("🤖 IT Helpdesk FAQ Chatbot")
st.caption("InternGrow AI Track — Task 2 (Base version: TF-IDF + Cosine Similarity)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your IT Helpdesk assistant. Ask me about passwords, "
                "VPN, WiFi, software installs, hardware requests, tickets, "
                "security, or printers."
            ),
        }
    ]

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box (always at the bottom, like WhatsApp/ChatGPT)
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get chatbot's answer
    result = matcher.get_best_match(user_input)
    answer = result["answer"]

    # Show and store the assistant's message
    with st.chat_message("assistant"):
        st.write(answer)
        if result["matched"]:
            st.caption(f"Matched category: {result['category']}  |  Similarity score: {result['score']}")
        else:
            st.caption(f"No confident match found (score: {result['score']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.write(
        "This is the base version of the FAQ chatbot using classic "
        "TF-IDF vectorization + Cosine Similarity matching."
    )
    st.write("Try asking things like:")
    st.markdown(
        "- *How do I reset my password?*\n"
        "- *My laptop battery drains fast*\n"
        "- *WiFi is really slow*\n"
        "- *how to connect vpn*"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
