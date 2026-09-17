"""
app.py - Simple Streamlit web app for the RAG Chat Bot.
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from rag_bot import RAGBot

st.set_page_config(page_title="RAG Chat", page_icon="🤖")


@st.cache_resource
def get_bot():
    """Create bot with lazy Ollama connection."""
    return RAGBot(llm_backend=None)


if "messages" not in st.session_state:
    st.session_state.messages = []

bot = get_bot()

# Initialize Ollama once
if not bot._ollama and not bot._generator:
    bot.llm_backend = "ollama"
    bot.llm_model = "phi3"
    bot._init_llm()

# --- Sidebar: upload + stats ---
with st.sidebar:
    st.title("RAG Chat")
    st.caption("Local | Free | No API keys")
    st.divider()

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "md"],
        key="uploader",
    )

    if uploaded and bot:
        existing = set(c.source for c in bot.chunks)
        if uploaded.name not in existing:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner(f"Processing {uploaded.name}..."):
                    result = bot.add_document(tmp_path, uploaded.name)
                st.success(f"Added: {result['num_chunks']} chunks")
                st.session_state.messages = []
            except Exception as e:
                st.error(str(e))
            finally:
                os.unlink(tmp_path)

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.divider()
    stats = bot.get_stats()
    st.write(f"Chunks: {stats['num_chunks']}")
    st.write(f"Sources: {stats['num_sources']}")

# --- Main: chat UI ---
st.title("RAG Chat Bot")
st.caption("Upload a document, then ask a question!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

prompt = st.chat_input("Ask about your documents...")
if prompt and bot.get_stats()["num_chunks"] > 0:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = bot.query(prompt)
                st.markdown(result.answer)
                if result.sources:
                    st.caption(f"Sources: {', '.join(result.sources)}")
                st.session_state.messages.append({
                    "role": "assistant", "content": result.answer,
                    "sources": result.sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant", "content": f"Error: {e}",
                })
elif prompt:
    st.info("Please upload a document first.", icon="📎")