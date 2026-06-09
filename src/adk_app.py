"""User chat UI backed by the Google ADK agent (adk_agent.py).

Parallel to app.py (the LangChain chat). Same UX, same shared watcher/vector
store — only the answering backend differs. Run on its own port so both can run
side by side:  streamlit run src/adk_app.py   (or: python run.py --adk)
"""
import os
import sys
import threading
from pathlib import Path

# Allow imports from src/ when running as `streamlit run src/adk_app.py`
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from ingest import start_watcher, ingest_existing
from adk_agent import build_app, ask

load_dotenv()

st.set_page_config(page_title="RAG Test (ADK)", page_icon="🧪", layout="centered")

# ADMIN_URL is set by run.py so the user can jump to the admin console.
_admin_url = os.getenv("ADMIN_URL")
if _admin_url:
    st.sidebar.link_button("🛠️ Open Admin", _admin_url, use_container_width=True)

st.title("RAG Test — ADK")
st.caption("Experimental Google ADK backend. Drop PDFs/TXT into /docs — ingested automatically.")


# Shares the same watcher + vector store as the LangChain app (just a different
# process). cache_resource keeps each created once per process.
@st.cache_resource(show_spinner="Starting watcher…")
def init_watcher():
    threading.Thread(target=ingest_existing, daemon=True).start()
    return start_watcher()


@st.cache_resource(show_spinner="Initialising ADK agent…")
def get_app():
    return build_app()


init_watcher()
app = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(app, prompt, history=st.session_state.messages[:-1])
        st.write(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
