import sys
import threading
from pathlib import Path

# Allow imports from src/ when running as `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from ingest import start_watcher, ingest_existing
from agent import build_chain, ask

load_dotenv()

st.set_page_config(page_title="RAG Test", page_icon="📚", layout="centered")
st.title("RAG Test")
st.caption("Drop PDFs or TXT files into /docs — they are ingested automatically.")


# cache_resource is process-wide (shared across sessions/tabs), so the watcher
# and chain are each created exactly once regardless of how many tabs connect.
@st.cache_resource(show_spinner="Starting watcher…")
def init_watcher():
    # ingest_existing can take a long time (quota retries, large files).
    # Run it in the background so the UI becomes usable immediately.
    threading.Thread(target=ingest_existing, daemon=True).start()
    return start_watcher()


@st.cache_resource(show_spinner="Initialising RAG chain…")
def get_chain():
    return build_chain()


init_watcher()
chain = get_chain()

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
            # Pass prior turns (everything except the question just appended) so
            # the agent can resolve follow-up references. Session-only — not
            # persisted anywhere.
            result = ask(chain, prompt, history=st.session_state.messages[:-1])
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
