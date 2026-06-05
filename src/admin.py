"""Admin console — document ingestion, isolated from the chatbot.

Run separately from app.py (its own process / port):

    streamlit run src/admin.py --server.port 8502

It deliberately imports only the ingestion layer, never `agent`/the chat UI.
Uploaded files are saved to /docs and converted into the same RAG index the
chatbot reads, so admin uploads behave exactly like dropping a file into /docs.
"""
import sys
from pathlib import Path

# Allow imports from src/ when running as `streamlit run src/admin.py`
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from ingest import ingest_upload, indexed_summary, delete_document, SUPPORTED_SUFFIXES

load_dotenv()

st.set_page_config(page_title="Admin · Document Ingest", page_icon="🛠️", layout="centered")
st.title("🛠️ Admin — Document Ingest")
st.caption(
    "Upload PDF or TXT files. They are saved to /docs and converted into the "
    "RAG index — the same path used when dropping files into /docs directly."
)

# --- Upload -----------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload documents",
    type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
    accept_multiple_files=True,
)

if uploaded and st.button("Ingest", type="primary"):
    for f in uploaded:
        with st.spinner(f"Ingesting {f.name}…"):
            try:
                result = ingest_upload(f.name, f.getvalue())
            except Exception as exc:
                st.error(f"{f.name}: {exc}")
                continue
        verb = "re-ingested (replaced existing)" if result["replaced"] else "ingested"
        st.success(f"{result['name']} {verb} — {result['chunks']} chunks")
    st.rerun()

# --- Current index ----------------------------------------------------------
st.divider()
st.subheader("Indexed documents")

summary = indexed_summary()
if not summary:
    st.info("No documents indexed yet.")
else:
    total = sum(summary.values())
    st.caption(f"{len(summary)} documents · {total} chunks")
    for name, n in sorted(summary.items()):
        col_name, col_btn = st.columns([4, 1])
        col_name.write(f"📄 {name} — {n} chunks")
        if col_btn.button("Delete", key=f"del::{name}"):
            removed = delete_document(name)
            st.toast(f"Deleted {name} ({removed} chunks)")
            st.rerun()
