"""Admin console — document ingestion, isolated from the chatbot.

Run separately from app.py (its own process / port):

    streamlit run src/admin.py --server.port 8502

It deliberately imports only the ingestion layer, never `agent`/the chat UI.
Uploaded files are saved to /docs and converted into the same RAG index the
chatbot reads, so admin uploads behave exactly like dropping a file into /docs.
"""
import os
import sys
from pathlib import Path

# Allow imports from src/ when running as `streamlit run src/admin.py`
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from ingest import ingest_upload, indexed_summary, delete_document, SUPPORTED_SUFFIXES, IngestQuotaError, IngestError
import usage

load_dotenv()

st.set_page_config(page_title="Admin · Document Ingest", page_icon="🛠️", layout="centered")

# CHAT_URL is set by run.py so the admin can jump back to the chat UI.
_chat_url = os.getenv("CHAT_URL")
if _chat_url:
    st.sidebar.link_button("💬 Open Chat", _chat_url, use_container_width=True)

st.title("Admin — Document Ingest")
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
        with st.spinner(f"Ingesting {f.name}… (large files may pause to respect API rate limits)"):
            try:
                result = ingest_upload(f.name, f.getvalue())
            except IngestQuotaError:
                st.error(
                    f"**{f.name} — quota exceeded**\n\n"
                    "You've hit the free-tier embedding limit (100 requests/min). "
                    "The file has too many chunks to ingest in one go.\n\n"
                    "**What to do:**\n"
                    "- Wait ~1 minute and try again\n"
                    "- Upgrade your Google AI Studio plan to raise the limit: "
                    "https://ai.dev/rate-limit"
                )
                continue
            except IngestError as exc:
                st.error(
                    f"**{f.name} — ingestion failed**\n\n"
                    f"{exc}\n\n"
                    "Check your `GOOGLE_API_KEY` in `.env` and that the file is a valid PDF/TXT."
                )
                continue
            except Exception as exc:
                st.error(
                    f"**{f.name} — unexpected error**\n\n"
                    f"`{type(exc).__name__}: {exc}`\n\n"
                    "If this keeps happening, check the terminal for the full stack trace."
                )
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


# --- Usage & rate limits ----------------------------------------------------
def _meter(col, label, used, limit):
    col.caption(label)
    if limit:
        col.write(f"**{used}** / {limit}")
        col.progress(min(used / limit, 1.0))
        if used >= limit:
            col.caption("⚠️ at limit")
    else:
        col.write(f"**{used}** / ?")


usage.prune()  # keep the on-disk log bounded

st.divider()
st.subheader("Usage & rate limits")

secs = usage.seconds_to_reset()
hrs, mins = divmod(secs // 60, 60)
top_l, top_r = st.columns([3, 1])
top_l.caption(
    "Self-tracked from this app's own API calls (chat + embeddings). Google "
    "exposes no usage API, so these are our counts, not Google's. Daily "
    f"counts reset in **{hrs}h {mins}m** (midnight Pacific)."
)
if top_r.button("🔄 Refresh"):
    st.rerun()

data = usage.stats()
if not data:
    st.info("No API calls recorded yet. Ask a question or ingest a file, then refresh.")
else:
    for model in sorted(data):
        d = data[model]
        lim = d.get("limit") or {}
        st.markdown(f"**{model}**")
        c1, c2, c3 = st.columns(3)
        _meter(c1, "Requests today (RPD)", d["rpd"], lim.get("rpd"))
        _meter(c2, "Requests / min (RPM)", d["rpm"], lim.get("rpm"))
        _meter(c3, "Tokens / min (TPM)", d["tpm"], lim.get("tpm"))
