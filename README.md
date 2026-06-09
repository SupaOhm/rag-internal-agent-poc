# RAG Internal Agent POC

A minimal Retrieval-Augmented Generation (RAG) chatbot that watches a local `/docs`
folder, auto-ingests PDFs and TXT files into ChromaDB, and answers questions via a
Streamlit chat UI — all free, no credit card required.

## Tech stack

| Component | Package |
|-----------|---------|
| LLM | Gemini 2.5 Flash (with 2.5 Flash-Lite fallback) via `langchain-google-genai` |
| Embeddings | `models/gemini-embedding-001` (Google AI Studio, free) |
| Vector store | ChromaDB (local, persistent) |
| UI | Streamlit |
| File watching | Watchdog |

The chat model and its fallback are configurable via the `GEMINI_CHAT_MODEL` and
`GEMINI_FALLBACK_MODEL` env vars. When the primary model hits its free-tier
quota, the agent automatically retries on the fallback model.

> **Experimental:** a parallel answering backend built on the [Google Agent
> Development Kit](https://google.github.io/adk-docs/) (`google-adk`) lives
> alongside the LangChain one. Same docs, same usage dashboard — different agent.
> See [Experimental: Google ADK backend](#experimental-google-adk-backend).

---

## Quick start

### 1. Get a free Google AI Studio API key

Go to <https://aistudio.google.com> → **Get API key** (no credit card needed).

> Free-tier limits are small and per-model (e.g. `gemini-2.5-flash` ≈ 20 requests/day,
> 5 requests/min). The Admin UI shows live usage against these caps — see below.

### 2. Configure your API key

```bash
cp .env.example .env
# Open .env and replace "your-key-here" with your real key
```

Optional env vars: `GEMINI_CHAT_MODEL` (default `gemini-2.5-flash`) and
`GEMINI_FALLBACK_MODEL` (default `gemini-2.5-flash-lite`) override the models used.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended. Use a virtual environment:
> `python -m venv .venv && source .venv/bin/activate`

### 4. Run the app

```bash
python run.py
```

Starts both interfaces at once:

| Interface | Default URL |
|-----------|-------------|
| User chat | `http://localhost:8501` |
| Admin (document upload) | `http://localhost:8502` |

If a port is already in use, the launcher picks the next free port and prints the actual URLs. Press **Ctrl+C** to stop both.

<details>
<summary>Run individually</summary>

```bash
streamlit run src/app.py               # user chat  (port 8501)
streamlit run src/admin.py             # admin UI   (port 8502)
streamlit run src/adk_app.py           # experimental ADK chat (port 8501)
```

Or launch the experimental ADK chat + admin together: `python run.py --adk`
(see [Experimental: Google ADK backend](#experimental-google-adk-backend)).
</details>

### 5. Add documents

Drop any **PDF** or **TXT** file into the `/docs` folder, or use the **Admin** UI to upload directly from the browser.
The watcher picks it up automatically and ingests it into ChromaDB — no restart needed.

---

## Project structure

```
rag-internal-agent-poc/
├── docs/              ← drop files here to auto-ingest
├── chroma_db/         ← vector store (auto-created, git-ignored)
├── usage_events.jsonl ← self-tracked API usage log (auto-created, git-ignored)
├── src/
│   ├── ingest.py      ← file watcher + ChromaDB ingestion (shared by both backends)
│   ├── agent.py       ← RAG chain (LangChain + Gemini) + session memory
│   ├── usage.py       ← self-tracked API usage + rate-limit counters (shared)
│   ├── app.py         ← Streamlit chat UI (user, LangChain backend)
│   ├── admin.py       ← Streamlit admin UI (upload + usage dashboard)
│   ├── adk_agent.py   ← experimental Google ADK answering layer (parallel to agent.py)
│   └── adk_app.py     ← experimental Streamlit chat UI (ADK backend)
├── run.py             ← one-command launcher (add --adk for the ADK chat)
├── .env.example       ← copy to .env and add your API key
├── .gitignore
├── requirements.txt
├── requirements-adk.txt ← extra deps for the ADK track (includes base reqs)
└── README.md
```

## How it works

1. **Watchdog** monitors `/docs` for new `.pdf` / `.txt` files.
2. New files are split into overlapping chunks and embedded via Google's
   `embedding-001` model.
3. Chunks are stored in a local **ChromaDB** collection that persists between
   sessions.
4. When you ask a question, the top-4 most relevant chunks are retrieved and
   passed to **Gemini 2.5 Flash** to generate an answer.
5. The source filename is shown below every answer.

### Conversation memory

The chat keeps a short, **session-only** history (nothing is persisted to disk).
Follow-up questions with pronouns or references ("what about *its* price?") are
resolved against recent turns into a standalone search query — but only when the
question actually looks like a follow-up, to avoid wasting an LLM call.

### Usage & rate-limit dashboard

Google exposes no API for current free-tier consumption, so the app counts its
own calls and appends them to `usage_events.jsonl` (shared across the chat and
admin processes). The **Admin** UI shows live requests/day, requests/min and
tokens/min against the known free-tier caps per model.

## Experimental: Google ADK backend

A second answering backend built on the **Google Agent Development Kit**
(`google-adk`) runs in parallel to the default LangChain one. It is *additive* —
`app.py` / `agent.py` are untouched and remain the default. The stack may shift
to ADK later, so this track is being matured before any switch.

What it shares vs. what differs:

- **Shared:** retrieval (the same ChromaDB vector store) and the usage log. A doc
  ingested by the watcher is queryable from both backends, and ADK chat calls show
  up in the same Admin dashboard.
- **Different:** the answering layer. Retrieval is exposed to the model as an ADK
  **tool** — the LLM forms the query and decides when to call it (this also
  resolves follow-up references, replacing the LangChain rewrite step).

```bash
pip install -r requirements-adk.txt    # google-adk + base reqs
python run.py --adk                     # ADK chat (8501) + admin (8502)
# or:  streamlit run src/adk_app.py     # ADK chat only
```

> On a quota 429, ADK prints a verbose internal traceback to the console *before*
> our handler catches it and retries on the fallback model. That console noise is
> expected — the chat UI still shows a clean answer.

## Running on WSL (Windows Subsystem for Linux)

Fully supported on **WSL2**. WSL1 is not supported (broken `inotify` means the file watcher won't work).

**Extra steps vs. macOS/Linux:**

1. Keep the project on the WSL filesystem (e.g. `~/rag-internal-agent-poc`), not on a
   Windows-mounted path like `/mnt/c/...` — `inotify` doesn't fire on Windows mounts,
   so the watcher would miss new files.

2. Streamlit won't auto-open a browser. After running `streamlit run src/app.py`,
   open **http://localhost:8501** manually in your Windows browser.

3. To drop files into `/docs` from Windows Explorer, navigate to:
   ```
   \\wsl$\Ubuntu\home\<your-username>\rag-internal-agent-poc\docs
   ```
   Copy files there and the watcher will pick them up automatically.

Everything else (install, API key setup, run command) is identical to macOS.

## Notes

- `chroma_db/`, `usage_events.jsonl` and `.env` are git-ignored — safe to commit the rest.
- Re-starting the app does **not** re-ingest already-indexed files.
- Tested on macOS (Apple Silicon & Intel), Linux, and WSL2.
