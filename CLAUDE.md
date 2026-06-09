# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python run.py                          # interactive menu: pick chat backend (Enter = LangChain); chat + admin always launch together
python run.py --langchain              # user chat + admin, skipping the menu
python run.py --adk                    # admin + experimental Google ADK chat instead of LangChain
python run.py --open                   # also open each app in the browser once it's serving
streamlit run src/app.py               # user chat only  (port 8501)
streamlit run src/adk_app.py           # experimental ADK chat only (port 8501)
streamlit run src/admin.py             # admin only       (port 8502)
pip install -r requirements.txt        # install deps (Python 3.10+, use a venv)
pip install -r requirements-adk.txt    # extra deps for the ADK track (includes the base reqs)
```

There is no test suite, linter, or build step. Verification is manual via the Streamlit UIs.

A `.env` with a real `GOOGLE_API_KEY` (free Google AI Studio key) is required for the agent and embeddings to work — copy `.env.example` and fill it in. Optional overrides: `GEMINI_CHAT_MODEL` (default `gemini-2.5-flash`), `GEMINI_FALLBACK_MODEL` (default `gemini-2.5-flash-lite`).

## Architecture

A local RAG chatbot: a file watcher ingests `/docs` into ChromaDB, and a Streamlit chat answers questions via Gemini. The codebase is small (`src/` = 5 modules) but several non-obvious design decisions tie it together:

**Two processes, one shared state on disk.** `run.py` spawns the user chat (`app.py`) and admin (`admin.py`) as *separate* Streamlit processes — both always launch together. (With no args `run.py` shows an interactive menu to choose the chat backend, LangChain vs ADK; `--langchain`/`--adk` skip it. It runs a preflight `GOOGLE_API_KEY`/ADK-deps check, waits for each app's port to accept connections before printing URLs, and passes each app its sibling's URL via env vars — `ADMIN_URL` to the chat, `CHAT_URL` to the admin — so each page can render a small link to the other.) They cannot share memory, so cross-process state lives on disk:
- ChromaDB at `chroma_db/` (the vector store) — each process opens its own handle to the same persistent collection `rag_docs`.
- `usage_events.jsonl` — an append-only usage log both processes write to and read from.

Within a single process, `ingest.get_vectorstore()` returns one lazily-created `Chroma` instance (guarded by `_vs_lock`). The background watcher thread and the query/retriever path share it, so a file dropped in `/docs` is retrievable immediately without restart.

**Ingestion path is unified.** Both the `/docs` file watcher (`DocHandler` → `ingest_file`) and the admin browser upload (`ingest_upload` → `save_upload` + `ingest_file`) funnel through the same `ingest_file()`. Admin uploads always land in `DOCS_DIR` first, so there is exactly one ingestion code path. Dedupe is by source filename (`metadata["source"]`); re-uploading a name deletes its old chunks (`remove_source`) then re-ingests, so editing a doc updates the index instead of duplicating. `_ingest_lock` serialises concurrent file events.

**The free-tier quota budget drives most of the agent logic** (`src/agent.py`). Gemini's free tier is ~20 requests/day per model, so the code aggressively avoids spending LLM calls:
- **Fallback model:** `_build_llm()` wraps the primary model with `.with_fallbacks([fallback])` — the fallback has its own separate daily quota bucket, so a 429 on the primary doesn't dead-end. Quota errors (`RESOURCE_EXHAUSTED`/`429`) are caught and returned as a friendly message, not raised.
- **Routing is regex, not an LLM call.** `classify()` uses `_AGGREGATE_RE` to detect "how many / list all" questions and route them to the map-reduce `_aggregate_answer` path (which scans *every* chunk, since top-k similarity silently misses chunks for exhaustive questions). Everything else takes the cheap top-4 semantic path.
- **Follow-up rewriting is gated.** `standalone_question()` only spends an LLM call to rewrite a query into a standalone form when there *is* history AND `_looks_like_followup()` (regex on pronouns / "and"/"what about") matches. Self-contained questions retrieve as-is. Note: retrieval runs on the *rewritten* query, but the answer prompt sees the *original* question plus chat history.

**Conversation memory is session-only.** History lives in `st.session_state.messages` (`app.py`) and is passed into `ask()`; nothing is persisted. Only the last `_MAX_HISTORY_MESSAGES` (8) turns are kept to protect the token budget.

**Usage tracking is self-reported** (`src/usage.py`), because Google exposes no API for current consumption. Chat calls are counted via `UsageCallback` (a LangChain `BaseCallbackHandler` attached to every `ChatGoogleGenerativeAI`, reading real model name + token counts off the response). Embeddings are counted by `TrackedGoogleEmbeddings` (subclass overriding `embed_documents`/`embed_query`) in `ingest.py` — note the free tier counts **one request per embedded text**, so a batch of N docs = N requests. `record()` never raises (tracking must not break the request path). `stats()` aggregates rpd/rpm/tpm against the hardcoded `LIMITS` table; daily counts reset at midnight Pacific.

**Parallel ADK track (experimental, not merged into the default path).** `src/adk_agent.py` + `src/adk_app.py` re-implement the answering layer on the Google Agent Development Kit (`google-adk`) instead of LangChain, as an alternate architecture under active development. They are *additive* — `app.py`/`agent.py` remain the default and are untouched. Key design choices, deliberately mirroring the LangChain side so the two diff cleanly:
- **Shared, not duplicated:** retrieval (`ingest.get_vectorstore`) and usage tracking (`usage.record`). A doc ingested by the watcher is queryable from both backends, and ADK chat calls land in the same admin dashboard.
- **Retrieval is an ADK tool.** `retrieve_docs(query)` is exposed to the model; the LLM forms the query and decides when to call it (this also handles follow-up reference resolution, replacing the LangChain `standalone_question` rewrite). The answer prompt instruction forbids answering outside retrieved context.
- **Same regex aggregate router** (`_AGGREGATE_RE`). The aggregate/exhaustive path is a manual map-reduce using the `google.genai` client directly (a single tool call can't scan a corpus larger than the context window).
- **Fallback model** is done by retrying on a second ADK agent (ADK has no `.with_fallbacks`); quota 429s are caught and returned as a friendly message.
- `build_app()`/`ask(app, question, history)` intentionally match `agent.build_chain()`/`ask()`. History is folded into the message text (ADK sessions are created fresh per `ask`) so `ask()` stays stateless and the caller owns history.
- Needs `pip install -r requirements-adk.txt` (verified against `google-adk` 2.2.0) + `GOOGLE_API_KEY`. Note: on a quota 429, ADK logs a verbose internal traceback to the console *before* our handler catches it and retries on the fallback — that noise is expected, not a failure; the UI still shows a clean answer.

## Constraints & gotchas

- **WSL2 only** (not WSL1): the watcher relies on `inotify`. Keep the project on the WSL filesystem, not a `/mnt/c/...` Windows mount — `inotify` doesn't fire there and the watcher silently misses files.
- Embedding ingestion of large files retries on quota (`_add_documents_with_retry`, batch size 80, ~5s pauses) and raises `IngestQuotaError`/`IngestError` with user-facing messages.
- `_wait_until_stable()` waits for a file's size to settle before ingesting, so partially-copied files aren't read.
- The `LIMITS` table in `usage.py` is hand-maintained from the AI Studio dashboard; models not listed are still tracked but show an unknown cap.
- Git-ignored (don't commit): `chroma_db/`, `usage_events.jsonl`, `.env`.
