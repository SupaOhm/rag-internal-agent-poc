"""Google ADK implementation of the RAG agent — a PARALLEL track to agent.py.

This is an experimental alternate architecture. The LangChain agent (agent.py)
remains the default; this module re-answers the same questions using the Google
Agent Development Kit (`google-adk`) instead of LangChain chains.

What is shared vs. new (see CLAUDE.md decisions):
  • SHARED: the vector store / retrieval (ingest.get_vectorstore) and the usage
    log (usage.record) — so docs ingested by the watcher are immediately
    queryable here too, and ADK chat calls show up in the same admin dashboard.
  • NEW: the answering layer. Retrieval is exposed to the model as an ADK
    *tool* (`retrieve_docs`); the LLM decides the query and calls it. The
    aggregate/exhaustive path stays a manual map-reduce (a single tool call
    can't scan a corpus larger than the context window).

`build_app()` + `ask(app, question, history)` mirror agent.build_chain()/ask()
so adk_app.py and app.py stay near-identical and easy to diff.

Requires `pip install -r requirements-adk.txt` and a GOOGLE_API_KEY in .env.
"""
import asyncio
import os
import re
import uuid

from dotenv import load_dotenv

from ingest import get_vectorstore
import usage

load_dotenv()

# Same env-configurable models + quota strategy as the LangChain agent: a
# primary model with a fallback that has its OWN daily quota bucket. ADK has no
# `.with_fallbacks`, so we retry on a separate fallback agent ourselves (below).
_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

_TOP_K = 4
_MAX_HISTORY_MESSAGES = 8

APP_NAME = "rag_adk"
USER_ID = "local"

_SYSTEM_INSTRUCTION = (
    "You answer questions about the user's internal documents.\n"
    "ALWAYS call the `retrieve_docs` tool first to fetch relevant context, then "
    "answer ONLY from what it returns. If the answer is not in the retrieved "
    "context, say you don't know.\n"
    "When the question refers to an earlier turn (pronouns like 'it', 'they', "
    "'the same'), use the conversation history to phrase a clear, standalone "
    "query for `retrieve_docs`. The history is only for resolving references — "
    "do NOT invent facts from it; answers must come from retrieved context."
)

# ---------------------------------------------------------------------------
# Aggregate routing — identical heuristic to agent.py. "how many / list all"
# questions map-reduce over EVERY chunk instead of top-k similarity (which
# silently misses chunks). Regex, not an LLM call, to save a request.
# ---------------------------------------------------------------------------
_AGGREGATE_RE = re.compile(
    r"\b(how many|how much|number of|count of|total number|list (all|every)|"
    r"name all|name every|all (the |of )|every |each (of |document)|"
    r"what are all|give me all|enumerate|altogether)\b",
    re.IGNORECASE,
)

_MAP_BATCH_CHARS = 24000


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _quota_message(exc: Exception) -> str:
    retry = re.search(r"retry in ([0-9.]+)s", str(exc))
    when = f" Try again in ~{int(float(retry.group(1)))}s" if retry else ""
    return (
        "⚠️ Gemini free-tier quota reached for both the primary and fallback "
        f"models.{when}, or set GEMINI_CHAT_MODEL to another model / enable "
        "billing. (No answer was generated.)"
    )


# ---------------------------------------------------------------------------
# The retrieval tool. ADK inspects the signature + docstring to build the tool
# schema the model sees, so both matter. Returns plain dicts/lists (ADK
# serialises the result back to the model as the tool response).
# ---------------------------------------------------------------------------
def retrieve_docs(query: str) -> dict:
    """Retrieve the most relevant document chunks for a query.

    Args:
        query: A standalone search query describing what to look up.

    Returns:
        A dict with `chunks` (the retrieved text) and `sources` (the filenames
        the chunks came from).
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=_TOP_K)
    sources = sorted({(d.metadata or {}).get("source", "unknown") for d in docs})
    return {
        "chunks": [d.page_content for d in docs],
        "sources": sources,
    }


def _build_agent(model: str):
    """Construct an ADK agent bound to a specific model."""
    from google.adk.agents import Agent

    return Agent(
        name="rag_agent",
        model=model,
        description="Answers questions from the user's internal documents.",
        instruction=_SYSTEM_INSTRUCTION,
        tools=[retrieve_docs],
    )


class AdkRagApp:
    """Bundles the primary + fallback ADK agents and a shared session service so
    `ask()` can dispatch and, on a quota 429, retry on the fallback model."""

    def __init__(self, primary, fallback, session_service):
        self.primary = primary
        self.fallback = fallback
        self.session_service = session_service

    def classify(self, question: str) -> str:
        return "aggregate" if _AGGREGATE_RE.search(question) else "semantic"


def build_app() -> "AdkRagApp":
    from google.adk.sessions import InMemorySessionService

    primary = _build_agent(_CHAT_MODEL)
    fallback = (
        _build_agent(_FALLBACK_MODEL)
        if _FALLBACK_MODEL and _FALLBACK_MODEL != _CHAT_MODEL
        else None
    )
    return AdkRagApp(primary, fallback, InMemorySessionService())


# ---------------------------------------------------------------------------
# Semantic path — run the ADK agent for one turn. History is folded into the
# message text (rather than a persisted ADK session) so ask() stays stateless
# and matches agent.ask(): the caller owns the history list.
# ---------------------------------------------------------------------------
def _format_history(history) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-_MAX_HISTORY_MESSAGES:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _record_event_usage(event) -> None:
    """Log any model token usage on an ADK event to the shared usage log so ADK
    chat calls appear in the admin dashboard alongside LangChain ones."""
    meta = getattr(event, "usage_metadata", None)
    if not meta:
        return
    total = getattr(meta, "total_token_count", 0) or 0
    # ADK doesn't surface the resolved model on the event; attribute to the
    # configured chat model. Good enough for free-tier RPD/TPM tracking.
    usage.record(_CHAT_MODEL, "chat", requests=1, tokens=total)


async def _run_agent(app: "AdkRagApp", agent, message: str):
    """Drive one ADK agent turn, collecting the final text + any tool sources."""
    from google.adk.runners import Runner
    from google.genai import types

    session_id = uuid.uuid4().hex
    await app.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        agent=agent, app_name=APP_NAME, session_service=app.session_service
    )
    content = types.Content(role="user", parts=[types.Part(text=message)])

    final_text = ""
    sources: set[str] = set()
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        _record_event_usage(event)
        # Harvest sources from the retrieve_docs tool response(s).
        if event.content and event.content.parts:
            for part in event.content.parts:
                fr = getattr(part, "function_response", None)
                if fr and fr.name == "retrieve_docs":
                    resp = fr.response or {}
                    for s in resp.get("sources", []):
                        sources.add(s)
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                final_text = text
    return final_text, sorted(sources)


def _semantic_answer(app: "AdkRagApp", question: str, history) -> dict:
    message = _format_history(history) + f"User question: {question}"
    try:
        answer, sources = asyncio.run(_run_agent(app, app.primary, message))
        return {"answer": answer, "sources": sources, "route": "semantic"}
    except Exception as exc:
        if _is_quota_error(exc) and app.fallback is not None:
            try:
                answer, sources = asyncio.run(
                    _run_agent(app, app.fallback, message)
                )
                return {"answer": answer, "sources": sources, "route": "semantic"}
            except Exception as exc2:
                if _is_quota_error(exc2):
                    return {"answer": _quota_message(exc2), "sources": [],
                            "route": "semantic", "error": True}
                raise
        if _is_quota_error(exc):
            return {"answer": _quota_message(exc), "sources": [],
                    "route": "semantic", "error": True}
        raise


# ---------------------------------------------------------------------------
# Aggregate path — map-reduce over the whole corpus. Uses the google.genai
# client directly (the same SDK ADK runs on): a single agent/tool call can't
# scan a corpus larger than the context window, so we batch + reduce ourselves.
# ---------------------------------------------------------------------------
_MAP_PROMPT = (
    "From the document excerpts below, extract EVERY item relevant to the "
    "question. Be exhaustive and terse — one item per line. If nothing is "
    "relevant, reply exactly NONE.\n\nQuestion: {q}\n\nExcerpts:\n{ctx}"
)
_REDUCE_PROMPT = (
    "Merge the partial extractions into one final answer. Deduplicate items. "
    "If the question asks a count, state the number THEN list the items.\n\n"
    "Question: {q}\n\nPartial extractions:\n{ctx}"
)


def _genai_client():
    from google import genai

    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def _generate(client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    meta = getattr(resp, "usage_metadata", None)
    tokens = getattr(meta, "total_token_count", 0) or 0 if meta else 0
    usage.record(model, "chat", requests=1, tokens=tokens)
    return (resp.text or "").strip()


def _batches(texts, budget: int):
    cur, size = [], 0
    for t in texts:
        if cur and size + len(t) > budget:
            yield cur
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        yield cur


def _aggregate_answer(question: str) -> dict:
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    sources = sorted({(m or {}).get("source", "unknown") for m in metas})

    if not docs:
        return {"answer": "No documents are indexed yet.", "sources": [],
                "route": "aggregate"}

    client = _genai_client()
    try:
        partials: list[str] = []
        for group in _batches(docs, _MAP_BATCH_CHARS):
            ctx = "\n\n---\n\n".join(group)
            out = _generate(client, _CHAT_MODEL,
                            _MAP_PROMPT.format(q=question, ctx=ctx))
            if out and out.upper() != "NONE":
                partials.append(out)

        if not partials:
            return {"answer": "I don't know.", "sources": sources,
                    "route": "aggregate"}

        final = _generate(client, _CHAT_MODEL,
                          _REDUCE_PROMPT.format(q=question, ctx="\n\n".join(partials)))
        return {"answer": final, "sources": sources, "route": "aggregate"}
    except Exception as exc:
        if _is_quota_error(exc):
            return {"answer": _quota_message(exc), "sources": sources,
                    "route": "aggregate", "error": True}
        raise


def ask(app: "AdkRagApp", question: str, history=None) -> dict:
    """Answer `question`. Mirrors agent.ask() — returns
    {answer, sources, route[, error]}."""
    route = app.classify(question)
    if route == "aggregate":
        return _aggregate_answer(question)
    return _semantic_answer(app, question, history)
