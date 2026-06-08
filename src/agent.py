import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

from ingest import get_vectorstore

load_dotenv()

# Models are env-configurable so quota/billing changes don't need a code edit.
# Free-tier daily request caps are small (e.g. gemini-2.5-flash = 20/day), so
# the primary model falls back to a second model that has its OWN daily quota
# bucket when the primary is exhausted.
_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

# ---------------------------------------------------------------------------
# Semantic path — answer a specific question from the few most relevant chunks.
# ---------------------------------------------------------------------------
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the context below to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n"
            "The chat history is only to resolve references (e.g. pronouns like "
            "'it', 'they') — do NOT invent facts from it; answers must come from "
            "the context.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------------------
# Conversation memory (current session only — nothing is persisted). Two needs:
#   1. The answer prompt can SEE prior turns (placeholder above).
#   2. Follow-ups like "who authored it" must be rewritten into a standalone
#      query BEFORE retrieval, or the vector search matches on the pronoun and
#      pulls the wrong chunks. The contextualize prompt below does that rewrite.
# Only the last few turns are kept — long histories burn the tight free-tier
# token budget and rarely help resolve a reference.
# ---------------------------------------------------------------------------
_MAX_HISTORY_MESSAGES = 8

# A query only needs the (extra, quota-costing) rewrite call if it actually
# leans on earlier turns — i.e. it carries a back-reference (a pronoun, "the
# same", "above", a leading "and/what about"). Self-contained questions —
# including short ones like "What is X?" — skip the rewrite and retrieve as-is.
# Heuristic, zero-cost; biased toward skipping (a missed rewrite just retrieves
# the literal query, which is the cheap, safe direction under a tight quota).
_FOLLOWUP_RE = re.compile(
    r"\b(it|its|it's|they|them|their|theirs|this|that|these|those|he|him|his|"
    r"she|her|hers|same|above|previous|former|latter|aforementioned)\b"
    r"|^(and|also|what about|how about|why|then)\b",
    re.IGNORECASE,
)


def _looks_like_followup(question: str) -> bool:
    return bool(_FOLLOWUP_RE.search(question.strip()))


_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question — which may "
            "reference earlier turns — rewrite it as a standalone question that "
            "is understandable without the history. Do NOT answer it. If it is "
            "already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _to_messages(history) -> list:
    """Convert app-level {'role','content'} dicts into LangChain messages,
    keeping only the most recent turns."""
    if not history:
        return []
    msgs = []
    for m in history[-_MAX_HISTORY_MESSAGES:]:
        content = m.get("content", "")
        if m.get("role") == "user":
            msgs.append(HumanMessage(content=content))
        elif m.get("role") == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs

# ---------------------------------------------------------------------------
# Router — classify aggregate/exhaustive questions ("how many", "list all") so
# they map-reduce over the whole corpus instead of top-k similarity (which
# silently misses chunks). This is a regex heuristic, NOT an LLM call: under a
# tight free-tier request budget, spending an API call just to route every
# query is wasteful, so we keep routing free and deterministic.
# ---------------------------------------------------------------------------
_AGGREGATE_RE = re.compile(
    r"\b(how many|how much|number of|count of|total number|list (all|every)|"
    r"name all|name every|all (the |of )|every |each (of |document)|"
    r"what are all|give me all|enumerate|altogether)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Aggregate path — map-reduce over EVERY chunk. Map: pull relevant items from
# each batch. Reduce: merge + dedupe into the final answer. Batching by a char
# budget keeps it working when the corpus grows past a single context window.
# ---------------------------------------------------------------------------
_MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "From the document excerpts below, extract EVERY item relevant to "
            "the question. Be exhaustive and terse — one item per line. "
            "If nothing is relevant, reply exactly NONE.\n\n"
            "Question: {question}\n\nExcerpts:\n{context}",
        ),
    ]
)

_REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Merge the partial extractions into one final answer. Deduplicate "
            "items. If the question asks a count, state the number THEN list "
            "the items.\n\nQuestion: {question}\n\nPartial extractions:\n{context}",
        ),
    ]
)

# Char budget per map batch. ~24k chars ≈ a few thousand tokens — well within
# context, small enough that many batches run if the corpus is large.
_MAP_BATCH_CHARS = 24000


def _build_llm():
    """Primary chat model with an automatic fallback to a second model that has
    a separate daily quota bucket, so a 429 on the primary doesn't dead-end."""
    primary = ChatGoogleGenerativeAI(model=_CHAT_MODEL, temperature=0)
    if _FALLBACK_MODEL and _FALLBACK_MODEL != _CHAT_MODEL:
        fallback = ChatGoogleGenerativeAI(model=_FALLBACK_MODEL, temperature=0)
        return primary.with_fallbacks([fallback])
    return primary


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _quota_message(exc: Exception) -> str:
    """Friendly, non-crashing message for a daily-quota 429."""
    retry = re.search(r"retry in ([0-9.]+)s", str(exc))
    when = f" Try again in ~{int(float(retry.group(1)))}s" if retry else ""
    return (
        "⚠️ Gemini free-tier quota reached for both the primary and fallback "
        f"models.{when}, or set GEMINI_CHAT_MODEL to another model / enable "
        "billing. (No answer was generated.)"
    )


class RagChain:
    """Bundles the semantic retrieval chain and the shared LLM so `ask()` can
    dispatch a query down the right path."""

    def __init__(self, llm, retriever, doc_chain, contextualize_chain):
        self.llm = llm
        self.retriever = retriever
        self.doc_chain = doc_chain
        # Rewrites a follow-up into a standalone question before retrieval.
        self.contextualize_chain = contextualize_chain

    def standalone_question(self, question: str, history: list) -> str:
        """Resolve references against history, but only spend the LLM call when
        there's history AND the question actually reads like a follow-up —
        otherwise retrieve the question as-is and save a request against quota."""
        if not history or not _looks_like_followup(question):
            return question
        return self.contextualize_chain.invoke(
            {"input": question, "chat_history": history}
        ).strip()

    def classify(self, question: str) -> str:
        # Heuristic, zero-cost routing — default to the cheaper semantic path.
        return "aggregate" if _AGGREGATE_RE.search(question) else "semantic"


def build_chain():
    # Shares the single per-process vector store, so documents ingested by the
    # background watcher are immediately retrievable.
    vectorstore = get_vectorstore()
    llm = _build_llm()
    doc_chain = create_stuff_documents_chain(llm, _PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
    return RagChain(llm, retriever, doc_chain, contextualize_chain)


def _batches(docs, budget: int):
    """Group docs into batches whose combined content stays under `budget` chars."""
    cur: list[Document] = []
    size = 0
    for d in docs:
        if cur and size + len(d.page_content) > budget:
            yield cur
            cur, size = [], 0
        cur.append(d)
        size += len(d.page_content)
    if cur:
        yield cur


def _aggregate_answer(chain: "RagChain", question: str) -> dict:
    """Map-reduce over the full corpus for count/list/summarise questions."""
    vs = get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(data["documents"], data["metadatas"])
    ]
    sources = sorted({(d.metadata or {}).get("source", "unknown") for d in docs})

    if not docs:
        return {"answer": "No documents are indexed yet.", "sources": [], "route": "aggregate"}

    # Map: extract relevant items from each batch.
    map_chain = _MAP_PROMPT | chain.llm
    partials: list[str] = []
    for group in _batches(docs, _MAP_BATCH_CHARS):
        context = "\n\n---\n\n".join(d.page_content for d in group)
        out = map_chain.invoke({"question": question, "context": context}).content.strip()
        if out and out.upper() != "NONE":
            partials.append(out)

    if not partials:
        return {"answer": "I don't know.", "sources": sources, "route": "aggregate"}

    # Reduce: merge partials into the final answer (also formats the count/list).
    reduce_chain = _REDUCE_PROMPT | chain.llm
    final = reduce_chain.invoke(
        {"question": question, "context": "\n\n".join(partials)}
    ).content.strip()
    return {"answer": final, "sources": sources, "route": "aggregate"}


def ask(chain: "RagChain", question: str, history=None) -> dict:
    route = chain.classify(question)
    chat_history = _to_messages(history)
    try:
        # Resolve references ("who authored it") into a standalone query for
        # retrieval. Gated: only costs an LLM call on genuine follow-ups.
        search_query = chain.standalone_question(question, chat_history)

        if route == "aggregate":
            return _aggregate_answer(chain, search_query)

        # Retrieve on the resolved query, but answer on the ORIGINAL question so
        # the model still sees the user's exact wording, plus full chat history.
        docs = chain.retriever.invoke(search_query)
        answer = chain.doc_chain.invoke(
            {"input": question, "chat_history": chat_history, "context": docs}
        )
        sources = sorted(
            {doc.metadata.get("source", "unknown") for doc in docs}
        )
        return {"answer": answer, "sources": sources, "route": "semantic"}
    except Exception as exc:
        if _is_quota_error(exc):
            return {"answer": _quota_message(exc), "sources": [], "route": route, "error": True}
        raise
