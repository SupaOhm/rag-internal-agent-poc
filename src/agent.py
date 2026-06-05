import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
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
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

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

    def __init__(self, llm, retrieval_chain):
        self.llm = llm
        self.retrieval_chain = retrieval_chain

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
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    return RagChain(llm, retrieval_chain)


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


def ask(chain: "RagChain", question: str) -> dict:
    route = chain.classify(question)
    try:
        if route == "aggregate":
            return _aggregate_answer(chain, question)

        result = chain.retrieval_chain.invoke({"input": question})
        sources = sorted(
            {doc.metadata.get("source", "unknown") for doc in result["context"]}
        )
        return {"answer": result["answer"], "sources": sources, "route": "semantic"}
    except Exception as exc:
        if _is_quota_error(exc):
            return {"answer": _quota_message(exc), "sources": [], "route": route, "error": True}
        raise
