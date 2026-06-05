from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

from ingest import get_vectorstore

load_dotenv()

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
# Router — classify the query so aggregate/exhaustive questions ("how many",
# "list all") don't go through similarity search, which only ever returns the
# top-k most similar chunks and silently misses the rest of the corpus.
# A cheap model is enough for a one-word label; reserve the strong model for
# the actual answer.
# ---------------------------------------------------------------------------
_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's question into exactly one label:\n"
            "- aggregate: needs the WHOLE corpus — counting, listing all, "
            "enumerating, or summarising across every document "
            "(e.g. 'how many projects', 'list all', 'what are all the ...').\n"
            "- semantic: a specific fact answerable from a few passages "
            "(e.g. 'what is OpsBot', 'who is the mentor').\n"
            "Reply with ONLY the single word: aggregate or semantic.",
        ),
        ("human", "{input}"),
    ]
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


class RagChain:
    """Bundles the router, the semantic retrieval chain, and the shared LLMs so
    `ask()` can dispatch a query down the right path."""

    def __init__(self, llm, router_llm, retrieval_chain):
        self.llm = llm
        self.retrieval_chain = retrieval_chain
        self._router = _ROUTER_PROMPT | router_llm

    def classify(self, question: str) -> str:
        label = self._router.invoke({"input": question}).content.strip().lower()
        # Default to semantic on anything unexpected — it's the cheaper path.
        return "aggregate" if "aggregate" in label else "semantic"


def build_chain():
    # Shares the single per-process vector store, so documents ingested by the
    # background watcher are immediately retrievable.
    vectorstore = get_vectorstore()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # Cheap model for the one-word routing decision.
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    doc_chain = create_stuff_documents_chain(llm, _PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    return RagChain(llm, router_llm, retrieval_chain)


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
    if route == "aggregate":
        return _aggregate_answer(chain, question)

    result = chain.retrieval_chain.invoke({"input": question})
    sources = sorted(
        {doc.metadata.get("source", "unknown") for doc in result["context"]}
    )
    return {"answer": result["answer"], "sources": sources, "route": "semantic"}
