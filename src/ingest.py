import re
import threading
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import usage


class TrackedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """GoogleGenerativeAIEmbeddings that logs each embed request for the admin
    usage dashboard. Free tier counts one request per embedded text, so a batch
    of N documents = N requests."""

    def embed_documents(self, texts, *args, **kwargs):
        out = super().embed_documents(texts, *args, **kwargs)
        usage.record(self.model, "embed", requests=len(texts))
        return out

    def embed_query(self, text, *args, **kwargs):
        out = super().embed_query(text, *args, **kwargs)
        usage.record(self.model, "embed", requests=1)
        return out


class IngestQuotaError(Exception):
    """Free-tier embedding quota exhausted after retries."""


class IngestError(Exception):
    """Unrecoverable ingestion error (not quota-related)."""


# Free tier: 100 embed requests/min. Keep batches under that with headroom.
_EMBED_BATCH = 80
_EMBED_BATCH_PAUSE = 5.0   # seconds between batches
_EMBED_MAX_RETRIES = 3


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _retry_delay(exc: Exception) -> float:
    """Parse the suggested retryDelay from the error, default 65s."""
    m = re.search(r"retryDelay.*?(\d+)s", str(exc))
    return float(m.group(1)) + 5 if m else 65.0


def _add_documents_with_retry(vs, chunks: list) -> None:
    """Add chunks in batches, retrying each batch on quota errors."""
    batches = [chunks[i:i + _EMBED_BATCH] for i in range(0, len(chunks), _EMBED_BATCH)]
    for idx, batch in enumerate(batches):
        if idx > 0:
            time.sleep(_EMBED_BATCH_PAUSE)
        for attempt in range(_EMBED_MAX_RETRIES + 1):
            try:
                vs.add_documents(batch)
                break
            except Exception as exc:
                if _is_quota_error(exc):
                    if attempt >= _EMBED_MAX_RETRIES:
                        raise IngestQuotaError(
                            "Google embedding quota exceeded (free tier: 100 requests/min). "
                            "The document is too large to ingest in one go on the free plan.\n"
                            "Options:\n"
                            "  • Wait ~1 min and retry with a smaller file\n"
                            "  • Upgrade your plan: https://ai.dev/rate-limit"
                        ) from exc
                    delay = _retry_delay(exc)
                    print(f"[ingest] quota hit — waiting {delay:.0f}s (retry {attempt + 1}/{_EMBED_MAX_RETRIES})…")
                    time.sleep(delay)
                else:
                    raise IngestError(f"Embedding failed: {exc}") from exc

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "rag_docs"

# Single shared vector store per process. Both the background watcher thread and
# the query/retriever path use this same instance so that documents ingested at
# runtime are immediately visible to the retriever without restarting the app.
_vectorstore = None
_vs_lock = threading.Lock()
# Serialise ingestion so concurrent file events don't double-write or race on
# the dedupe check.
_ingest_lock = threading.Lock()


def get_vectorstore():
    global _vectorstore
    with _vs_lock:
        if _vectorstore is None:
            embeddings = TrackedGoogleEmbeddings(model="models/gemini-embedding-001")
            _vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
                collection_name=COLLECTION,
            )
        return _vectorstore


def _indexed_sources() -> set[str]:
    """Filenames already present in the vector store."""
    vs = get_vectorstore()
    sources: set[str] = set()
    try:
        data = vs.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            if meta and "source" in meta:
                sources.add(meta["source"])
    except Exception as exc:
        # Surface the failure: an empty set here silently causes full
        # re-ingestion (e.g. if chroma_db/ is corrupted).
        print(f"[ingest] warning: could not read existing sources: {exc}")
    return sources


def _wait_until_stable(filepath: str, timeout: float = 30.0) -> None:
    """Wait until a file's size stops changing, so we don't ingest a partial
    write (e.g. a large PDF still being copied into /docs)."""
    last_size = -1
    waited = 0.0
    while waited < timeout:
        try:
            size = Path(filepath).stat().st_size
        except OSError:
            size = -1
        if size > 0 and size == last_size:
            return
        last_size = size
        time.sleep(0.5)
        waited += 0.5


def ingest_file(filepath: str, skip_if_indexed: bool = True) -> int:
    """Load, split, and add a file to the vector store. Returns the number of
    chunks added (0 if skipped or empty)."""
    path = Path(filepath)
    if path.suffix.lower() not in (".pdf", ".txt"):
        return 0

    with _ingest_lock:
        if skip_if_indexed and path.name in _indexed_sources():
            print(f"[ingest] {path.name} already indexed, skipping")
            return 0

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            print(f"[ingest] {path.name} produced no text, skipping")
            return 0

        for chunk in chunks:
            chunk.metadata["source"] = path.name

        _add_documents_with_retry(get_vectorstore(), chunks)
        print(f"[ingest] {path.name} → {len(chunks)} chunks added")
        return len(chunks)


class DocHandler(FileSystemEventHandler):
    def _handle(self, src_path: str):
        if not src_path.lower().endswith((".pdf", ".txt")):
            return
        _wait_until_stable(src_path)
        try:
            ingest_file(src_path)
        except Exception as exc:
            print(f"[ingest] error processing {src_path}: {exc}")

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        # Files that arrive via atomic rename/move (some downloads, editors)
        # fire on_moved, not on_created.
        if not event.is_directory:
            self._handle(event.dest_path)


def start_watcher():
    DOCS_DIR.mkdir(exist_ok=True)
    observer = Observer()
    observer.schedule(DocHandler(), str(DOCS_DIR), recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[ingest] watching {DOCS_DIR}")
    return observer


def ingest_existing():
    """Ingest files already in /docs that are not yet in the vector store."""
    DOCS_DIR.mkdir(exist_ok=True)
    already_indexed = _indexed_sources()
    for f in DOCS_DIR.iterdir():
        if f.suffix.lower() in (".pdf", ".txt") and f.name not in already_indexed:
            try:
                ingest_file(str(f))
            except Exception as exc:
                print(f"[ingest] error processing {f.name}: {exc}")


# ---------------------------------------------------------------------------
# Admin API — upload, list, and delete documents. Kept here (not in the chat
# app) so the admin surface shares the exact same ingestion path as /docs:
# files always land in DOCS_DIR and go through ingest_file().
# ---------------------------------------------------------------------------
SUPPORTED_SUFFIXES = (".pdf", ".txt")


def remove_source(name: str) -> int:
    """Delete every chunk belonging to `name` from the vector store. Returns the
    number of chunks removed."""
    vs = get_vectorstore()
    with _ingest_lock:
        data = vs.get(where={"source": name})
        ids = data.get("ids") or []
        if ids:
            vs.delete(ids=ids)
        return len(ids)


def save_upload(filename: str, data: bytes) -> Path:
    """Persist uploaded bytes into /docs, sanitising the name to block path
    traversal (only the final path component is kept)."""
    DOCS_DIR.mkdir(exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name or Path(safe_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported or invalid filename: {filename!r}")
    dest = DOCS_DIR / safe_name
    dest.write_bytes(data)
    return dest


def ingest_upload(filename: str, data: bytes) -> dict:
    """Admin entrypoint: save an uploaded file to /docs and (re)ingest it.

    Re-uploading the same name replaces its existing chunks, so editing and
    re-uploading a document updates the index instead of duplicating it.
    Returns {"name", "chunks", "replaced"}.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"{filename}: unsupported type (only .pdf/.txt)")

    dest = save_upload(filename, data)
    replaced = remove_source(dest.name)
    chunks = ingest_file(str(dest), skip_if_indexed=False)
    return {"name": dest.name, "chunks": chunks, "replaced": replaced}


def delete_document(name: str) -> int:
    """Remove a document from both the index and /docs so it is not re-ingested
    on the next startup. Returns chunks removed."""
    removed = remove_source(name)
    safe = DOCS_DIR / Path(name).name
    if safe.exists():
        safe.unlink()
    return removed


def indexed_summary() -> dict[str, int]:
    """Map of indexed source filename → chunk count."""
    vs = get_vectorstore()
    data = vs.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data.get("metadatas") or []:
        src = (meta or {}).get("source")
        if src:
            counts[src] = counts.get(src, 0) + 1
    return counts
