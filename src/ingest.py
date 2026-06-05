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
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
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


def ingest_file(filepath: str, skip_if_indexed: bool = True):
    path = Path(filepath)
    if path.suffix.lower() not in (".pdf", ".txt"):
        return

    with _ingest_lock:
        if skip_if_indexed and path.name in _indexed_sources():
            print(f"[ingest] {path.name} already indexed, skipping")
            return

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            print(f"[ingest] {path.name} produced no text, skipping")
            return

        for chunk in chunks:
            chunk.metadata["source"] = path.name

        get_vectorstore().add_documents(chunks)
        print(f"[ingest] {path.name} → {len(chunks)} chunks added")


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
