import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "rag_docs"


def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def ingest_file(filepath: str):
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        return

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = path.name

    vs = get_vectorstore()
    vs.add_documents(chunks)
    print(f"[ingest] {path.name} → {len(chunks)} chunks added")


class DocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith((".pdf", ".txt")):
            time.sleep(1)  # wait for file write to complete
            try:
                ingest_file(event.src_path)
            except Exception as exc:
                print(f"[ingest] error processing {event.src_path}: {exc}")


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
    vs = get_vectorstore()

    already_indexed: set[str] = set()
    try:
        data = vs.get()
        for meta in data.get("metadatas") or []:
            if meta and "source" in meta:
                already_indexed.add(meta["source"])
    except Exception:
        pass

    for f in DOCS_DIR.iterdir():
        if f.suffix.lower() in (".pdf", ".txt") and f.name not in already_indexed:
            try:
                ingest_file(str(f))
            except Exception as exc:
                print(f"[ingest] error processing {f.name}: {exc}")
