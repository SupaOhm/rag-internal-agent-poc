from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "rag_docs"

_PROMPT = PromptTemplate(
    template=(
        "Use the context below to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": _PROMPT},
    )
    return chain


def ask(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    sources = sorted(
        {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
    )
    return {"answer": result["result"], "sources": sources}
