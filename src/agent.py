from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

from ingest import get_vectorstore

load_dotenv()

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


def build_chain():
    # Shares the single per-process vector store, so documents ingested by the
    # background watcher are immediately retrievable.
    vectorstore = get_vectorstore()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    doc_chain = create_stuff_documents_chain(llm, _PROMPT)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(retriever, doc_chain)


def ask(chain, question: str) -> dict:
    result = chain.invoke({"input": question})
    sources = sorted(
        {doc.metadata.get("source", "unknown") for doc in result["context"]}
    )
    return {"answer": result["answer"], "sources": sources}
