
import argparse
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

try:  
    from langchain_classic.chains import RetrievalQA
except ImportError:  
    from langchain.chains import RetrievalQA

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a document Q&A assistant. Answer the question using ONLY the "
        "context below. If the answer is not in the context, reply exactly: "
        "\"I couldn't find that in the provided documents.\" Do not use outside "
        "knowledge.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


def load_documents(docs_dir: str):
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())  
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    if not docs:
        raise SystemExit(f"No PDF/TXT files found in {docs_dir}")
    return docs


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)  # metadata (source, page) is preserved


def get_vectorstore(docs_dir: str, rebuild: bool = False):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if rebuild and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    chunks = chunk_documents(load_documents(docs_dir))
    print(f"Indexing {len(chunks)} chunks...")
    return Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)


def get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="recursive/flint-v1.0:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )


def build_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


def ask(qa, question: str):
    result = qa.invoke({"query": question})
    print(f"\nAnswer:\n{result['result']}\n")

    seen = set()
    print("Sources:")
    for d in result["source_documents"][:1]:
        name = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page")  # PyPDFLoader pages are 0-indexed
        label = f"{name}, page {page + 1}" if page is not None else name
        if label not in seen:
            seen.add(label)
            print(f"  - {label}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./docs", help="folder with PDFs/TXTs")
    parser.add_argument("--ask", help="single question (omit for interactive)")
    parser.add_argument("--rebuild", action="store_true", help="re-index documents")
    args = parser.parse_args()

    qa = build_qa_chain(get_vectorstore(args.docs, args.rebuild))

    if args.ask:
        ask(qa, args.ask)
    else:
        print("Ask a question (type 'exit' to quit).")
        while (q := input("> ").strip()).lower() not in {"exit", "quit"}:
            if q:
                ask(qa, q)


if __name__ == "__main__":
    main()
