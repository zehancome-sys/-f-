from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from RAG.document_loader import load_corpus
from RAG.document_splitter import split_documents


def embedding_documents(vector_store, chunks: list) -> None:
    """把所有 chunk 加入向量库（计算向量并插入）。"""
    vector_store.add_documents(chunks)


if __name__ == "__main__":
    ROOT_DIR = "../data"
    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    vector_store = Chroma(
        collection_name="research_corpus",
        embedding_function=embeddings,
        persist_directory="../chroma_db",
    )
    docs = load_corpus(ROOT_DIR)
    chunks = split_documents(docs)
    embedding_documents(vector_store, chunks)
