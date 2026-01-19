from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from RAG.document_splitter import split_documents

def embedding_documents(vector_store):
    chunks=split_documents()

    # 把所有 chunk 加入向量库（计算向量并插入）
    vector_store.add_documents(chunks)


if __name__=="__main__":
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4"
    )
    vector_store = Chroma(
        collection_name="research_corpus",  # 集合/命名空间，用于区分不同语料库
        embedding_function=embeddings,  # 上面创建的 embedding 函数（或对象）
        persist_directory="../chroma_db",  # 本地持久化目录（会在该目录下存数据）
    )
    embedding_documents(vector_store)