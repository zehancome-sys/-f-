import json
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.document_loader import load_corpus


def split_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """
    把文档拆分成适合做 embedding 的小段（chunks）。
    chunk_size 控制每个片段最大字符数，chunk_overlap 保证上下文连续性。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = str(uuid4())
        d.metadata["chunk_index"] = i
        if "page" in d.metadata:
            d.metadata["page_display"] = int(d.metadata["page"]) + 1

    return chunks


if __name__ == "__main__":
    docs = load_corpus("../data")
    chunks = split_documents(docs)
    chunks_json = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    print(json.dumps(chunks_json, ensure_ascii=False, indent=2))
