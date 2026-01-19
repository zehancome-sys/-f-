from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from RAG.document_loader import load_corpus
import json

# 切分设置：chunk_size 控制每个文档片段最大字符数，chunk_overlap 保证上下文连续性
def split_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    docs=load_corpus("../data")

    # 把文档拆分成适合做 embedding 的小段（chunks）
    chunks = splitter.split_documents(docs)

    # 为每个 chunk 添加唯一 ID、索引、以及对 PDF 页码做友好显示（page_display）
    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = str(uuid4())   # 唯一 id，便于溯源与引用
        d.metadata["chunk_index"] = i           # 在整个语料中的顺序索引
        # 如果 loader 提供了 'page' 字段（如 PyMuPDFLoader），将其转为以 1 开始显示
        if "page" in d.metadata:
            # 注意：不同 loader 的 page 起始可能为 0 或 1，+1 是为了向终端用户显示更直观的页码
            d.metadata["page_display"] = int(d.metadata["page"]) + 1

    return chunks

if __name__ == "__main__":
    chunks = split_documents()
    
    # 将chunks转换为JSON格式
    chunks_json = []
    for chunk in chunks:
        chunk_dict = {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        chunks_json.append(chunk_dict)
    
    # 按JSON格式打印，使用indent参数美化输出
    print(json.dumps(chunks_json, ensure_ascii=False, indent=2))
