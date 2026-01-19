from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from pathlib import Path

def load_one_file(fp: Path):
    """
    根据文件后缀选择合适的 loader 并返回该文件解析后的文档列表（每页或每个文档为一个 Document）
    - PyMuPDFLoader 返回每页为一个 Document（metadata 里通常含 'page'）
    - UnstructuredWordDocumentLoader 返回 Word 文档切分结果
    - TextLoader 用于 .txt/.md 文本文件
    """
    suffix = fp.suffix.lower()
    if suffix == ".pdf":
        # PyMuPDFLoader：针对 PDF，每页单独作为一个 Document，metadata 会带 page 等信息
        return PyMuPDFLoader(str(fp)).load()
    if suffix in [".docx", ".doc"]:
        # Word 文档：解析后通常没有页码，需要用 source 或其他字段做定位
        return UnstructuredWordDocumentLoader(str(fp)).load()
    if suffix in [".txt", ".md"]:
        # 读取纯文本，注意指定编码防止乱码
        return TextLoader(str(fp), encoding="utf-8").load()
    # 如果文件类型不支持，则抛出异常，便于在目录遍历时发现遗漏或新格式
    raise ValueError(f"不支持的文件类型: {fp}")


def load_corpus(root_dir: str):
    """
    遍历给定目录，递归查找支持的文件类型，统一加载并为每个 Document 设置 metadata['source']（文件路径）
    返回所有文档的列表（每个文档可能是原始页或文本段落，取决于 loader）
    """
    root = Path(root_dir)
    all_docs = []
    # rglob("*") 会递归列出目录下所有文件
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in [".pdf", ".docx", ".doc", ".txt", ".md"]:
            docs = load_one_file(fp)
            # 给每个文档补上来源路径，方便后续溯源展示
            for d in docs:
                d.metadata["source"] = str(fp)
            all_docs.extend(docs)
    return all_docs


if __name__=="__main__":
    # 加载目录下的所有资料（该路径可按需更改）
    docs = load_corpus("../data")
    print(docs)