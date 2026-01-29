from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".doc", ".txt", ".md")

LOADER_MAP = {
    ".pdf": lambda p: PyMuPDFLoader(str(p)).load(),
    ".docx": lambda p: UnstructuredWordDocumentLoader(str(p)).load(),
    ".doc": lambda p: UnstructuredWordDocumentLoader(str(p)).load(),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8").load(),
    ".md": lambda p: TextLoader(str(p), encoding="utf-8").load(),
}


def load_one_file(fp: Path) -> list:
    """
    根据文件后缀选择合适的 loader 并返回该文件解析后的文档列表（每页或每个文档为一个 Document）
    - PyMuPDFLoader 返回每页为一个 Document（metadata 里通常含 'page'）
    - UnstructuredWordDocumentLoader 返回 Word 文档切分结果
    - TextLoader 用于 .txt/.md 文本文件
    """
    suffix = fp.suffix.lower()
    if suffix not in LOADER_MAP:
        raise ValueError(f"不支持的文件类型: {fp}")
    return LOADER_MAP[suffix](fp)


def load_corpus(root_dir: str) -> list:
    """
    遍历给定目录，递归查找支持的文件类型，统一加载并为每个 Document 设置 metadata['source']（文件路径）
    返回所有文档的列表（每个文档可能是原始页或文本段落，取决于 loader）
    """
    root = Path(root_dir)
    all_docs = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_SUFFIXES:
            docs = load_one_file(fp)
            for d in docs:
                d.metadata["source"] = str(fp)
            all_docs.extend(docs)
    return all_docs


if __name__ == "__main__":
    docs = load_corpus("../data")
    print(docs)
