"""RAG 存储层 —— 提供向量检索与文档摄入能力。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from config import AppConfig


# Short timeout for model download to avoid hanging in demo environments
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")


@dataclass(frozen=True)
class RAGResult:
    document: str
    metadata: dict
    distance: float | None


class RAGStore:
    def __init__(self, config: AppConfig, collection_name: str = "sz_hk_docs") -> None:
        self.config = config
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=config.rag_db_path)
        self._collection = None
        self._init_error: str | None = None

    def _ensure_collection(self) -> None:
        if self._collection is not None:
            return
        if self._init_error:
            return

        result: list = [None]
        exception: list[Exception | None] = [None]

        def _load() -> None:
            try:
                embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )
                result[0] = self.client.get_or_create_collection(
                    name=self.collection_name, embedding_function=embedding_fn
                )
            except Exception as exc:
                exception[0] = exc

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        thread.join(timeout=10)

        if exception[0] is not None:
            self._init_error = f"知识库模型加载失败：{exception[0]}"
        elif result[0] is not None:
            self._collection = result[0]
        else:
            self._init_error = "知识库模型加载超时（网络不可达），请在有网络的环境下运行 rag_ingest.py"

    def search(self, query: str, *, top_k: int = 4) -> list[RAGResult]:
        if not query.strip():
            return []
        self._ensure_collection()
        if self._collection is None or self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        output: list[RAGResult] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            output.append(RAGResult(document=document, metadata=metadata, distance=distance))
        return output

    def has_documents(self) -> bool:
        self._ensure_collection()
        if self._collection is None:
            return False
        return self._collection.count() > 0


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        yield text[start:end]
        if end == length:
            break
        start = end - overlap


def ingest_corpus(config: AppConfig, *, reset: bool = True) -> int:
    corpus_dir = Path(config.rag_corpus_path)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=config.rag_db_path)
    if reset:
        try:
            client.delete_collection("sz_hk_docs")
        except Exception:
            pass
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model
    )
    collection = client.get_or_create_collection(
        name="sz_hk_docs", embedding_function=embedding_fn
    )
    files = [
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    ]
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for file_path in files:
        relative_path = file_path.relative_to(corpus_dir).as_posix()
        safe_prefix = relative_path.replace("/", "_")
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for idx, chunk in enumerate(chunk_text(content)):
            ids.append(f"{safe_prefix}-{idx}")
            documents.append(chunk)
            metadatas.append({"source": str(file_path), "chunk": idx, "title": file_path.stem})
    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)
