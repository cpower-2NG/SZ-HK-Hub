from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from config import AppConfig


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
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn
        )

    def search(self, query: str, *, top_k: int = 4) -> list[RAGResult]:
        if not query.strip():
            return []
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=top_k)
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
        return self.collection.count() > 0


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
        start = max(start + 1, end - overlap)


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
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for idx, chunk in enumerate(chunk_text(content)):
            ids.append(f"{file_path.stem}-{idx}")
            documents.append(chunk)
            metadatas.append({"source": str(file_path), "chunk": idx, "title": file_path.stem})
    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)
