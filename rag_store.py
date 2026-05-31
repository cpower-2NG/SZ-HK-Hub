"""RAG 存储层 —— 提供向量检索与文档摄入能力。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ── 必须在导入 chromadb / sentence-transformers 之前设置 ──
# 强制使用 hf-mirror 镜像，避免国内直连 huggingface.co 被墙
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

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
        thread.join(timeout=30)  # 模型已缓存，加载应很快

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

        # 提取关键词
        keywords = _extract_keywords(query)
        total_docs = self._collection.count()

        # 语义检索取更多候选
        results = self._collection.query(
            query_texts=[query], n_results=min(max(top_k * 4, 16), total_docs)
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        # 分别收集"命中关键词"和"未命中"的文档
        matched: list[tuple[float, str, dict, float | None]] = []
        unmatched: list[tuple[float, str, dict, float | None]] = []

        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = distances[idx] if idx < len(distances) else None

            keyword_score = 0.0
            if keywords:
                doc_lower = doc.lower()
                for kw in keywords:
                    if kw.lower() in doc_lower:
                        keyword_score += 1.0

            semantic_score = 1.0 / (1.0 + (dist or 0.5))
            combined = keyword_score + semantic_score * 0.3

            if keyword_score > 0:
                matched.append((combined, doc, meta, dist))
            else:
                unmatched.append((combined, doc, meta, dist))

        # 排序
        matched.sort(key=lambda x: x[0], reverse=True)
        unmatched.sort(key=lambda x: x[0], reverse=True)

        # 有关键词命中时只返回命中文档，不凑数
        output: list[RAGResult] = []
        if matched:
            for _, doc, meta, dist in matched[:top_k]:
                output.append(RAGResult(document=doc, metadata=meta, distance=dist))
        else:
            # 完全没有关键词命中时才回退纯语义
            for _, doc, meta, dist in unmatched[:top_k]:
                output.append(RAGResult(document=doc, metadata=meta, distance=dist))

        return output

    def has_documents(self) -> bool:
        self._ensure_collection()
        if self._collection is None:
            return False
        return self._collection.count() > 0


def _extract_keywords(query: str) -> list[str]:
    """从中文查询中提取有意义的片段作为关键词。"""
    import re
    keywords: list[str] = []

    # 提取英文/数字词（如 ZA Bank、B2P）
    eng_words = re.findall(r"[A-Za-z0-9]+", query)
    keywords.extend(w for w in eng_words if len(w) >= 2)

    # 提取连续中文字符（2-6字为有效关键词片段）
    chinese_chars = re.findall(r"[\u4e00-\u9fff]+", query)
    for segment in chinese_chars:
        # 对整个中文段本身
        keywords.append(segment)
        # 对长段也提取 2-4 字滑动窗口
        if len(segment) >= 2:
            for size in [2, 3, 4]:
                for i in range(len(segment) - size + 1):
                    keywords.append(segment[i:i + size])

    # 去重
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)
    return unique


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterable[str]:
    """按段落边界智能分块：优先以空行切分，长段落再按句子切分。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    # 1) 先按空行（段落）切分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            # 当前段落放不下，先输出积累的 chunk
            if current:
                yield current
            # 如果单个段落超长，按句子切分
            if len(para) > chunk_size:
                yield from _chunk_long_paragraph(para, chunk_size, overlap)
                current = ""
            else:
                current = para

    if current:
        yield current


def _chunk_long_paragraph(para: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """对超长段落按句子边界切分。"""
    # 按句号、换行等边界切分
    import re
    sentences = re.split(r"(?<=[。！？\n])\s*", para)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunk = ""
    for sent in sentences:
        if len(chunk) + len(sent) <= chunk_size:
            chunk = chunk + sent if chunk else sent
        else:
            if chunk:
                yield chunk
            # 如果单个句子也超长，硬切
            if len(sent) > chunk_size:
                start = 0
                while start < len(sent):
                    yield sent[start:start + chunk_size]
                    start += chunk_size - overlap
                chunk = ""
            else:
                chunk = sent
    if chunk:
        yield chunk


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
