"""RAG 混合搜索单元测试。"""

from __future__ import annotations

from dataclasses import dataclass

from rag_store import _extract_keywords, RAGStore, RAGResult


# ── 关键词提取 ────────────────────────────────────────────


def test_extract_keywords_chinese_english_mixed() -> None:
    kw = _extract_keywords("ZA Bank开户需要什么材料")
    assert "ZA" in kw
    assert "Bank" in kw
    assert "开户" in kw
    assert "材料" in kw
    assert "什么" in kw


def test_extract_keywords_dedup() -> None:
    kw = _extract_keywords("香港香港开户开户")
    assert kw.count("香港") == 1
    assert kw.count("开户") == 1


def test_extract_keywords_empty() -> None:
    assert _extract_keywords("") == []


# ── 混合搜索结果排序 ──────────────────────────────────────

# 使用 mock 模拟 ChromaDB 返回结果


@dataclass
class MockQueryResult:
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]


def test_hybrid_search_keyword_match_ranks_first(make_config, monkeypatch) -> None:
    """含关键词的文档应排在最前面。"""
    store = RAGStore(make_config())
    store._collection = _FakeCollection(
        documents=[
            "交通指南：从福田到西九龙坐高铁14分钟",  # 不含 ZA/Bank/开户
            "ZA Bank是虚拟银行，可以在线开户",       # 含 ZA + Bank + 开户
            "香港消费指南：汇率和支付方式",           # 不含
        ],
        distances=[0.5, 0.6, 0.4],  # 第三个语义最近但无关
        count=3,
    )

    results = store.search("ZA Bank开户")
    assert len(results) > 0
    assert "ZA Bank" in results[0].document  # 关键词命中的排第一


def test_hybrid_search_no_keyword_match_uses_semantic(make_config, monkeypatch) -> None:
    """无关键词命中时回退纯语义搜索。"""
    store = RAGStore(make_config())
    store._collection = _FakeCollection(
        documents=["内容A", "内容B", "内容C"],
        distances=[0.1, 0.5, 0.9],
        count=3,
    )

    results = store.search("完全无关的查询词")
    # 应返回语义最近的结果
    assert len(results) > 0
    assert results[0].document == "内容A"


def test_hybrid_search_empty_query(make_config) -> None:
    store = RAGStore(make_config())
    assert store.search("") == []


def test_hybrid_search_empty_collection(make_config) -> None:
    store = RAGStore(make_config())
    store._collection = _FakeCollection(documents=[], distances=[], count=0)
    assert store.search("开户") == []


# ── Fake ChromaDB Collection ──────────────────────────────


class _FakeCollection:
    def __init__(self, documents: list[str], distances: list[float], count: int):
        self._documents = documents
        self._distances = distances
        self._count = count

    def count(self) -> int:
        return self._count

    def query(self, query_texts, n_results):
        n = min(n_results, len(self._documents))
        return {
            "documents": [self._documents[:n]],
            "metadatas": [[{"source": f"doc_{i}.md", "title": f"doc_{i}"} for i in range(n)]],
            "distances": [self._distances[:n]],
        }
