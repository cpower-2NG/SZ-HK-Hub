from __future__ import annotations

import pytest

from rag_store import chunk_text, _extract_keywords


# ── 段落分块测试（新版） ──────────────────────────────────


def test_chunk_text_paragraph_boundary() -> None:
    """段落按空行边界拼接，不会在段落中间切断。"""
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = list(chunk_text(text, chunk_size=200, overlap=20))
    # 三个短段落应合并为一个 chunk
    assert len(chunks) == 1
    assert "第一段" in chunks[0]
    assert "第三段" in chunks[0]


def test_chunk_text_splits_large_paragraphs() -> None:
    """超长段落按句子切分。"""
    text = "A" * 600
    chunks = list(chunk_text(text, chunk_size=500))
    assert len(chunks) >= 2
    # 有 overlap，总长度应大于原文
    total = sum(len(c) for c in chunks)
    assert total >= 600


def test_chunk_text_long_paragraph_with_sentences() -> None:
    """超长段落含中文句号时按句子边界切。"""
    text = ("第一句内容。" * 80)  # 约 480 字符
    chunks = list(chunk_text(text, chunk_size=100))
    for c in chunks:
        # 每个 chunk 应以句号结尾（除了最后一个）
        assert c.endswith("。") or len(c) < 100


def test_chunk_text_mixed_paragraphs() -> None:
    """混合短段落和长段落的文本。"""
    text = "短段落。\n\n" + ("长段落内容。" * 100)
    chunks = list(chunk_text(text, chunk_size=200))
    assert len(chunks) >= 2
    # 第一段是短段落
    assert "短段落" in chunks[0]


def test_chunk_text_empty_input() -> None:
    assert list(chunk_text("", chunk_size=8, overlap=2)) == []


def test_chunk_text_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        list(chunk_text("hello", chunk_size=0, overlap=0))


def test_chunk_text_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        list(chunk_text("hello", chunk_size=5, overlap=5))


# ── 关键词提取测试（新版） ────────────────────────────────


def test_extract_keywords_chinese_segments() -> None:
    """中英文混合查询提取正确关键词。"""
    kw = _extract_keywords("ZA Bank开户材料")
    assert "ZA" in kw
    assert "Bank" in kw
    assert "开户" in kw
    assert "开户材料" in kw
    assert "材料" in kw


def test_extract_keywords_english_only() -> None:
    kw = _extract_keywords("MTR fare HongKong")
    assert "MTR" in kw
    assert "fare" in kw
    assert "HongKong" in kw


def test_extract_keywords_chinese_only() -> None:
    kw = _extract_keywords("香港开户需要什么材料")
    assert "香港" in kw
    assert "开户" in kw
    assert "什么" in kw
    assert "材料" in kw


def test_extract_keywords_deduplicate() -> None:
    """重复关键词去重。"""
    kw = _extract_keywords("开户开户开户")
    assert kw.count("开户") == 1


def test_extract_keywords_empty_query() -> None:
    assert _extract_keywords("") == []
    assert _extract_keywords("   ") == []

