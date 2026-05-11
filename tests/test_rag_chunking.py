from __future__ import annotations

import pytest

from rag_store import chunk_text


def test_chunk_text_basic_overlap() -> None:
    text = "abcdefghij"
    chunks = list(chunk_text(text, chunk_size=4, overlap=1))
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_empty_input() -> None:
    assert list(chunk_text("", chunk_size=8, overlap=2)) == []


def test_chunk_text_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        list(chunk_text("hello", chunk_size=0, overlap=0))


def test_chunk_text_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        list(chunk_text("hello", chunk_size=5, overlap=5))
