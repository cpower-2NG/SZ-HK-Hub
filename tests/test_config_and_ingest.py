from __future__ import annotations

from pathlib import Path

import config as config_module
import rag_store as rag_module
from config import AppConfig
from rag_store import ingest_corpus


class FakeCollection:
    def __init__(self):
        self.add_calls = []

    def add(self, ids, documents, metadatas):
        self.add_calls.append((ids, documents, metadatas))


class FakeClient:
    def __init__(self):
        self.deleted = []
        self.collection = FakeCollection()

    def delete_collection(self, name: str):
        self.deleted.append(name)

    def get_or_create_collection(self, name: str, embedding_function=None):
        return self.collection


def test_app_config_from_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MCP_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    cfg = AppConfig.from_env()
    assert cfg.openai_api_key is None
    assert cfg.mcp_base_url is None
    assert cfg.ollama_model == "qwen2.5:1.5b"


def test_app_config_from_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MCP_BASE_URL", "https://mcp.example")
    monkeypatch.setenv("REQUEST_TIMEOUT", "30")
    cfg = AppConfig.from_env()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.mcp_base_url == "https://mcp.example"
    assert cfg.request_timeout == 30


def test_ingest_corpus_reads_md_txt_and_respects_reset(tmp_path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    db_dir = tmp_path / "db"
    corpus_dir.mkdir()
    db_dir.mkdir()

    (corpus_dir / "guide.md").write_text("A" * 620, encoding="utf-8")
    (corpus_dir / "note.txt").write_text("B" * 100, encoding="utf-8")
    (corpus_dir / "ignore.pdf").write_text("pdf", encoding="utf-8")

    fake_client = FakeClient()

    monkeypatch.setattr(rag_module.chromadb, "PersistentClient", lambda path: fake_client)
    monkeypatch.setattr(
        rag_module.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: object(),
    )

    cfg = AppConfig(
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-sonnet-20241022",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:1.5b",
        vision_provider="openai",
        mcp_base_url=None,
        mcp_api_key=None,
        exchange_rate_api_url="https://open.er-api.com/v6/latest/HKD",
        exchange_rate_api_url_backup="https://api.nxvav.cn/api/exchange-rate/",
        mtr_realtime_api_url="https://rt.data.gov.hk/v1/transport/mtr/getSchedule.php",
        immigration_csv_url="https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv",
        mcp_port_tool="port_traffic",
        mcp_mtr_tool="mtr_schedule",
        mcp_exchange_tool="exchange_rate",
        mcp_route_tool="route_planner",
        mcp_file_tool="file_ops",
        google_maps_api_key=None,
        user_data_path="./user_data",
        rag_corpus_path=str(corpus_dir),
        rag_db_path=str(db_dir),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        request_timeout=15,
    )

    count = ingest_corpus(cfg, reset=True)

    assert "sz_hk_docs" in fake_client.deleted
    assert count >= 3
    assert len(fake_client.collection.add_calls) == 1
    ids, documents, metadatas = fake_client.collection.add_calls[0]
    assert len(ids) == len(documents) == len(metadatas) == count
    assert all(Path(m["source"]).suffix in {".md", ".txt"} for m in metadatas)
