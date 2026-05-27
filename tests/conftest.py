from __future__ import annotations

import pytest

from config import AppConfig


@pytest.fixture
def make_config():
    def _make(**overrides):
        base = dict(
            openai_api_key=None,
            openai_base_url="https://api.openai.com/v1",
            openai_model="gpt-4o-mini",
            anthropic_api_key=None,
            anthropic_base_url="https://api.anthropic.com",
            anthropic_model="claude-3-5-sonnet-20241022",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5:1.5b",
            vision_provider="openai",
            ocr_space_api_url="https://api.ocr.space/parse/image",
            ocr_space_api_key=None,
            mcp_base_url=None,
            mcp_api_key=None,
            exchange_rate_api_url="https://open.er-api.com/v6/latest/HKD",
            port_wait_time_api_url="https://www.immd.gov.hk/opendata/control-points/estimated-waiting-time-zh.json",
            mtr_schedule_api_url="https://rt.data.gov.hk/v1/transport/mtr/getSchedule.php",
            mtr_default_line="TML",
            mtr_default_station="AUS",
            mcp_port_tool="port_traffic",
            mcp_mtr_tool="mtr_schedule",
            mcp_exchange_tool="exchange_rate",
            rag_corpus_path="./rag_corpus",
            rag_db_path="./rag_db",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            request_timeout=15,
        )
        base.update(overrides)
        return AppConfig(**base)

    return _make
