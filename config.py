from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    anthropic_api_key: str | None
    anthropic_base_url: str
    anthropic_model: str
    ollama_base_url: str
    ollama_model: str
    vision_provider: str
    ocr_space_api_url: str
    ocr_space_api_key: str | None
    mcp_base_url: str | None
    mcp_api_key: str | None
    exchange_rate_api_url: str
    port_wait_time_api_url: str
    mtr_schedule_api_url: str
    mtr_default_line: str
    mtr_default_station: str
    mcp_port_tool: str
    mcp_mtr_tool: str
    mcp_exchange_tool: str
    rag_corpus_path: str
    rag_db_path: str
    embedding_model: str
    request_timeout: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            vision_provider=os.getenv("VISION_PROVIDER", "openai"),
            ocr_space_api_url=os.getenv("OCR_SPACE_API_URL", "https://api.ocr.space/parse/image"),
            ocr_space_api_key=os.getenv("OCR_SPACE_API_KEY"),
            mcp_base_url=os.getenv("MCP_BASE_URL"),
            mcp_api_key=os.getenv("MCP_API_KEY"),
            exchange_rate_api_url=os.getenv(
                "EXCHANGE_RATE_API_URL", "https://open.er-api.com/v6/latest/HKD"
            ),
            port_wait_time_api_url=os.getenv(
                "PORT_WAIT_TIME_API_URL",
                "https://www.immd.gov.hk/opendata/control-points/estimated-waiting-time-zh.json",
            ),
            mtr_schedule_api_url=os.getenv(
                "MTR_SCHEDULE_API_URL",
                "https://rt.data.gov.hk/v1/transport/mtr/getSchedule.php",
            ),
            mtr_default_line=os.getenv("MTR_DEFAULT_LINE", "TML"),
            mtr_default_station=os.getenv("MTR_DEFAULT_STATION", "AUS"),
            mcp_port_tool=os.getenv("MCP_PORT_TOOL", "port_traffic"),
            mcp_mtr_tool=os.getenv("MCP_MTR_TOOL", "mtr_schedule"),
            mcp_exchange_tool=os.getenv("MCP_EXCHANGE_TOOL", "exchange_rate"),
            rag_corpus_path=str(Path(os.getenv("RAG_CORPUS_PATH", "./rag_corpus")).resolve()),
            rag_db_path=str(Path(os.getenv("RAG_DB_PATH", "./rag_db")).resolve()),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "15")),
        )
