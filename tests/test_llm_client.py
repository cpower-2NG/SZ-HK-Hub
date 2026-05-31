from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm_client as llm_module
from errors import ConfigError
from llm_client import LLMClient


def _resp(status_code: int, payload: dict, text: str = ""):
    return SimpleNamespace(status_code=status_code, text=text, json=lambda: payload)


def test_provider_priority_openai_over_others(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: True)
    client = LLMClient(
        make_config(openai_api_key="sk-openai", anthropic_api_key="sk-anthropic")
    )
    assert client.provider == "openai"


def test_provider_falls_back_to_ollama_when_no_keys(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: True)
    client = LLMClient(make_config(openai_api_key=None, anthropic_api_key=None))
    assert client.provider == "ollama"


def test_provider_none_when_no_keys_and_ollama_unreachable(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: False)
    client = LLMClient(make_config(openai_api_key=None, anthropic_api_key=None))
    assert client.provider is None


def test_chat_raises_when_not_configured(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: False)
    client = LLMClient(make_config(openai_api_key=None, anthropic_api_key=None))
    with pytest.raises(ConfigError, match="未配置任何 LLM"):
        client.chat("sys", "user")


def test_chat_json_parses_wrapped_json_content(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: False)
    client = LLMClient(make_config(openai_api_key="sk-openai"))

    def fake_post(url, headers, json, timeout):
        return _resp(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": '说明：```json\\n{"tasks":["a","b","c"]}\\n```'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    result = client.chat_json("你是规划器", "给我计划")
    assert result["tasks"] == ["a", "b", "c"]


def test_chat_ollama_json_mode_injects_instruction(make_config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured = {}

    def fake_post(url, json, timeout):
        captured["system"] = json["messages"][0]["content"]
        return _resp(200, {"choices": [{"message": {"content": "{\"ok\":true}"}}]})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module.LLMClient, "_ollama_reachable", lambda self: True)
    client = LLMClient(make_config(openai_api_key=None, anthropic_api_key=None))
    _ = client.chat("普通系统提示", "用户问题", json_mode=True)
    assert "JSON" in captured["system"] or "json" in captured["system"]
