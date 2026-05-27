from __future__ import annotations

from types import SimpleNamespace

import pytest

import mcp_client as mcp_module
from errors import ConfigError, ServiceError
from mcp_client import MCPClient


def _resp(status_code: int, payload: dict, text: str = ""):
    return SimpleNamespace(status_code=status_code, text=text, json=lambda: payload)


def test_call_tool_requires_mcp_base_url(make_config) -> None:
    client = MCPClient(make_config(mcp_base_url=None))
    with pytest.raises(ConfigError, match="MCP_BASE_URL"):
        client.call_tool("port_traffic", {"port": "深圳湾"})


def test_get_exchange_rate_success(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(exchange_rate_api_url="https://example.com"))

    def fake_get(url, timeout):
        assert "example.com" in url
        return _resp(200, {"rates": {"CNY": 0.93}})

    monkeypatch.setattr(mcp_module.requests, "get", fake_get)
    result = client.get_exchange_rate()
    assert result.base == "HKD"
    assert result.target == "CNY"
    assert result.rate == pytest.approx(0.93)


def test_get_exchange_rate_missing_currency(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(exchange_rate_api_url="https://example.com"))

    monkeypatch.setattr(
        mcp_module.requests,
        "get",
        lambda url, timeout: _resp(200, {"rates": {"USD": 0.12}}),
    )

    with pytest.raises(ServiceError, match="目标币种"):
        client.get_exchange_rate(target="CNY")


def test_get_port_traffic_parses_multiple_keys(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(200, {"result": {"wait_minutes": 18}})

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.get_port_traffic("深圳湾")
    assert result.queue_minutes == 18


def test_get_port_traffic_falls_back_to_api(make_config, monkeypatch) -> None:
    client = MCPClient(
        make_config(mcp_base_url=None, port_wait_time_api_url="https://example.com")
    )

    def fake_get(url, timeout):
        assert "example.com" in url
        return _resp(
            200,
            [
                {
                    "controlPoint": "深圳灣口岸",
                    "direction": "Departure",
                    "waitingTime": "Less than 30 minutes",
                }
            ],
        )

    monkeypatch.setattr(mcp_module.requests, "get", fake_get)
    result = client.get_port_traffic("深圳湾")
    assert result.queue_minutes == 30


def test_get_mtr_schedule_falls_back_to_api(make_config, monkeypatch) -> None:
    client = MCPClient(
        make_config(mcp_base_url=None, mtr_schedule_api_url="https://example.com")
    )

    def fake_get(url, params, timeout):
        assert "example.com" in url
        assert params["line"] == "tml"
        assert params["sta"] == "AUS"
        return _resp(
            200,
            {
                "status": 1,
                "data": {"TML": {"AUS": {"UP": [{"time": "1010"}, {"time": "1016"}]}}},
            },
        )

    monkeypatch.setattr(mcp_module.requests, "get", fake_get)
    result = client.get_mtr_schedule("西九龙")
    assert result.interval_minutes == 6


def test_get_mtr_schedule_http_error(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(500, {}, text="server error")

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    with pytest.raises(ServiceError, match="MCP 工具调用失败"):
        client.get_mtr_schedule("西九龙")
