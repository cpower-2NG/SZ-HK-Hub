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

    with pytest.raises(ServiceError, match="汇率"):
        client.get_exchange_rate(target="CNY")


def test_get_port_traffic_parses_multiple_keys(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(200, {"result": {"wait_minutes": 18}})

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.get_port_traffic("深圳湾")
    assert result.queue_minutes == 18


def test_get_mtr_schedule_http_error(make_config, monkeypatch) -> None:
    client = MCPClient(make_config())

    def fake_get(url, timeout):
        return _resp(500, {}, text="server error")

    monkeypatch.setattr(mcp_module.requests, "get", fake_get)
    # MTR 实时 API 失败时返回默认间隔 8 分钟（不抛异常）
    result = client.get_mtr_schedule("西九龙")
    assert result.interval_minutes == 8


# ── 新增：路线规划测试 ────────────────────────────────────


def test_get_route_returns_route_plan(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(200, {
            "result": {
                "origin": "福田",
                "destination": "西九龙",
                "routes": [
                    {"mode": "高铁", "duration_min": 14, "cost_hkd": 80, "note": "最快"},
                    {"mode": "港铁", "duration_min": 50, "cost_hkd": 40, "note": "经济"},
                ],
                "source": "preset",
            }
        })

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.get_route("福田", "西九龙")
    assert result.origin == "福田"
    assert result.destination == "西九龙"
    assert len(result.routes) == 2
    assert result.routes[0].mode == "高铁"
    assert result.routes[0].duration_min == 14
    assert result.routes[0].cost_hkd == 80.0


# ── 新增：文件操作测试 ────────────────────────────────────


def test_file_save_success(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        assert json["action"] == "save"
        return _resp(200, {"result": {"action": "save", "filename": "plan.json", "status": "ok"}})

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.file_save("plan.json", {"trip": "深圳湾→西九龙"})
    assert result.action == "save"
    assert result.status == "ok"


def test_file_load_not_found(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(200, {"result": {"action": "load", "filename": "x.json", "status": "not_found", "data": None}})

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.file_load("x.json")
    assert result.status == "not_found"
    assert result.data is None


def test_file_list_returns_files(make_config, monkeypatch) -> None:
    client = MCPClient(make_config(mcp_base_url="https://mcp.example"))

    def fake_post(url, headers, json, timeout):
        return _resp(200, {"result": {"action": "list", "files": [{"name": "a.json", "size": 100}]}})

    monkeypatch.setattr(mcp_module.requests, "post", fake_post)
    result = client.file_list()
    assert result.action == "list"
    assert len(result.files) == 1
    assert result.files[0]["name"] == "a.json"
