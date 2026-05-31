"""MCP 服务端集成测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mcp_server import app, USER_DATA_DIR

client = TestClient(app)


# ── 健康检查 ──────────────────────────────────────────────


def test_health_check() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "port_traffic" in data["tools"]
    assert "route_planner" in data["tools"]
    assert "file_ops" in data["tools"]


# ── 口岸排队 ──────────────────────────────────────────────


def test_port_traffic_default() -> None:
    resp = client.post("/tools/port_traffic", json={})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["port"] == "深圳湾"
    assert isinstance(result["queue_minutes"], int)


def test_port_traffic_specific() -> None:
    resp = client.post("/tools/port_traffic", json={"port": "福田"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["port"] == "福田"


def test_port_traffic_unknown_port_returns_default() -> None:
    resp = client.post("/tools/port_traffic", json={"port": "未知口岸"})
    assert resp.status_code == 200
    assert isinstance(resp.json()["result"]["queue_minutes"], int)


# ── 港铁班次 ──────────────────────────────────────────────


def test_mtr_schedule_default() -> None:
    resp = client.post("/tools/mtr_schedule", json={})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["station"] == "西九龙"
    assert isinstance(result["interval_minutes"], int)


def test_mtr_schedule_specific() -> None:
    resp = client.post("/tools/mtr_schedule", json={"station": "金钟"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["station"] == "金钟"


# ── 路线规划 ──────────────────────────────────────────────


def test_route_planner_preset_hit() -> None:
    resp = client.post("/tools/route_planner", json={"origin": "福田", "destination": "西九龙"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["source"] == "preset"
    assert len(result["routes"]) >= 1
    assert result["routes"][0]["mode"]


def test_route_planner_unknown_returns_default() -> None:
    resp = client.post("/tools/route_planner", json={"origin": "火星", "destination": "金星"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["source"] == "mock"
    assert len(result["routes"]) >= 1


# ── 文件操作 ──────────────────────────────────────────────


def test_file_ops_save_and_load() -> None:
    data = {"key": "value", "list": [1, 2, 3]}
    # save
    resp = client.post("/tools/file_ops", json={
        "action": "save", "filename": "test_ops.json", "data": data,
    })
    assert resp.status_code == 200
    assert resp.json()["result"]["status"] == "ok"

    # load
    resp = client.post("/tools/file_ops", json={
        "action": "load", "filename": "test_ops.json",
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["status"] == "ok"
    assert result["data"] == data


def test_file_ops_load_not_found() -> None:
    resp = client.post("/tools/file_ops", json={
        "action": "load", "filename": "nonexistent.json",
    })
    assert resp.status_code == 200
    assert resp.json()["result"]["status"] == "not_found"


def test_file_ops_list() -> None:
    # 先保存一个文件确保有内容
    client.post("/tools/file_ops", json={
        "action": "save", "filename": "list_test.json", "data": {"x": 1},
    })
    resp = client.post("/tools/file_ops", json={"action": "list"})
    assert resp.status_code == 200
    files = resp.json()["result"]["files"]
    assert any(f["name"] == "list_test.json" for f in files)


def test_file_ops_invalid_action() -> None:
    resp = client.post("/tools/file_ops", json={"action": "delete"})
    assert resp.status_code == 400


def test_file_ops_save_missing_data() -> None:
    resp = client.post("/tools/file_ops", json={
        "action": "save", "filename": "test.json",
    })
    assert resp.status_code == 400


def test_file_ops_path_traversal_blocked() -> None:
    """路径穿越攻击被阻止，文件名被清理。"""
    resp = client.post("/tools/file_ops", json={
        "action": "save",
        "filename": "../../etc/passwd",
        "data": {"malicious": True},
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["filename"] == "passwd"  # 只取文件名部分
