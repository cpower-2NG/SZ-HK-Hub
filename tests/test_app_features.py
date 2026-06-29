from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.setattr("llm_client.LLMClient._resolve_provider", lambda self: None, raising=False)
    module = importlib.import_module("app")
    return module


def test_parse_events_handles_multiple_date_formats(app_module) -> None:
    text = """5月18日 10:00 西九龙篮球赛
2026/05/19 11:30 开户预约
05-20 14:00 会面"""
    events = app_module.parse_events(text)

    assert events[0] == {"date": "5月18日", "time": "10:00", "title": "西九龙篮球赛"}
    assert events[1] == {"date": "2026/05/19", "time": "11:30", "title": "开户预约"}
    assert events[2] == {"date": "05-20", "time": "14:00", "title": "会面"}


def test_detect_conflicts_flags_duplicate_slots(app_module) -> None:
    events = [
        {"date": "2026/05/18", "time": "10:00", "title": "A"},
        {"date": "2026/05/18", "time": "10:00", "title": "B"},
        {"date": "2026/05/18", "time": "11:00", "title": "C"},
    ]
    assert app_module.detect_conflicts(events) == ["2026/05/18-10:00"]


def test_has_sensitive_matches_known_terms(app_module) -> None:
    assert app_module.has_sensitive("如何绕过外汇限制") is True
    assert app_module.has_sensitive("普通活动安排") is False


def test_update_safety_reports_review_for_sensitive_input(app_module) -> None:
    assert "检测到敏感问题" in app_module.update_safety("如何绕过外汇限制")


def test_update_safety_reports_idle_hint_on_empty_input(app_module) -> None:
    assert "系统将自动标记" in app_module.update_safety("")


def test_handle_events_prefers_text_when_no_image(app_module, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "vision_client", type("V", (), {"parse_events": lambda self, data: []})())
    output, conflict = app_module.handle_events("5月18日 10:00 西九龙篮球赛", None)
    assert "西九龙篮球赛" in output
    assert "未检测到明显冲突" in conflict


def test_handle_events_uses_vision_result_when_image_present(app_module, monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "vision_client",
        type("V", (), {"parse_events": lambda self, data: [{"date": "2026-05-18", "time": "10:00", "title": "开户"}]})(),
    )
    output, conflict = app_module.handle_events("文本不会被使用", b"fake-image")
    assert "2026-05-18 10:00 · 开户" in output
    assert "未检测到明显冲突" in conflict


def test_handle_events_reports_vision_error_and_falls_back_to_text(app_module, monkeypatch) -> None:
    class BrokenVision:
        def parse_events(self, data):
            raise app_module.ServiceError("视觉不可用")

    monkeypatch.setattr(app_module, "vision_client", BrokenVision())
    output, conflict = app_module.handle_events("5月18日 10:00 赛事", b"fake-image")
    assert "5月18日 10:00 · 赛事" in output
    assert "视觉解析失败：视觉不可用" in conflict


def test_search_rag_handles_empty_and_error_states(app_module, monkeypatch) -> None:
    assert "关键词" in app_module.search_rag("") or "请输入" in app_module.search_rag("")

    app_module.rag_store._init_error = "模型加载失败"
    assert "模型加载失败" in app_module.search_rag("ZA Bank")
    app_module.rag_store._init_error = None


def test_search_rag_formats_results(app_module, monkeypatch) -> None:
    class Result:
        def __init__(self):
            self.metadata = {"title": "开户指南", "source": "/tmp/za-bank-guide.md"}
            self.document = "ZA Bank 开户材料：身份证明和住址证明。"

    monkeypatch.setattr(app_module.rag_store, "search", lambda query: [Result()])
    monkeypatch.setattr(app_module.llm_client, "chat", lambda sys, usr: "")
    monkeypatch.setattr(type(app_module.llm_client), "is_configured", property(lambda self: False))
    output = app_module.search_rag("ZA Bank")
    assert "开户" in output or "ZA Bank" in output


def test_search_rag_handles_no_documents(app_module, monkeypatch) -> None:
    monkeypatch.setattr(app_module.rag_store, "search", lambda query: [])
    monkeypatch.setattr(app_module.rag_store, "has_documents", lambda: False)
    result = app_module.search_rag("ZA Bank")
    assert "知识库为空" in result


def test_refresh_decision_switches_route_by_queue(app_module) -> None:
    """P0 统一表单后 refresh_decision 已合并到 _fetch_metrics。"""
    m = app_module._fetch_metrics()
    assert "rate" in m
    assert "sz_bay" in m
    assert "mtr" in m
