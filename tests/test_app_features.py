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
    assert app_module.update_safety("如何绕过外汇限制") == "检测到敏感问题，请人工复核并遵守合规要求。"


def test_update_safety_reports_idle_hint_on_empty_input(app_module) -> None:
    assert app_module.update_safety("") == "系统将自动标记敏感金融/法律问题。"


def test_handle_events_prefers_text_when_no_image(app_module, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "vision_client", type("V", (), {"parse_events": lambda self, data: []})())
    output, conflict = app_module.handle_events("5月18日 10:00 西九龙篮球赛", None)
    assert "西九龙篮球赛" in output
    assert conflict == "未检测到明显冲突。"


def test_handle_events_uses_vision_result_when_image_present(app_module, monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "vision_client",
        type("V", (), {"parse_events": lambda self, data: [{"date": "2026-05-18", "time": "10:00", "title": "开户"}]})(),
    )
    output, conflict = app_module.handle_events("文本不会被使用", b"fake-image")
    assert "2026-05-18 10:00 · 开户" in output
    assert conflict == "未检测到明显冲突。"


def test_handle_events_reports_vision_error_and_falls_back_to_text(app_module, monkeypatch) -> None:
    class BrokenVision:
        def parse_events(self, data):
            raise app_module.ServiceError("视觉不可用")

    monkeypatch.setattr(app_module, "vision_client", BrokenVision())
    output, conflict = app_module.handle_events("5月18日 10:00 赛事", b"fake-image")
    assert "5月18日 10:00 · 赛事" in output
    assert "视觉解析失败：视觉不可用" in conflict


def test_search_rag_handles_empty_and_error_states(app_module, monkeypatch) -> None:
    assert app_module.search_rag("") == "请输入关键词以检索政策或开户信息。"

    app_module.rag_store._init_error = "模型加载失败"
    assert app_module.search_rag("ZA Bank") == "知识库暂不可用：模型加载失败"
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
    assert "开户指南" in output
    assert "ZA Bank 开户材料" in output


def test_search_rag_handles_no_documents(app_module, monkeypatch) -> None:
    monkeypatch.setattr(app_module.rag_store, "search", lambda query: [])
    monkeypatch.setattr(app_module.rag_store, "has_documents", lambda: False)
    assert app_module.search_rag("ZA Bank") == "知识库为空，请先在 rag_corpus/ 补充资料并运行 rag_ingest.py。"


def test_refresh_decision_switches_route_by_queue(app_module, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_fetch_metrics", lambda: ("1 HKD = 0.92 CNY", "深圳湾 20 分钟", "西九龙 6 分钟一班", 20))
    assert app_module.refresh_decision()[3] == "推荐路线：深圳湾口岸 → 西九龙高铁站"

    monkeypatch.setattr(app_module, "_fetch_metrics", lambda: ("1 HKD = 0.92 CNY", "深圳湾 30 分钟", "西九龙 6 分钟一班", 30))
    assert app_module.refresh_decision()[3] == "推荐路线：福田口岸 → 港铁东铁线"

    monkeypatch.setattr(app_module, "_fetch_metrics", lambda: ("汇率数据不可用", "深圳湾 数据不可用", "西九龙 数据不可用", None))
    assert app_module.refresh_decision()[3] == "推荐路线：请先配置口岸实时数据"
