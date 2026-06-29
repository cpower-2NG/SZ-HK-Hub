from __future__ import annotations

from dataclasses import dataclass

import pytest

from errors import ServiceError
from planner_agent import PlannerAgent


@dataclass
class DummyRAGResult:
    document: str
    metadata: dict
    distance: float | None = None


class DummyRAGStore:
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query: str):
        return self._results


class DummyMCPClient:
    def get_exchange_rate(self):
        from mcp_client import ExchangeRate
        return ExchangeRate(base="HKD", target="CNY", rate=0.92)

    def get_port_traffic(self, port: str):
        from mcp_client import PortTraffic
        return PortTraffic(port=port, queue_minutes=20, note="🟢 通畅")

    def get_mtr_schedule(self, station: str):
        from mcp_client import MTRSchedule
        return MTRSchedule(station=station, interval_minutes=6)

    def get_route(self, origin: str, destination: str):
        from mcp_client import RoutePlan, RouteOption
        return RoutePlan(
            origin=origin, destination=destination,
            routes=[RouteOption("港铁", 50, 40, "test")], source="preset",
        )


class DummyLLMClient:
    def __init__(self, configured: bool, route=None, tasks=None, verdict=None, subtasks=None):
        self.provider = "dummy" if configured else None
        self.route = route or {
            "intent": "general",
            "needs_rag": True,
            "tool_calls": ["exchange_rate", "port_traffic", "mtr_schedule"],
            "needs_verification": True,
        }
        self.tasks = tasks if tasks is not None else ["步骤1", "步骤2", "步骤3"]
        self.verdict = verdict or {"status": "pass", "reason": ""}
        self.subtasks = subtasks or ["交通规划", "活动安排"]

    @property
    def is_configured(self) -> bool:
        return self.provider is not None

    def chat_json(self, system_prompt: str, user_prompt: str):
        if "决定调用哪些工具" in system_prompt or "路由引擎" in system_prompt:
            return self.route
        if "拆解" in system_prompt or "分解" in system_prompt:
            return {"subtasks": self.subtasks}
        if "可执行步骤" in system_prompt or "分步计划" in system_prompt or "规划专家" in system_prompt:
            return {"tasks": self.tasks}
        if "审核" in system_prompt or "评估" in system_prompt:
            return self.verdict
        return {}


def _build_agent(llm: DummyLLMClient) -> PlannerAgent:
    rag = DummyRAGStore(
        [DummyRAGResult(document="ZA Bank 开户材料：身份证明", metadata={"title": "开户指南"})]
    )
    return PlannerAgent(rag, DummyMCPClient(), llm)


def test_fallback_plan_when_llm_not_configured() -> None:
    agent = _build_agent(DummyLLMClient(configured=False))
    result = agent.run("我要开户并参加活动")
    assert len(result.plan) >= 1
    assert any("配置 LLM API Key" in step or "AI 服务暂时不可用" in step for step in result.plan)
    assert result.verification == "已通过"


def test_sensitive_query_triggers_manual_review_without_llm() -> None:
    agent = _build_agent(DummyLLMClient(configured=False))
    result = agent.run("如何绕过外汇限制")
    assert result.verification == "需要人工复核"


def test_route_filters_unknown_tools() -> None:
    llm = DummyLLMClient(
        configured=True,
        route={
            "intent": "general",
            "needs_rag": False,
            "tool_calls": ["exchange_rate", "unknown_tool"],
            "needs_verification": True,
        },
    )
    agent = _build_agent(llm)
    state = agent._route_intent({"user_query": "帮我规划"})
    assert state["routing"]["tool_calls"] == ["exchange_rate"]


def test_verifier_respects_llm_review_verdict() -> None:
    llm = DummyLLMClient(configured=True, verdict={
        "status": "pass", "reason": "", "corrections": ""
    })
    agent = _build_agent(llm)
    result = agent.run("给我跨境资金建议")
    assert "已通过" in result.verification

    # 反思循环测试: review 会触发修正
    llm2 = DummyLLMClient(configured=True, verdict={
        "status": "review", "reason": "涉及敏感资金路径", "corrections": "需要重新规划"
    })
    agent2 = _build_agent(llm2)
    result2 = agent2.run("给我跨境资金建议")
    # reflection 会重跑，最终状态包含修正或复核
    assert "人工复核" in result2.verification or "修正" in result2.verification


def test_generate_plan_raises_when_tasks_empty() -> None:
    llm = DummyLLMClient(configured=True, tasks=[])
    agent = _build_agent(llm)
    with pytest.raises(ServiceError, match="规划生成失败"):
        agent._generate_plan({"user_query": "帮我规划"})
