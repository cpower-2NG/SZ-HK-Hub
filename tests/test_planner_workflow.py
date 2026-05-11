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
        return {"rate": 0.92}

    def get_port_traffic(self, port: str):
        return {"port": port, "queue_minutes": 20}

    def get_mtr_schedule(self, station: str):
        return {"station": station, "interval_minutes": 6}


class DummyLLMClient:
    def __init__(self, configured: bool, route=None, tasks=None, verdict=None):
        self.provider = "dummy" if configured else None
        self.route = route or {
            "intent": "general",
            "needs_rag": True,
            "tool_calls": ["exchange_rate", "port_traffic", "mtr_schedule"],
            "needs_verification": True,
        }
        self.tasks = tasks if tasks is not None else ["步骤1", "步骤2", "步骤3"]
        self.verdict = verdict or {"status": "pass", "reason": ""}

    @property
    def is_configured(self) -> bool:
        return self.provider is not None

    def chat_json(self, system_prompt: str, user_prompt: str):
        if "决定调用哪些工具" in system_prompt:
            return self.route
        if "输出 3-5 条可执行步骤" in system_prompt:
            return {"tasks": self.tasks}
        if "合规审核助手" in system_prompt:
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
    assert len(result.plan) == 2
    assert "配置 LLM API Key" in result.plan[0]
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
    llm = DummyLLMClient(configured=True, verdict={"status": "review", "reason": "涉及敏感资金路径"})
    agent = _build_agent(llm)
    result = agent.run("给我跨境资金建议")
    assert result.verification.startswith("需要人工复核")


def test_generate_plan_raises_when_tasks_empty() -> None:
    llm = DummyLLMClient(configured=True, tasks=[])
    agent = _build_agent(llm)
    with pytest.raises(ServiceError, match="规划生成失败"):
        agent._generate_plan({"user_query": "帮我规划"})
