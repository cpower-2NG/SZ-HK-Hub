from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, CompiledGraph, StateGraph

from errors import ConfigError, ServiceError
from llm_client import LLMClient
from mcp_client import MCPClient
from rag_store import RAGStore

SENSITIVE_TERMS = ["绕过外汇", "套现", "非法", "洗钱", "违规开户", "避税"]
TOOL_ORDER = ["exchange_rate", "port_traffic", "mtr_schedule"]
TOOL_CHOICES = set(TOOL_ORDER)


class AgentState(TypedDict, total=False):
    user_query: str
    route: dict[str, Any]
    rag_results: list[dict[str, Any]]
    tool_results: dict[str, Any]
    plan: list[str]
    verification: str


@dataclass
class PlannerResult:
    plan: list[str]
    verification: str


class PlannerAgent:
    def __init__(self, rag_store: RAGStore, mcp_client: MCPClient, llm_client: LLMClient) -> None:
        self.rag_store = rag_store
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_intent)
        graph.add_node("execute", self._execute_actions)
        graph.add_node("plan", self._generate_plan)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("route")
        graph.add_edge("route", "execute")
        graph.add_edge("execute", "plan")
        graph.add_edge("plan", "verify")
        graph.add_edge("verify", END)
        return graph.compile()

    def run(self, user_query: str) -> PlannerResult:
        result = self.graph.invoke({"user_query": user_query})
        return PlannerResult(
            plan=result.get("plan", []), verification=result.get("verification", "已通过")
        )

    def _route_intent(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if not self.llm_client.is_configured:
            return {
                "route": {
                    "intent": "general",
                    "needs_rag": True,
                    "tool_calls": TOOL_ORDER,
                    "needs_verification": True,
                }
            }
        system_prompt = "你是跨境任务规划代理，请根据用户需求决定调用哪些工具。输出 JSON。"
        user_prompt = (
            "识别用户意图，决定是否需要检索知识库、是否调用实时工具。"
            "JSON 格式："
            '{"intent":"", "needs_rag":true/false, "tool_calls":["exchange_rate","port_traffic","mtr_schedule"], "needs_verification":true/false}.'
            f"\n用户需求：{query}"
        )
        route = self.llm_client.chat_json(system_prompt, user_prompt)
        route["tool_calls"] = [tool for tool in route.get("tool_calls", []) if tool in TOOL_CHOICES]
        return {"route": route}

    def _execute_actions(self, state: AgentState) -> AgentState:
        route = state.get("route", {})
        query = state.get("user_query", "")
        rag_results = []
        if route.get("needs_rag"):
            rag_results = [
                {
                    "document": result.document,
                    "metadata": result.metadata,
                    "distance": result.distance,
                }
                for result in self.rag_store.search(query)
            ]
        tool_results: dict[str, Any] = {}
        for tool in route.get("tool_calls", []):
            try:
                if tool == "exchange_rate":
                    tool_results["exchange_rate"] = self.mcp_client.get_exchange_rate()
                elif tool == "port_traffic":
                    tool_results["port_traffic"] = self.mcp_client.get_port_traffic("深圳湾")
                elif tool == "mtr_schedule":
                    tool_results["mtr_schedule"] = self.mcp_client.get_mtr_schedule("西九龙")
            except (ConfigError, ServiceError) as exc:
                tool_results[tool] = {"error": str(exc)}
        return {"rag_results": rag_results, "tool_results": tool_results}

    def _generate_plan(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if not self.llm_client.is_configured:
            return {
                "plan": [
                    "请配置 LLM API Key 以启用智能规划。",
                    "若已配置，可重新提交需求生成分步计划。",
                ]
            }
        rag_context = "\n".join(
            f"- {item['metadata'].get('title', '参考资料')}：{item['document']}"
            for item in state.get("rag_results", [])
        )
        tool_context = "\n".join(
            f"- {key}: {value}" for key, value in state.get("tool_results", {}).items()
        )
        system_prompt = "你是跨境任务规划助手，输出 3-5 条可执行步骤，使用中文。"
        user_prompt = (
            f"用户需求：{query}\n\n"
            f"知识库：\n{rag_context or '无'}\n\n"
            f"实时工具：\n{tool_context or '无'}\n\n"
            "请输出 JSON：{'tasks':['步骤1','步骤2','步骤3']}。"
        )
        tasks = self.llm_client.chat_json(system_prompt, user_prompt).get("tasks", [])
        plan = [task for task in tasks if isinstance(task, str)]
        if not plan:
            raise ServiceError("规划生成失败")
        return {"plan": plan}

    def _verify(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if any(term in query for term in SENSITIVE_TERMS):
            return {"verification": "需要人工复核"}
        if self.llm_client.is_configured:
            system_prompt = "你是合规审核助手，只回答 JSON。"
            user_prompt = (
                "判断以下内容是否涉及金融/法律敏感操作，"
                "输出 JSON：{'status':'pass'|'review','reason':''}。\n"
                f"{query}"
            )
            verdict = self.llm_client.chat_json(system_prompt, user_prompt)
            if verdict.get("status") == "review":
                return {"verification": f"需要人工复核：{verdict.get('reason', '').strip()}"}
        return {"verification": "已通过"}
