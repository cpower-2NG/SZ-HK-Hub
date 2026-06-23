from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from errors import ConfigError, ServiceError
from llm_client import LLMClient
from mcp_client import MCPClient
from rag_store import RAGStore

SENSITIVE_TERMS = ["绕过外汇", "套现", "非法", "洗钱", "违规开户", "避税"]
TOOL_ORDER = ["exchange_rate", "port_traffic", "mtr_schedule"]
TOOL_CHOICES = set(TOOL_ORDER)


class AgentState(TypedDict, total=False):
    user_query: str
    routing: dict[str, Any]
    rag_results: list[dict[str, Any]]
    tool_results: dict[str, Any]
    plan_steps: list[str]
    verification_status: str


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
            plan=result.get("plan_steps", []), verification=result.get("verification_status", "已通过")
        )

    def _route_intent(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if not self.llm_client.is_configured:
            return {
                "routing": {
                    "intent": "general",
                    "needs_rag": True,
                    "tool_calls": TOOL_ORDER,
                    "needs_verification": True,
                }
            }
        system_prompt = (
            "你是深港跨境任务路由引擎。分析用户需求，决定需要调用哪些工具和知识库。"
            "输出 JSON。"
        )
        user_prompt = (
            "可用工具：\n"
            "- exchange_rate：实时汇率（HKD↔CNY等）\n"
            "- port_traffic：口岸排队时长与客流数据\n"
            "- mtr_schedule：港铁实时到站信息（含东铁线罗湖/落马洲跨境站）\n\n"
            "知识库包含：通关政策、口岸指南、交通攻略、海关规定、旅游景点、科大黄大仙等地点出行指南。\n\n"
            "判断规则：\n"
            "- 涉及过关/签证/通行证 → needs_rag=true\n"
            "- 涉及交通/路线/口岸选择 → needs_rag=true, tool_calls 含 port_traffic+mtr_schedule\n"
            "- 涉及消费/换汇/购物 → tool_calls 含 exchange_rate\n"
            "- 涉及旅游/景点 → needs_rag=true\n"
            "- 涉及海关/违禁品/免税 → needs_rag=true\n"
            "- 涉及金融开户/敏感操作 → needs_verification=true\n"
            "JSON 格式：{\"intent\":\"简短意图描述\",\"needs_rag\":true/false,\"tool_calls\":[\"exchange_rate\",\"port_traffic\",\"mtr_schedule\"],\"needs_verification\":true/false}\n\n"
            f"用户需求：{query}"
        )
        route = self.llm_client.chat_json(system_prompt, user_prompt)
        route["tool_calls"] = [tool for tool in route.get("tool_calls", []) if tool in TOOL_CHOICES]
        return {"routing": route}

    def _execute_actions(self, state: AgentState) -> AgentState:
        route = state.get("routing", {})
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
                    rate = self.mcp_client.get_exchange_rate()
                    tool_results["exchange_rate"] = {
                        "base": rate.base,
                        "target": rate.target,
                        "rate": rate.rate,
                        "summary": f"1 {rate.base} = {rate.rate:.4f} {rate.target}",
                    }
                elif tool == "port_traffic":
                    port = self.mcp_client.get_port_traffic("深圳湾")
                    tool_results["port_traffic"] = {
                        "port": port.port,
                        "queue_minutes": port.queue_minutes,
                        "today_total": port.today_total,
                        "note": port.note,
                        "summary": f"{port.port}：排队约{port.queue_minutes}分钟，{port.note}",
                    }
                    # 同时查罗湖、落马洲
                    for extra_port in ["罗湖", "落马洲"]:
                        try:
                            ep = self.mcp_client.get_port_traffic(extra_port)
                            tool_results[f"port_{extra_port}"] = {
                                "port": ep.port,
                                "queue_minutes": ep.queue_minutes,
                                "summary": f"{ep.port}：排队约{ep.queue_minutes}分钟，{ep.note}",
                            }
                        except Exception:
                            pass
                elif tool == "mtr_schedule":
                    mtr = self.mcp_client.get_mtr_schedule("罗湖")
                    trains_text = ""
                    if mtr.next_trains:
                        trains_text = " | ".join(
                            f"[{t['direction']}] {t['dest']}方向 {t['platform']}号月台 {t['arrive_in']}"
                            for t in mtr.next_trains
                        )
                    tool_results["mtr_schedule"] = {
                        "station": mtr.station,
                        "interval_minutes": mtr.interval_minutes,
                        "next_trains": mtr.next_trains,
                        "summary": f"{mtr.station}：约{mtr.interval_minutes}分钟一班。{trains_text}",
                    }
            except (ConfigError, ServiceError) as exc:
                tool_results[tool] = {"error": str(exc), "summary": f"获取失败：{exc}"}
        return {"rag_results": rag_results, "tool_results": tool_results}

    def _generate_plan(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if not self.llm_client.is_configured:
            return {
                "plan_steps": [
                    "请配置 LLM API Key 以启用智能规划。",
                    "若已配置，可重新提交需求生成分步计划。",
                ]
            }

        # 组织知识库内容（压缩每篇上限，避免 token 爆掉）
        rag_context = "\n".join(
            f"📚 {item['metadata'].get('title', '参考资料')}：{item['document'][:400]}"
            for item in state.get("rag_results", [])[:3]
        )

        # 组织实时工具结果
        tool_parts = []
        for key, value in state.get("tool_results", {}).items():
            if isinstance(value, dict):
                tool_parts.append(f"- {value.get('summary', str(value))}")
            else:
                tool_parts.append(f"- {key}: {value}")
        tool_context = "\n".join(tool_parts) if tool_parts else "暂无实时数据"

        system_prompt = (
            "你是深港跨境生活规划专家。你拥有丰富的深港双城出行经验，熟悉各口岸特点、"
            "交通路线、景点分布、海关规定。\n\n"
            "你的任务是：根据用户需求 + 知识库参考 + 实时工具数据，生成一份**具体、可执行、有细节**的分步计划。\n\n"
            "输出要求：\n"
            "1. 每条步骤必须是**可立刻执行**的具体行动，不是泛泛建议\n"
            "2. 包含**具体时间、地点、路线、费用、注意事项**\n"
            "3. 结合实时数据（汇率、排队时长、列车班次）给出最优建议\n"
            "4. 如果有多个选择，明确推荐最优方案并说明理由\n"
            "5. 步骤按时间线排列，覆盖出发前准备 → 过关 → 交通 → 活动 → 返程全流程\n"
            "6. 输出 5-8 条步骤\n\n"
            "输出 JSON：{\"tasks\":[\"步骤1...\",\"步骤2...\",\"步骤3...\"]}\n"
            "每条任务格式示例：\n"
            "\"【出发准备】确认港澳通行证及签注有效，通过12306预订下周五深圳北→西九龙高铁票（14分钟, CNY 68-90），兑换HK$500现金备用（当前汇率1HKD=...CNY）\"\n"
            "\"【过关路线】8:00前抵达福田口岸（排队约10分钟），过关后乘港铁东铁线（约6分钟一班）至九龙塘站转观塘线至彩虹站（约45分钟, HK$40）\"\n"
        )
        user_prompt = (
            f"## 用户需求\n{query}\n\n"
            f"## 知识库参考\n{rag_context or '（无匹配知识库内容，请基于常识推荐）'}\n\n"
            f"## 实时工具数据\n{tool_context}\n\n"
            "请根据以上信息，生成一份详细、可执行的深港跨境规划，输出 JSON。"
        )

        tasks = self.llm_client.chat_json(system_prompt, user_prompt).get("tasks", [])
        plan = [task for task in tasks if isinstance(task, str)]
        if not plan:
            raise ServiceError("规划生成失败")
        return {"plan_steps": plan}

    def _verify(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        if any(term in query for term in SENSITIVE_TERMS):
            return {"verification_status": "需要人工复核"}
        if self.llm_client.is_configured:
            system_prompt = "你是合规审核助手，只回答 JSON。"
            user_prompt = (
                "判断以下内容是否涉及金融/法律敏感操作，"
                "输出 JSON：{'status':'pass'|'review','reason':''}。\n"
                f"{query}"
            )
            verdict = self.llm_client.chat_json(system_prompt, user_prompt)
            if verdict.get("status") == "review":
                return {"verification_status": f"需要人工复核：{verdict.get('reason', '').strip()}"}
        return {"verification_status": "已通过"}
