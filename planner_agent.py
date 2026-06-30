from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from errors import ConfigError, ServiceError
from llm_client import LLMClient
from mcp_client import MCPClient
from rag_store import RAGStore
from vision_client import VisionClient

SENSITIVE_TERMS = ["绕过外汇", "套现", "非法", "洗钱", "违规开户", "避税"]
TOOL_ORDER = ["exchange_rate", "port_traffic", "mtr_schedule", "route_planner"]
TOOL_CHOICES = set(TOOL_ORDER)
MAX_REFLECTION_ROUNDS = 2


class AgentState(TypedDict, total=False):
    user_query: str
    raw_fields: dict[str, str]
    attachment: bytes | None
    routing: dict[str, Any]
    rag_results: list[dict[str, Any]]
    schedule_events: list[dict[str, str]]
    schedule_conflicts: list[str]
    tool_results: dict[str, Any]
    plan_steps: list[str]
    verification_status: str
    reflection_count: int
    correction_notes: str
    review_detail: dict[str, Any]


@dataclass
class PlannerResult:
    plan: list[str]
    verification: str
    schedule_events: list[dict[str, str]] = field(default_factory=list)
    schedule_conflicts: list[str] = field(default_factory=list)
    data_summary: str = ""
    review_detail: dict[str, Any] = field(default_factory=dict)


class PlannerAgent:
    def __init__(
        self,
        rag_store: RAGStore,
        mcp_client: MCPClient,
        llm_client: LLMClient,
        vision_client: VisionClient | None = None,
    ) -> None:
        self.rag_store = rag_store
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.vision_client = vision_client
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_intent)
        graph.add_node("decompose", self._decompose_tasks)
        graph.add_node("execute", self._execute_actions)
        graph.add_node("plan", self._generate_plan)
        graph.add_node("review", self._review)

        graph.set_entry_point("route")
        # route → decompose → execute → plan → review
        graph.add_edge("route", "decompose")
        graph.add_edge("decompose", "execute")
        graph.add_edge("execute", "plan")
        graph.add_edge("plan", "review")

        # Reflection 循环：review → plan (修正) 或 END (完成)
        graph.add_conditional_edges(
            "review",
            self._should_reflect,
            {"plan": "plan", END: END},
        )
        return graph.compile()

    def run(
        self,
        user_query: str,
        raw_fields: dict[str, str] | None = None,
        attachment: bytes | None = None,
    ) -> PlannerResult:
        state: AgentState = {
            "user_query": user_query,
            "raw_fields": raw_fields or {},
            "attachment": attachment,
            "reflection_count": 0,
            "correction_notes": "",
        }
        result = self.graph.invoke(state)
        return PlannerResult(
            plan=result.get("plan_steps", []),
            verification=result.get("verification_status", "已通过"),
            schedule_events=result.get("schedule_events", []),
            schedule_conflicts=result.get("schedule_conflicts", []),
            data_summary=result.get("correction_notes", ""),
            review_detail=result.get("review_detail", {}),
        )

    def _route_intent(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        has_attachment = state.get("attachment") is not None

        if not self.llm_client.is_configured:
            return {
                "routing": {
                    "intent": "general",
                    "needs_rag": True,
                    "needs_vision": has_attachment,
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
            "- mtr_schedule：港铁实时到站信息（含东铁线罗湖/落马洲跨境站）\n"
            "- route_planner：跨境路线规划（含 Google Maps 实时路线）\n\n"
            "知识库包含：通关政策、口岸指南、交通攻略、海关规定、银行开户、旅游景点、目的地出行指南。\n\n"
            "判断规则：\n"
            "- 涉及过关/签证/通行证 → needs_rag=true\n"
            "- 涉及交通/路线/口岸选择 → needs_rag=true, tool_calls 含 port_traffic+mtr_schedule+route_planner\n"
            "- 涉及消费/换汇/购物 → tool_calls 含 exchange_rate\n"
            "- 涉及旅游/景点 → needs_rag=true\n"
            "- 涉及海关/违禁品/免税 → needs_rag=true\n"
            "- 涉及银行开户/金融 → needs_rag=true, needs_verification=true\n"
            "- 有图片附件 → needs_vision=true\n"
            "JSON 格式：{\"intent\":\"简短意图描述\",\"needs_rag\":true/false,\"needs_vision\":false,\"tool_calls\":[\"exchange_rate\",\"port_traffic\",\"mtr_schedule\"],\"needs_verification\":true/false}\n\n"
            f"用户需求：{query}\n"
            f"包含图片附件：{'是' if has_attachment else '否'}"
        )
        try:
            route = self.llm_client.chat_json(system_prompt, user_prompt)
            route["tool_calls"] = [tool for tool in route.get("tool_calls", []) if tool in TOOL_CHOICES]
        except ServiceError:
            route = {
                "intent": "general",
                "needs_rag": True,
                "needs_vision": has_attachment,
                "tool_calls": TOOL_ORDER,
                "needs_verification": True,
            }
        if has_attachment:
            route["needs_vision"] = True
        return {"routing": route}

    def _decompose_tasks(self, state: AgentState) -> AgentState:
        """将复杂需求拆解为逻辑子任务，便于后续 RAG 检索和工具调用更有针对性。"""
        query = state.get("user_query", "")
        route = state.get("routing", {})

        # 简单需求跳过拆解
        if not self.llm_client.is_configured or len(query) < 20:
            return {"routing": route}

        try:
            system_prompt = (
                "你是深港跨境任务拆解引擎。将用户复杂需求分解为 2-4 个独立子任务，"
                "每个子任务应是一个可用于搜索引擎检索的**具体查询**。输出 JSON。"
            )
            user_prompt = (
                f"用户需求：{query}\n"
                f"意图：{route.get('intent', 'general')}\n\n"
                "拆解要求：\n"
                "- 每个子任务必须是可检索的关键词查询（如「香港科技大学交通路线」而非「规划交通」）\n"
                "- 覆盖所有用户提到的需求维度（交通、开户、旅游、海关等）\n"
                "- 格式：{\"subtasks\":[\"查询1\",\"查询2\",\"查询3\"]}\n"
                "示例：\n"
                "输入「去港科大打比赛顺便开户旅游」→ [\"香港科技大学交通路线\",\"香港虚拟银行开户材料\",\"香港半日游景点推荐\"]"
            )
            result = self.llm_client.chat_json(system_prompt, user_prompt)
            subtasks = result.get("subtasks", [query])
            # 将子任务挂到 routing 中，generate_plan 阶段会用到
            route["subtasks"] = subtasks if isinstance(subtasks, list) else [query]
            return {"routing": route}
        except ServiceError:
            return {"routing": route}

    def _execute_actions(self, state: AgentState) -> AgentState:
        route = state.get("routing", {})
        query = state.get("user_query", "")
        attachment = state.get("attachment")

        # ── Vision 日程解析 ──
        schedule_events: list[dict[str, str]] = []
        schedule_conflicts: list[str] = []
        if route.get("needs_vision") and attachment and self.vision_client:
            try:
                schedule_events = self.vision_client.parse_events(attachment)
            except (ConfigError, ServiceError):
                pass

        # ── RAG 检索（子任务驱动） ──
        rag_results: list[dict[str, Any]] = []
        if route.get("needs_rag"):
            subtasks = route.get("subtasks", [query])
            seen_sources: set[str] = set()
            for sub in (subtasks if isinstance(subtasks, list) else [query]):
                for result in self.rag_store.search(str(sub)):
                    src = result.metadata.get("source", "") or ""
                    if src not in seen_sources:
                        seen_sources.add(src)
                        rag_results.append({
                            "document": result.document,
                            "metadata": result.metadata,
                            "distance": result.distance,
                        })
            # 按 distance 排序，取前 5 篇
            rag_results.sort(key=lambda r: r.get("distance") or 999)
            rag_results = rag_results[:5]

        # ── MCP 工具调用 ──
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
                    for port_name in ["深圳湾", "罗湖", "落马洲"]:
                        try:
                            pt = self.mcp_client.get_port_traffic(port_name)
                            key = f"port_{port_name}" if port_name != "深圳湾" else "port_traffic"
                            tool_results[key] = {
                                "port": pt.port,
                                "queue_minutes": pt.queue_minutes,
                                "summary": f"{pt.port}：排队约{pt.queue_minutes}分钟，{pt.note}",
                            }
                        except Exception:
                            pass
                elif tool == "mtr_schedule":
                    for st in ["罗湖", "落马洲"]:
                        try:
                            mtr = self.mcp_client.get_mtr_schedule(st)
                            trains_text = ""
                            if mtr.next_trains:
                                trains_text = " | ".join(
                                    f"[{t['direction']}] {t['dest']} {t['platform']}月台 {t['arrive_in']}"
                                    for t in mtr.next_trains
                                )
                            tool_results[f"mtr_{st}"] = {
                                "station": mtr.station,
                                "interval_minutes": mtr.interval_minutes,
                                "summary": f"{mtr.station}：约{mtr.interval_minutes}分钟一班。{trains_text}",
                            }
                        except Exception:
                            pass
                elif tool == "route_planner":
                    # 从用户需求中提取目的地用于路线规划
                    dest = state.get("raw_fields", {}).get("dest", "西九龙")
                    origin = "福田"  # 默认深圳出发
                    try:
                        route = self.mcp_client.get_route(origin, dest)
                        route_lines = [
                            f"{r.mode}：约{r.duration_min}分钟 HK${r.cost_hkd:.0f}"
                            for r in route.routes[:2]
                        ]
                        tool_results["route_planner"] = {
                            "origin": origin,
                            "destination": dest,
                            "source": route.source,
                            "summary": f"{origin}→{dest}（{route.source}）：" + " | ".join(route_lines),
                        }
                    except (ConfigError, ServiceError):
                        pass
            except (ConfigError, ServiceError) as exc:
                tool_results[tool] = {"error": str(exc), "summary": f"获取失败：{exc}"}

        # ── 日程冲突检测 ──
        if schedule_events:
            schedule_conflicts = self._detect_schedule_conflicts(schedule_events)

        return {
            "rag_results": rag_results,
            "tool_results": tool_results,
            "schedule_events": schedule_events,
            "schedule_conflicts": schedule_conflicts,
        }

    @staticmethod
    def _detect_schedule_conflicts(events: list[dict[str, str]]) -> list[str]:
        """检测日程时间冲突。"""
        conflicts: list[str] = []
        for i, a in enumerate(events):
            for b in events[i + 1:]:
                if a.get("date") == b.get("date") and a.get("time") == b.get("time"):
                    conflicts.append(
                        f"{a.get('date', '?')} {a.get('time', '?')}: "
                        f"「{a.get('title', '?')}」与「{b.get('title', '?')}」时间冲突"
                    )
        return conflicts

    def _generate_plan(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        correction = state.get("correction_notes", "")
        schedule_events = state.get("schedule_events", [])
        schedule_conflicts = state.get("schedule_conflicts", [])

        if not self.llm_client.is_configured:
            return {
                "plan_steps": [
                    "请配置 LLM API Key 以启用智能规划。",
                    "若已配置，可重新提交需求生成分步计划。",
                ]
            }

        # 组织知识库内容
        rag_context = "\n".join(
            f"📚 {item['metadata'].get('title', '参考资料')}：{item['document'][:800]}"
            for item in state.get("rag_results", [])[:5]
        )

        # 组织实时工具结果
        tool_parts = []
        for key, value in state.get("tool_results", {}).items():
            if isinstance(value, dict):
                tool_parts.append(f"- {value.get('summary', str(value))}")
            else:
                tool_parts.append(f"- {key}: {value}")
        tool_context = "\n".join(tool_parts) if tool_parts else "暂无实时数据"

        # 日程信息
        schedule_text = ""
        if schedule_events:
            schedule_text = "\n".join(
                f"- {e.get('date', '?')} {e.get('time', '?')}：{e.get('title', '?')}"
                for e in schedule_events
            )
        conflict_text = "\n".join(f"- {c}" for c in schedule_conflicts) if schedule_conflicts else "无"

        # 修正提示
        correction_text = f"\n\n⚠️ 上一版规划的问题，请修正：{correction}" if correction else ""

        system_prompt = (
            "你是深港跨境生活规划专家。你拥有丰富的深港双城出行经验，熟悉各口岸特点、"
            "交通路线、景点分布、海关规定。\n\n"
            "你的任务是：根据用户需求 + 知识库参考 + 实时工具数据 + 日程信息，生成一份**具体、可执行、有细节**的分步计划。\n\n"
            "输出要求：\n"
            "1. 每条步骤必须是**可立刻执行**的具体行动，不是泛泛建议\n"
            "2. 包含**具体时间、地点、路线、费用、注意事项**\n"
            "3. 结合实时数据（汇率、排队时长、列车班次）给出最优建议\n"
            "4. 如果有多个选择，明确推荐最优方案并说明理由\n"
            "5. 步骤按时间线排列，覆盖出发前准备 → 过关 → 交通 → 活动 → 返程全流程\n"
            "6. 如果含开户需求，必须提醒携带港澳通行证+身份证+入境小票\n"
            "7. 如果含购物/消费，必须结合当前汇率给出人民币参考价\n"
            "8. 必须标注返程时口岸关闭时间和港铁末班车\n"
            "9. 输出 5-8 条步骤\n\n"
            "⚠️ 重要：\n"
            "- 如果知识库和工具数据中没有某地点/路线的确切信息，必须在对应步骤开头标注 ⚠️[待确认]\n"
            "- 不要编造不存在的地名、公交线路、餐厅名称\n"
            "- 口岸、港铁站名必须来自实时工具数据或知识库，禁止臆造\n\n"
            "输出 JSON：{\"tasks\":[\"步骤1...\",\"步骤2...\",\"步骤3...\"]}\n\n"
            "示例1（出行+比赛+旅游）：\n"
            "\"【出发准备】确认港澳通行证及签注，下载MTR Mobile，兑换HK$500（1HKD=0.87CNY≈¥435），开通手机漫游\"\n"
            "\"【过关路线】8:00前抵罗湖口岸（排队约15分）→东铁线至九龙塘（35分,HK$40）→观塘线至彩虹（10分）→11M小巴至科大（20分,HK$8），总约1.5h\"\n"
            "\"【比赛】14:00-18:00参赛，赛后参观蘑菇观景台\"\n"
            "\"【旅游】18:30乘91M至彩虹→港铁至尖沙咀，天星小轮($4)→太平山顶缆车($88往返)→庙街夜市($80-120)\"\n"
            "\"【返程】21:30前离开尖沙咀→东铁线回罗湖(50分,末班23:00)，口岸24:00关闭，确保23:00前过关\"\n\n"
            "示例2（开户）：\n"
            "\"【开户准备】携带港澳通行证+内地身份证，过关时保留白色入境小票，下载ZA Bank App（最便捷虚拟银行，0存款0管理费）\"\n"
            "\"【过关】福田口岸→落马洲站→东铁线至九龙塘，全程约40分(HK$40)，到港后打开ZA Bank App立即开始开户流程\"\n"
            "\"【开户操作】身处香港境内时(GPS验证)，拍摄证件→人脸识别→填写信息（用途填跨境消费/投资理财）→即时开通，全程约15分钟\"\n\n"
            "示例3（海关/购物）：\n"
            "\"【海关注意】免税额度：1L烈酒(>30度)+19支香烟；携带现金≥HK$120,000必须红通道申报，违者最高罚HK$500,000+监禁2年\"\n"
            "\"【购物预算】维港周边商场支持支付宝/微信（按实时汇率结算），茶餐厅备HK$200现金（部分仅收现金），八达通充值HK$100\"\n"
        )
        # 子任务拆解信息
        subtasks = state.get("routing", {}).get("subtasks", [])
        subtask_text = ""
        if subtasks:
            subtask_text = "\n".join(f"- {s}" for s in subtasks)
            subtask_text = f"## 任务拆解\n{subtask_text}\n请按子任务逐一覆盖。\n\n"

        user_prompt = (
            f"## 用户需求\n{query}\n\n"
            f"{subtask_text}"
            f"## 日程信息（从海报/截图解析）\n{schedule_text or '（无外部日程）'}\n"
            f"## 日程冲突\n{conflict_text}\n\n"
            f"## 知识库参考\n{rag_context or '（无匹配知识库内容，请基于常识推荐）'}\n\n"
            f"## 实时工具数据\n{tool_context}\n"
            f"{correction_text}\n\n"
            "请根据以上信息，生成一份详细、可执行的深港跨境规划，输出 JSON。"
        )

        try:
            tasks = self.llm_client.chat_json(system_prompt, user_prompt).get("tasks", [])
        except ServiceError:
            # LLM 调用失败时降级为基于 RAG 数据的简化规划
            if rag_context:
                return {
                    "plan_steps": [
                        f"（AI 服务暂时不可用，以下是基于知识库的参考信息）",
                        *[f"📚 {item['metadata'].get('title', '?')}：{item['document'][:200]}..."
                          for item in state.get("rag_results", [])[:3]],
                    ]
                }
            raise
        plan = [task for task in tasks if isinstance(task, str)]
        if not plan:
            raise ServiceError("规划生成失败")
        return {"plan_steps": plan}

    def _review(self, state: AgentState) -> AgentState:
        """两层审核：①硬合规红线 ②质量评审（含地点验证）。"""
        query = state.get("user_query", "")
        plan = state.get("plan_steps", [])
        reflection_count = state.get("reflection_count", 0)

        # ── 第①层：合规红线（规则，不消耗 LLM） ──
        if any(term in query for term in SENSITIVE_TERMS):
            return {"verification_status": "⛔ 需人工审核：命中敏感词"}

        if not self.llm_client.is_configured:
            return {"verification_status": "✅ 通过"}

        # ── 地点验证（Google Geocoding） ──
        geo_results: dict[str, Any] = {}
        geo_verified: list[dict] = []
        geo_unverified: list[dict] = []
        try:
            geo_results = self.mcp_client.batch_geocode_places(plan)
            for name, g in geo_results.items():
                if g.found:
                    geo_verified.append({"name": name, "address": g.formatted_address, "type": g.place_type})
                else:
                    geo_unverified.append({"name": name, "query": g.query})
        except Exception:
            pass

        geo_note = ""
        if geo_unverified:
            names = [g["name"] for g in geo_unverified[:10]]
            geo_note = f"以下 {len(geo_unverified)} 个地名未在 Google Maps 中找到（可能是编造或拼写错误）：{', '.join(names)}"

        # ── 第②层：LLM 质量评审 ──
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        system_prompt = (
            "你是深港跨境规划审核员。严格审查规划，只输出 JSON。\n"
            "① 硬合规红线：涉及洗钱/违规开户/绕过外汇 → status=block\n"
            "② 质量评审（status=pass/review）：\n"
            "   - 地点是否编造？（如有「未找到」的地名，必须要求修正）\n"
            "   - 步骤是否有具体时间/路线/费用？\n"
            "   - 是否遗漏返程安排？\n"
            "status=review 时 corrections 必填一句话修改指令。"
        )
        user_prompt = (
            f"## 用户需求\n{query}\n\n"
            f"## 规划\n{plan_text}\n\n"
            f"## 地点验证\n{geo_note or '✅ 所有地点均已通过 Google Maps 验证'}\n\n"
            "输出 JSON：{\"status\":\"pass\"|\"review\"|\"block\",\"reason\":\"\",\"corrections\":\"\"}"
        )

        try:
            verdict = self.llm_client.chat_json(system_prompt, user_prompt)
        except ServiceError:
            return {
                "verification_status": "✅ 通过",
                "review_detail": {
                    "geo_verified": geo_verified,
                    "geo_unverified": geo_unverified,
                    "llm_status": "pass",
                    "llm_reason": "",
                },
            }

        status = verdict.get("status", "pass")
        reason = verdict.get("reason", "")
        corrections = verdict.get("corrections", "")

        # 构建审核详情
        review_detail: dict[str, Any] = {
            "geo_verified": geo_verified,
            "geo_unverified": geo_unverified,
            "llm_status": status,
            "llm_reason": reason or ("无问题" if status == "pass" else ""),
            "red_line_check": "pass",
        }

        if status == "block":
            return {
                "verification_status": f"⛔ 不合规：{reason or '建议被驳回'}",
                "review_detail": review_detail,
            }

        if status == "review" and not corrections and reason:
            corrections = f"请修正：{reason}"

        if status == "review" and corrections and reflection_count < MAX_REFLECTION_ROUNDS:
            note = f"🔄 第{reflection_count + 1}轮修正"
            if geo_unverified:
                note += f" | {len(geo_unverified)}个地名待确认"
            return {
                "verification_status": note,
                "correction_notes": corrections,
                "reflection_count": reflection_count + 1,
                "review_detail": review_detail,
            }

        if status == "review":
            return {
                "verification_status": f"⚠️ 需人工确认：{reason or '审核未通过'}",
                "review_detail": review_detail,
            }

        return {
            "verification_status": "✅ 通过",
            "review_detail": review_detail,
        }

    def _should_reflect(self, state: AgentState) -> str:
        """决定是否需要 Reflection 修正循环。"""
        count = state.get("reflection_count", 0)
        notes = state.get("correction_notes", "")
        if notes and count < MAX_REFLECTION_ROUNDS:
            return "plan"
        return END
