from __future__ import annotations

import re
from pathlib import Path

import gradio as gr

from config import AppConfig
from errors import ConfigError, ServiceError
from llm_client import LLMClient
from mcp_client import MCPClient
from planner_agent import PlannerAgent, SENSITIVE_TERMS
from rag_store import RAGStore
from vision_client import VisionClient

DESCRIPTION = """
SZ-HK Hub 基于 LangGraph + 高级 RAG + MCP 工具链，为深港双城生活提供可验证、可追踪、可落地的决策支持。
需要配置 LLM API 与 MCP 服务接口以启用全量能力。
""".strip()


config = AppConfig.from_env()
llm_client = LLMClient(config)
rag_store = RAGStore(config)
mcp_client = MCPClient(config)
planner_agent = PlannerAgent(rag_store, mcp_client, llm_client)
vision_client = VisionClient(config)


def format_list(items: list[str], empty_message: str) -> str:
    if not items:
        return empty_message
    return "\n".join(f"- {item}" for item in items)


def has_sensitive(text: str) -> bool:
    return any(term in text for term in SENSITIVE_TERMS)


def parse_events(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    date_patterns = [
        r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        r"(\d{1,2}[/-]\d{1,2})",
        r"(\d{1,2})月(\d{1,2})日",
    ]
    time_pattern = r"([01]?\d|2[0-3]):([0-5]\d)"
    events = []
    for line in lines:
        date = "待确认日期"
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                candidate = match.group(0)
                if pattern == date_patterns[2]:
                    month = int(match.group(1))
                    day = int(match.group(2))
                else:
                    parts = re.split(r"[/-]", candidate)
                    if len(parts) < 2:
                        continue
                    month = int(parts[-2])
                    day = int(parts[-1])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    date = candidate
                    break
        time_match = re.search(time_pattern, line)
        time = time_match.group(0) if time_match else "待确认时间"
        title = line.replace(date, "").replace(time, "").strip() or "未命名活动"
        events.append({"date": date, "time": time, "title": title})
    return events


def detect_conflicts(events: list[dict[str, str]]) -> list[str]:
    seen = set()
    conflicts = []
    for event in events:
        key = f"{event['date']}-{event['time']}"
        if key in seen:
            conflicts.append(key)
        seen.add(key)
    return conflicts


def handle_planner(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "请输入跨境需求描述。", "未提交"
    try:
        result = planner_agent.run(text)
    except ConfigError as exc:
        return format_list([f"配置缺失：{exc}"], "暂无规划结果。"), "需要配置"
    except ServiceError as exc:
        return format_list([f"服务异常：{exc}"], "暂无规划结果。"), "需要人工复核"
    return format_list(result.plan, "暂无规划结果。"), result.verification


def handle_events(text: str, file_data: bytes | None) -> tuple[str, str]:
    text = text.strip()
    events = []
    error_note = ""
    if file_data:
        try:
            events = vision_client.parse_events(file_data)
        except (ConfigError, ServiceError) as exc:
            error_note = f"视觉解析失败：{exc}"
    if not events and text:
        events = parse_events(text)
    if not events:
        return "暂无活动解析结果。", error_note or "未检测到明显冲突。"
    event_lines = [f"{event['date']} {event['time']} · {event['title']}" for event in events]
    conflicts = detect_conflicts(events)
    conflict_message = (
        f"检测到 {len(conflicts)} 条时间冲突，请调整日程。" if conflicts else "未检测到明显冲突。"
    )
    if error_note:
        conflict_message = f"{error_note} / {conflict_message}"
    return format_list(event_lines, "暂无活动解析结果。"), conflict_message


def describe_event_file(file_data: bytes | None) -> str:
    if not file_data:
        return "未选择活动截图。"
    return "已上传活动截图。"


def _fetch_metrics() -> tuple[str, str, str, int | None]:
    rate_text = "汇率数据未配置"
    queue_text = "深圳湾 数据未配置"
    mtr_text = "西九龙 数据未配置"
    queue_value = None
    try:
        rate = mcp_client.get_exchange_rate()
        rate_text = f"1 {rate.base} = {rate.rate:.2f} {rate.target}"
    except (ConfigError, ServiceError):
        rate_text = "汇率数据不可用"
    try:
        port = mcp_client.get_port_traffic("深圳湾")
        queue_value = port.queue_minutes
        queue_text = f"{port.port} {port.queue_minutes} 分钟"
    except (ConfigError, ServiceError):
        queue_text = "深圳湾 数据不可用"
    try:
        mtr = mcp_client.get_mtr_schedule("西九龙")
        mtr_text = f"{mtr.station} {mtr.interval_minutes} 分钟一班"
    except (ConfigError, ServiceError):
        mtr_text = "西九龙 数据不可用"
    return rate_text, queue_text, mtr_text, queue_value


def refresh_overview() -> tuple[str, str, str]:
    rate_text, queue_text, mtr_text, _ = _fetch_metrics()
    return rate_text, queue_text, mtr_text


def refresh_decision() -> tuple[str, str, str, str]:
    rate_text, queue_text, mtr_text, queue_value = _fetch_metrics()
    if queue_value is None:
        route = "推荐路线：请先配置口岸实时数据"
    elif queue_value < 25:
        route = "推荐路线：深圳湾口岸 → 西九龙高铁站"
    else:
        route = "推荐路线：福田口岸 → 港铁东铁线"
    return rate_text, queue_text, mtr_text, route


def search_rag(query: str) -> str:
    trimmed = query.strip()
    if not trimmed:
        return "请输入关键词以检索政策或开户信息。"
    results = rag_store.search(trimmed)
    if not results:
        if not rag_store.has_documents():
            return "知识库为空，请先在 rag_corpus/ 补充资料并运行 rag_ingest.py。"
        return "暂无匹配文档，请尝试其它关键词。"
    lines = []
    for result in results:
        source = result.metadata.get("source") or ""
        source_name = Path(source).name if source else ""
        title = result.metadata.get("title") or source_name or "参考资料"
        snippet = result.document.replace("\n", " ").strip()
        lines.append(f"{title}：{snippet[:120]}...")
    return format_list(lines, "暂无匹配文档，请尝试其它关键词。")


def update_safety(text: str) -> str:
    if not text.strip():
        return "系统将自动标记敏感金融/法律问题。"
    if has_sensitive(text):
        return "检测到敏感问题，请人工复核并遵守合规要求。"
    return "未发现明显敏感词，可继续智能处理。"


initial_overview = refresh_overview()
initial_decision = refresh_decision()

with gr.Blocks(title="SZ-HK Hub · 深港跨境专业生活助手") as demo:
    gr.Markdown("# SZ-HK Hub · 深港跨境专业生活助手")
    gr.Markdown(DESCRIPTION)

    gr.Markdown("## 今日跨境概览")
    with gr.Row():
        overview_rate = gr.Textbox(label="实时汇率", value=initial_overview[0], interactive=False)
        overview_queue = gr.Textbox(label="口岸人流", value=initial_overview[1], interactive=False)
        overview_mtr = gr.Textbox(label="港铁", value=initial_overview[2], interactive=False)
    overview_refresh = gr.Button("刷新概览")
    overview_refresh.click(
        fn=refresh_overview,
        outputs=[overview_rate, overview_queue, overview_mtr],
    )

    gr.Markdown("## 体验工作台")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Planner + Verifier (LangGraph)")
            planner_input = gr.Textbox(
                label="跨境需求描述",
                placeholder="例如：下周六去西九龙打卡并参加比赛，顺便在 ZA Bank 开户。",
                lines=4,
            )
            planner_button = gr.Button("生成规划")
            planner_output = gr.Markdown("暂无规划结果。")
            verify_status = gr.Textbox(label="校验状态", value="已通过", interactive=False)
            planner_button.click(
                fn=handle_planner,
                inputs=planner_input,
                outputs=[planner_output, verify_status],
            )

        with gr.Column():
            gr.Markdown("### 活动/情报解析 (Vision)")
            event_file = gr.File(label="上传活动截图", file_count="single", type="binary")
            event_file_note = gr.Textbox(value="未选择活动截图。", interactive=False)
            event_file.change(
                fn=describe_event_file,
                inputs=event_file,
                outputs=event_file_note,
            )
            event_input = gr.Textbox(
                label="活动文案",
                placeholder="例如：5月18日 10:00 西九龙篮球赛 / 5月18日 11:00 开户",
                lines=4,
            )
            event_button = gr.Button("解析日程")
            event_output = gr.Markdown("暂无活动解析结果。")
            event_conflict = gr.Textbox(value="未检测到明显冲突。", interactive=False)
            event_button.click(
                fn=handle_events,
                inputs=[event_input, event_file],
                outputs=[event_output, event_conflict],
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 实时决策支持 (MCP)")
            decision_rate = gr.Textbox(label="汇率", value=initial_decision[0], interactive=False)
            decision_queue = gr.Textbox(label="深圳湾口岸", value=initial_decision[1], interactive=False)
            decision_mtr = gr.Textbox(label="西九龙港铁", value=initial_decision[2], interactive=False)
            decision_route = gr.Textbox(label="推荐路线", value=initial_decision[3], interactive=False)
            decision_refresh = gr.Button("更新数据")
            decision_refresh.click(
                fn=refresh_decision,
                outputs=[decision_rate, decision_queue, decision_mtr, decision_route],
            )

        with gr.Column():
            gr.Markdown("### RAG 知识检索 (Vector DB)")
            rag_query = gr.Textbox(label="政策/开户问题", placeholder="例如：ZA Bank 开户材料")
            rag_button = gr.Button("检索文档")
            rag_results = gr.Markdown("暂无检索结果。")
            rag_button.click(fn=search_rag, inputs=rag_query, outputs=rag_results)

        with gr.Column():
            gr.Markdown("### 安全护栏检测")
            safety_input = gr.Textbox(label="输入敏感问题", placeholder="例如：如何绕过外汇限制？")
            safety_output = gr.Textbox(value=update_safety(""), interactive=False)
            safety_input.input(fn=update_safety, inputs=safety_input, outputs=safety_output)


if __name__ == "__main__":
    demo.launch()
