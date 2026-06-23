from __future__ import annotations

import re
from pathlib import Path

import gradio as gr
import requests

from config import AppConfig
from errors import ConfigError, ServiceError
from llm_client import LLMClient
from mcp_client import MCPClient
from planner_agent import PlannerAgent, SENSITIVE_TERMS
from rag_store import RAGStore
from vision_client import VisionClient

DESCRIPTION = "深港双城 · 一站规划 | 实时口岸数据 + 港铁动态 + RAG 知识库 + AI 规划引擎"

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
/* header */
.header-bar { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.5rem; background: #0f766e; border-radius: 8px; color: #fff; margin-bottom: 1.2rem; }
.header-bar .icon { font-size: 2rem; }
.header-bar h1 { font-size: 1.4rem; margin: 0; font-weight: 700; color: #fff !important; }
.header-bar .sub { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }
/* 数据卡片行 */
.metrics-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metrics-row > * { flex: 1; min-width: 140px; }
.metric-card { text-align: center; padding: 0.7rem 0.5rem; border-radius: 8px; border: 1px solid #d1d5db; background: #fff; }
.metric-card .label { font-size: 0.72rem; color: #6b7280; margin-bottom: 2px; }
.metric-card .value { font-size: 1.05rem; font-weight: 700; color: #1f2937; }
.metric-card.accent { border-left: 4px solid #0f766e; }
.metric-card.warn  { border-left: 4px solid #d97706; }
.metric-card.info  { border-left: 4px solid #2563eb; }
/* tab 内容区 */
.tab-content { padding: 0.5rem 0; }
.planner-row { display: flex; gap: 1rem; align-items: flex-start; }
.planner-row .input-col { flex: 3; }
.planner-row .action-col { flex: 1; display: flex; flex-direction: column; gap: 0.6rem; }
/* 规划步骤 */
.plan-step { padding: 0.55rem 0.9rem; margin: 0.3rem 0; border-left: 3px solid #0f766e; background: #f0fdf4; border-radius: 0 6px 6px 0; font-size: 0.9rem; line-height: 1.5; }
/* 杂项 */
footer { visibility: hidden; }
.gradio-button-primary { background: #0f766e !important; }
"""

config = AppConfig.from_env()
llm_client = LLMClient(config)
rag_store = RAGStore(config)
mcp_client = MCPClient(config)
planner_agent = PlannerAgent(rag_store, mcp_client, llm_client)
vision_client = VisionClient(config)


def format_list(items: list[str], empty_message: str) -> str:
    if not items:
        return f"<em>{empty_message}</em>"
    return "\n".join(f"<div class='plan-step'>📍 {item}</div>" for item in items)


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
        return "<em>请输入跨境需求描述。</em>", "⏳ 等待提交"
    try:
        result = planner_agent.run(text)
    except ConfigError as exc:
        return format_list([f"⚠️ 配置缺失：{exc}"], "暂无规划结果。"), "⚠️ 需要配置"
    except ServiceError as exc:
        return format_list([f"⚠️ 服务异常：{exc}"], "暂无规划结果。"), "⚠️ 需要人工复核"
    return format_list(result.plan, "暂无规划结果。"), f"✅ {result.verification}"


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
        f"⚠️ 检测到 {len(conflicts)} 条时间冲突，请调整日程。" if conflicts else "✅ 未检测到明显冲突。"
    )
    if error_note:
        conflict_message = f"{error_note} / {conflict_message}"
    return format_list(event_lines, "暂无活动解析结果。"), conflict_message


def describe_event_file(file_data: bytes | None) -> str:
    if not file_data:
        return "📷 未选择截图"
    return "📷 已上传截图"


def _fetch_metrics() -> dict:
    result: dict = {
        "rate": "—", "rate_detail": "",
        "sz_bay": "—", "sz_bay_detail": "",
        "lo_wu": "—", "lo_wu_detail": "",
        "lMc": "—", "lMc_detail": "",
        "mtr": "—", "mtr_detail": "",
    }
    try:
        rate = mcp_client.get_exchange_rate()
        result["rate"] = f"{rate.rate:.4f}"
        result["rate_detail"] = f"1 HKD = {rate.rate:.4f} CNY"
    except Exception:
        result["rate"] = "—"
        result["rate_detail"] = "数据不可用"

    for port_key, port_name in [("sz_bay", "深圳湾"), ("lo_wu", "罗湖"), ("lMc", "落马洲")]:
        try:
            pt = mcp_client.get_port_traffic(port_name)
            result[port_key] = f"{pt.queue_minutes}分钟"
            result[f"{port_key}_detail"] = f"{port_name}：约{pt.queue_minutes}分钟（{pt.note}）"
        except Exception:
            result[port_key] = "—"
            result[f"{port_key}_detail"] = f"{port_name}：数据不可用"

    try:
        mtr = mcp_client.get_mtr_schedule("罗湖")
        result["mtr"] = f"约{mtr.interval_minutes}分钟/班"
        result["mtr_detail"] = f"东铁线罗湖：约{mtr.interval_minutes}分钟一班"
    except Exception:
        result["mtr"] = "—"
        result["mtr_detail"] = "数据不可用"

    return result


def initial_metrics() -> dict:
    return _fetch_metrics()


def search_rag(query: str) -> str:
    trimmed = query.strip()
    if not trimmed:
        return "<em>请输入关键词以检索政策或开户信息。</em>"
    if rag_store._init_error:
        return f"⚠️ 知识库暂不可用：{rag_store._init_error}"
    results = rag_store.search(trimmed)
    if not results:
        if not rag_store.has_documents():
            return "⚠️ 知识库为空，请先运行 <code>python rag_ingest.py</code> 导入文档。"
        return "暂无匹配文档，请尝试其它关键词。"

    context_parts: list[str] = []
    for r in results:
        src = r.metadata.get("source", "")
        fname = Path(src).name if src else "参考资料"
        context_parts.append(f"【来源：{fname}】\n{r.document}")
    context = "\n\n---\n\n".join(context_parts)

    if llm_client.is_configured:
        try:
            system_prompt = (
                "你是深港跨境生活助手。请根据以下参考资料，用简洁中文回答用户问题。"
                "回答控制在 3-5 句话以内，直接给出关键信息，不要罗列来源。"
                "如果参考资料不足以回答问题，请如实说明。"
            )
            user_prompt = f"参考资料：\n{context}\n\n用户问题：{trimmed}"
            answer = llm_client.chat(system_prompt, user_prompt)
            return answer.strip()
        except Exception:
            pass

    lines = []
    for result in results:
        source = result.metadata.get("source") or ""
        source_name = Path(source).name if source else ""
        title = result.metadata.get("title") or source_name or "参考资料"
        snippet = result.document.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"【{title}】{snippet}")
    return format_list(lines, "暂无匹配文档。")


def update_safety(text: str) -> str:
    if not text.strip():
        return "🔍 系统将自动标记敏感金融/法律问题。"
    if has_sensitive(text):
        return "🔴 检测到敏感问题，请人工复核并遵守合规要求。"
    return "🟢 未发现明显敏感词，可继续智能处理。"


# ── 初始数据 ──
m = initial_metrics()

with gr.Blocks(title="SZ-HK Hub · 深港跨境生活助手", css=CSS, theme=gr.themes.Soft()) as demo:
    # ── Header ──
    gr.HTML(f"""
    <div class="header-bar">
        <span class="icon">🇭🇰</span>
        <div>
            <h1>SZ-HK Hub · 深港跨境生活助手</h1>
            <div class="sub">{DESCRIPTION}</div>
        </div>
    </div>
    """)

    # ── 实时数据卡片 ──
    with gr.Row(elem_classes="metrics-row"):
        rate_disp = gr.HTML(f"<div class='metric-card info'><div class='label'>💱 HKD → CNY</div><div class='value'>{m['rate']}</div></div>")
        szb_disp = gr.HTML(f"<div class='metric-card'><div class='label'>🚌 深圳湾口岸</div><div class='value'>{m['sz_bay']}</div></div>")
        lw_disp = gr.HTML(f"<div class='metric-card'><div class='label'>🚆 罗湖口岸</div><div class='value'>{m['lo_wu']}</div></div>")
        lmc_disp = gr.HTML(f"<div class='metric-card'><div class='label'>🚆 落马洲口岸</div><div class='value'>{m['lMc']}</div></div>")
        mtr_disp = gr.HTML(f"<div class='metric-card info'><div class='label'>🚇 东铁线班次</div><div class='value'>{m['mtr']}</div></div>")

    refresh_btn = gr.Button("🔄 刷新实时数据", variant="secondary", size="sm")

    # ── Tab 主体 ──
    with gr.Tabs():
        # Tab 1: 智能规划
        with gr.Tab("🧠 智能规划"):
            with gr.Row():
                with gr.Column(scale=3, elem_classes="input-col"):
                    planner_input = gr.Textbox(
                        label="描述你的跨境需求",
                        placeholder="例如：下周六去香港科大打比赛，2点到6点，顺便旅游，晚上回来",
                        lines=4,
                    )
                with gr.Column(scale=1, elem_classes="action-col"):
                    planner_button = gr.Button("🚀 生成规划", variant="primary", size="lg")
                    verify_status = gr.Textbox(label="校验", value="—", interactive=False, show_label=True)
            planner_output = gr.HTML("<em style='color:#6b7280;'>输入需求后点击「生成规划」，AI 将结合实时数据为你定制详细方案。</em>")
            planner_button.click(
                fn=handle_planner,
                inputs=planner_input,
                outputs=[planner_output, verify_status],
            )

        # Tab 2: 知识检索
        with gr.Tab("📚 知识检索"):
            with gr.Row():
                rag_query = gr.Textbox(
                    label="搜索政策、开户、旅游、交通信息",
                    placeholder="例如：ZA Bank 开户材料 | 香港科大怎么去 | 免税额度",
                    scale=4,
                )
                rag_button = gr.Button("🔍 检索", variant="primary", scale=1)
            rag_results = gr.HTML("<em style='color:#6b7280;'>输入关键词检索知识库。</em>")
            rag_button.click(fn=search_rag, inputs=rag_query, outputs=rag_results)

        # Tab 3: 日程解析
        with gr.Tab("📅 日程解析"):
            with gr.Row():
                with gr.Column(scale=3):
                    event_input = gr.Textbox(
                        label="粘贴活动文案",
                        placeholder="例如：6月28日 14:00 港科大篮球赛\n6月28日 18:00 尖沙咀晚餐",
                        lines=4,
                    )
                with gr.Column(scale=1):
                    event_file = gr.File(label="或上传截图", file_count="single", type="binary")
                    event_file_note = gr.Markdown("📷 未选择截图")
                    event_file.change(fn=describe_event_file, inputs=event_file, outputs=event_file_note)
            event_button = gr.Button("📋 解析日程", variant="primary")
            event_output = gr.HTML("<em style='color:#6b7280;'>待解析。</em>")
            event_conflict = gr.Textbox(value="✅ 未检测到冲突。", interactive=False)
            event_button.click(
                fn=handle_events,
                inputs=[event_input, event_file],
                outputs=[event_output, event_conflict],
            )

        # Tab 4: 合规检测
        with gr.Tab("🛡️ 合规检测"):
            safety_input = gr.Textbox(
                label="输入需检测的内容",
                placeholder="例如：如何绕过外汇限制？",
                lines=2,
            )
            safety_output = gr.Textbox(value="🔍 系统将自动标记敏感金融/法律问题。", interactive=False)
            safety_input.input(fn=update_safety, inputs=safety_input, outputs=safety_output)

    # ── 刷新回调 ──
    def refresh_all():
        m2 = _fetch_metrics()
        return [
            f"<div class='metric-card info'><div class='label'>💱 HKD → CNY</div><div class='value'>{m2['rate']}</div></div>",
            f"<div class='metric-card'><div class='label'>🚌 深圳湾口岸</div><div class='value'>{m2['sz_bay']}</div></div>",
            f"<div class='metric-card'><div class='label'>🚆 罗湖口岸</div><div class='value'>{m2['lo_wu']}</div></div>",
            f"<div class='metric-card'><div class='label'>🚆 落马洲口岸</div><div class='value'>{m2['lMc']}</div></div>",
            f"<div class='metric-card info'><div class='label'>🚇 东铁线班次</div><div class='value'>{m2['mtr']}</div></div>",
        ]

    refresh_btn.click(
        fn=refresh_all,
        outputs=[rate_disp, szb_disp, lw_disp, lmc_disp, mtr_disp],
    )


if __name__ == "__main__":
    demo.launch()
