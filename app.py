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
vision_client = VisionClient(config)
planner_agent = PlannerAgent(rag_store, mcp_client, llm_client, vision_client)


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


def handle_planner(
    purpose: str, destination: str, date_time: str, budget: str, extra: str, attachment: bytes | None
) -> tuple[str, str, str]:
    """统一 Pipeline：表单 + 附件 → Planner → 规划(含日程) + 审核结果 + 审核详情。"""

    parts = []
    label_map = {"出行目的": purpose, "目的地": destination, "日期时间": date_time, "预算": budget}
    for label, val in label_map.items():
        if val.strip():
            parts.append(f"{label}：{val.strip()}")
    if extra.strip():
        parts.append(f"补充需求：{extra.strip()}")

    query = "；".join(parts)
    if not query:
        return "<em>请至少填写一项需求。</em>", "⏳ 等待提交", ""

    try:
        result = planner_agent.run(
            user_query=query,
            raw_fields={"purpose": purpose, "dest": destination, "datetime": date_time, "budget": budget},
            attachment=attachment,
        )
    except ConfigError as exc:
        return format_list([f"⚠️ 配置缺失：{exc}"], ""), "⚠️ 需要配置", ""
    except ServiceError as exc:
        return format_list([f"⚠️ 服务异常：{exc}"], ""), "⚠️ 需要人工复核", ""

    # ── 规划 + 日程（合并为一个 HTML 块） ──
    plan_parts: list[str] = []

    # 日程信息（如有）
    if result.schedule_events:
        plan_parts.append("<div style='margin-bottom:0.8rem;padding:0.5rem 0.8rem;background:#fefce8;border:1px solid #fde68a;border-radius:6px;font-size:0.85rem;'>")
        plan_parts.append("<strong>📅 从海报解析的日程：</strong>")
        plan_parts.append("<br>".join(
            f"<span style='margin-left:0.5rem;'>· {e.get('date','?')} {e.get('time','?')} — {e.get('title','?')}</span>"
            for e in result.schedule_events
        ))
        if result.schedule_conflicts:
            plan_parts.append("<br><span style='color:#d97706;'>⚠️ " + " / ".join(result.schedule_conflicts) + "</span>")
        plan_parts.append("</div>")

    # 规划步骤
    plan_steps_html = format_list(result.plan, "暂无规划结果。")
    plan_parts.append(plan_steps_html)

    # ── 审核详情 ──
    review_html = _build_review_html(result.review_detail)

    return "\n".join(plan_parts), f"{result.verification}", review_html


def _build_review_html(detail: dict) -> str:
    """构建可折叠的审核详情 HTML。"""
    if not detail:
        return ""

    parts: list[str] = ["<details open style='margin-top:0.8rem;'>",
                         "<summary style='cursor:pointer;font-weight:600;color:#374151;'>🔍 审核详情</summary>",
                         "<div style='margin-top:0.4rem;font-size:0.85rem;'>"]

    # LLM 审核结论
    status = detail.get("llm_status", "?")
    reason = detail.get("llm_reason", "")
    status_color = {"pass": "#065f46", "review": "#92400e", "block": "#991b1b"}.get(status, "#6b7280")
    parts.append(f"<p>📝 LLM 评审：<span style='color:{status_color};font-weight:600;'>{status}</span> — {reason}</p>")

    # 合规红线
    red = detail.get("red_line_check", "?")
    parts.append(f"<p>🛡️ 合规红线：{'✅ 通过' if red == 'pass' else '⛔ 命中'}</p>")

    # 地点验证
    verified = detail.get("geo_verified", [])
    unverified = detail.get("geo_unverified", [])
    if verified or unverified:
        parts.append("<p style='margin-bottom:2px;'>📍 地点验证（地图）：</p>")
        if verified:
            v_names = ", ".join('<span title="{}" style="color:#065f46;">{}</span>'.format(
                v.get("address",""), v["name"]) for v in verified[:8])
            parts.append(f"<p style='margin:0 0 2px 1rem;'>✅ {len(verified)} 个已验证：{v_names}</p>")
        if unverified:
            u_names = ", ".join(f"<span style='color:#dc2626;font-weight:600;'>{u['name']}</span>" for u in unverified[:8])
            parts.append(f"<p style='margin:0 0 0 1rem;'>❌ {len(unverified)} 个未找到：{u_names}</p>")
            names = [u["name"] for u in unverified]
            parts.append(f"<p style='margin:2px 0 0 1rem;font-size:0.78rem;color:#6b7280;'>这些地名在 地图 中无匹配，可能是 LLM 编造或拼写错误。</p>")
    else:
        parts.append("<p style='color:#6b7280;'>📍 地点验证：未启用（需 AMAP_API_KEY）</p>")

    parts.append("</div></details>")
    return "\n".join(parts)


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

    # ═══════════════════════════════════════════
    # 统一表单区
    # ═══════════════════════════════════════════
    gr.Markdown("## 📝 出行需求")

    with gr.Row():
        purpose = gr.Dropdown(
            choices=["比赛", "旅游", "开户", "购物", "探亲", "商务", "看演出", "其他"],
            label="出行目的", value=None, allow_custom_value=True, scale=1,
        )
        destination = gr.Textbox(
            label="目的地", placeholder="港科大 / 中环 / 尖沙咀…", scale=1,
        )
    with gr.Row():
        date_time = gr.Textbox(
            label="日期 & 时间", placeholder="下周六 14:00-18:00", scale=1,
        )
        budget = gr.Textbox(
            label="预算（可选）", placeholder="HK$500", scale=1,
        )
    extra = gr.Textbox(
        label="补充需求（自由描述）",
        placeholder="例如：顺便去西贡吃海鲜，晚上看维港夜景，需要当天往返…",
        lines=3,
    )
    attachment = gr.File(label="📎 上传海报/截图（可选）", file_count="single", type="binary")

    with gr.Row():
        planner_button = gr.Button("🚀 一键生成规划", variant="primary", size="lg", scale=2)
        verify_status = gr.Textbox(label="📋 审核结果", value="—", interactive=False, scale=1)

    # ═══════════════════════════════════════════
    # 输出区（全宽规划 + 底部审核）
    # ═══════════════════════════════════════════
    gr.Markdown("---")
    gr.Markdown("### 📋 规划方案")
    planner_output = gr.HTML("<em style='color:#6b7280;'>填写需求后点击「一键生成规划」。</em>")
    review_detail_output = gr.HTML("")

    planner_button.click(
        fn=handle_planner,
        inputs=[purpose, destination, date_time, budget, extra, attachment],
        outputs=[planner_output, verify_status, review_detail_output],
    )

    # ═══════════════════════════════════════════
    # 底部：RAG 快速检索 + 审核
    # ═══════════════════════════════════════════
    with gr.Accordion("🔍 高级工具：知识检索 & 审核", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                rag_query = gr.Textbox(
                    label="搜索政策/开户/旅游信息",
                    placeholder="例如：ZA Bank 开户材料 | 免税额度 | 香港科大怎么去",
                )
                rag_button = gr.Button("🔍 检索", variant="secondary", size="sm")
                rag_results = gr.HTML("<em style='color:#6b7280;'>输入关键词检索知识库。</em>")
                rag_button.click(fn=search_rag, inputs=rag_query, outputs=rag_results)
            with gr.Column(scale=1):
                safety_input = gr.Textbox(
                    label="审核",
                    placeholder="输入需检测的内容…",
                )
                safety_output = gr.Textbox(value="🔍 自动标记敏感问题。", interactive=False)
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
