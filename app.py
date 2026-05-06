import random
import re
from pathlib import Path

import gradio as gr

DESCRIPTION = """
SZ-HK Hub 基于多智能体协作、先进 RAG 以及 MCP 协议，为深港双城生活提供可验证、可追踪、可落地的决策支持。
数据为模拟演示，可替换为真实 MCP API。
""".strip()

SENSITIVE_TERMS = ["绕过外汇", "套现", "非法", "洗钱", "违规开户", "避税"]

RAG_DOCS = [
    {
        "title": "ZA Bank 数字银行开户指南",
        "tags": ["开户", "ZA Bank", "材料"],
        "summary": "需准备身份证、港澳通行证及住址证明。支持线上预约视频验证。",
    },
    {
        "title": "港铁西九龙通关时段",
        "tags": ["港铁", "西九龙", "通关"],
        "summary": "高峰时段 08:00-10:00，建议提前 30 分钟到达。",
    },
    {
        "title": "深圳湾口岸人流提示",
        "tags": ["口岸", "深圳湾", "通关"],
        "summary": "周末人流高，建议避开 11:00-13:00。",
    },
    {
        "title": "跨境支付合规提示",
        "tags": ["合规", "支付", "外汇"],
        "summary": "严格遵守外汇管理规定，避免拆分交易。",
    },
]

BASE_METRICS = {
    "rate": 0.92,
    "queue": 28,
    "mtr": 6,
}

PLANNER_TEMPLATES = [
    {
        "keywords": ["开户", "银行", "ZA", "数字银行"],
        "tasks": [
            "确认开户材料清单并预约视频验证",
            "规划通关时间并选择人流较低口岸",
            "准备跨境支付方案与合规提示",
        ],
    },
    {
        "keywords": ["比赛", "活动", "打卡", "展览"],
        "tasks": [
            "解析活动时间与地点",
            "同步港铁时刻并预留通关时间",
            "生成可共享的日程清单",
        ],
    },
]


def format_list(items: list[str], empty_message: str) -> str:
    if not items:
        return empty_message
    return "\n".join(f"- {item}" for item in items)


def has_sensitive(text: str) -> bool:
    return any(term in text for term in SENSITIVE_TERMS)


def generate_plan(text: str) -> list[str]:
    matched = []
    for template in PLANNER_TEMPLATES:
        if any(keyword in text for keyword in template["keywords"]):
            matched.extend(template["tasks"])
    if matched:
        return matched
    return [
        "梳理跨境需求并拆解成可执行子任务",
        "检索最新政策与开户信息",
        "综合实时数据给出路线与支付建议",
    ]


def handle_planner(text: str) -> tuple[str, str]:
    text = text.strip()
    tasks = generate_plan(text)
    status = "需要人工复核" if has_sensitive(text) else "已通过"
    return format_list(tasks, "暂无规划结果。"), status


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
                date = match.group(0)
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


def handle_events(text: str) -> tuple[str, str]:
    events = parse_events(text)
    event_lines = [f"{event['date']} {event['time']} · {event['title']}" for event in events]
    conflicts = detect_conflicts(events)
    conflict_message = (
        f"检测到 {len(conflicts)} 条时间冲突，请调整日程。" if conflicts else "未检测到明显冲突。"
    )
    return format_list(event_lines, "暂无活动解析结果。"), conflict_message


def describe_event_file(file_path: str | None) -> str:
    if not file_path:
        return "未选择活动截图。"
    return f"已选择文件：{Path(file_path).name}（演示版暂不解析图片内容）"


def compute_metrics() -> tuple[float, int, int]:
    rate = round(BASE_METRICS["rate"] + (random.random() - 0.5) * 0.02, 2)
    queue = max(10, round(BASE_METRICS["queue"] + (random.random() - 0.5) * 10))
    mtr = max(3, round(BASE_METRICS["mtr"] + (random.random() - 0.5) * 2))
    return rate, queue, mtr


def refresh_overview() -> tuple[str, str, str]:
    rate, queue, mtr = compute_metrics()
    return (
        f"1 HKD = {rate} CNY",
        f"深圳湾 {queue} 分钟",
        f"西九龙 {mtr} 分钟一班",
    )


def refresh_decision() -> tuple[str, str, str, str]:
    rate, queue, mtr = compute_metrics()
    route = "推荐路线：深圳湾口岸 → 西九龙高铁站" if queue < 25 else "推荐路线：福田口岸 → 港铁东铁线"
    return (
        f"1 HKD = {rate} CNY",
        f"预计 {queue} 分钟",
        f"{mtr} 分钟一班",
        route,
    )


def search_rag(query: str) -> str:
    trimmed = query.strip()
    if not trimmed:
        return "请输入关键词以检索政策或开户信息。"
    results = [
        f"{doc['title']}：{doc['summary']}"
        for doc in RAG_DOCS
        if any(trimmed in tag or tag in trimmed for tag in doc["tags"])
    ]
    return format_list(results, "暂无匹配文档，请尝试其它关键词。")


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
            gr.Markdown("### Planner + Verifier")
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
            gr.Markdown("### 活动/情报解析")
            event_file = gr.File(label="上传活动截图", file_count="single", type="filepath")
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
                inputs=event_input,
                outputs=[event_output, event_conflict],
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 实时决策支持")
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
            gr.Markdown("### RAG 知识检索")
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
