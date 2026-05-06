const plannerInput = document.getElementById("planner-input");
const plannerOutput = document.getElementById("planner-output");
const verifyStatus = document.getElementById("verify-status");

const eventInput = document.getElementById("event-input");
const eventOutput = document.getElementById("event-output");
const eventConflict = document.getElementById("event-conflict");
const eventFile = document.getElementById("event-file");

const rateValue = document.getElementById("rate-value");
const queueValue = document.getElementById("queue-value");
const mtrValue = document.getElementById("mtr-value");
const routeSuggestion = document.getElementById("route-suggestion");

const heroRate = document.getElementById("hero-rate");
const heroQueue = document.getElementById("hero-queue");
const heroMtr = document.getElementById("hero-mtr");

const ragQuery = document.getElementById("rag-query");
const ragResults = document.getElementById("rag-results");

const safetyInput = document.getElementById("safety-input");
const safetyOutput = document.getElementById("safety-output");

const sensitiveTerms = ["绕过外汇", "套现", "非法", "洗钱", "违规开户", "避税"];

const ragDocs = [
  {
    title: "ZA Bank 数字银行开户指南",
    tags: ["开户", "ZA Bank", "材料"],
    summary: "需准备身份证、港澳通行证及住址证明。支持线上预约视频验证。",
  },
  {
    title: "港铁西九龙通关时段",
    tags: ["港铁", "西九龙", "通关"],
    summary: "高峰时段 08:00-10:00，建议提前 30 分钟到达。",
  },
  {
    title: "深圳湾口岸人流提示",
    tags: ["口岸", "深圳湾", "通关"],
    summary: "周末人流高，建议避开 11:00-13:00。",
  },
  {
    title: "跨境支付合规提示",
    tags: ["合规", "支付", "外汇"],
    summary: "严格遵守外汇管理规定，避免拆分交易。",
  },
];

const baseMetrics = {
  rate: 0.92,
  queue: 28,
  mtr: 6,
};

const plannerTemplates = [
  {
    keywords: ["开户", "银行", "ZA", "数字银行"],
    tasks: [
      "确认开户材料清单并预约视频验证",
      "规划通关时间并选择人流较低口岸",
      "准备跨境支付方案与合规提示",
    ],
  },
  {
    keywords: ["比赛", "活动", "打卡", "展览"],
    tasks: [
      "解析活动时间与地点",
      "同步港铁时刻并预留通关时间",
      "生成可共享的日程清单",
    ],
  },
];

function renderList(target, items) {
  target.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function generatePlan(text) {
  const matchedTasks = plannerTemplates
    .filter((template) => template.keywords.some((key) => text.includes(key)))
    .flatMap((template) => template.tasks);

  if (matchedTasks.length) {
    return matchedTasks;
  }

  return [
    "梳理跨境需求并拆解成可执行子任务",
    "检索最新政策与开户信息",
    "综合实时数据给出路线与支付建议",
  ];
}

function updateVerification(text) {
  const hasSensitive = sensitiveTerms.some((term) => text.includes(term));
  verifyStatus.textContent = hasSensitive ? "需要人工复核" : "已通过";
  verifyStatus.className = `status ${hasSensitive ? "warning" : "ok"}`;
}

function parseEvents(text) {
  const lines = text.split("\n").filter((line) => line.trim());
  const datePatterns = [
    /(\d{4}[/-]\d{1,2}[/-]\d{1,2})/,
    /(\d{1,2}[/-]\d{1,2})/,
    /(\d{1,2})月(\d{1,2})日/,
  ];
  const timePattern = /([01]?\d|2[0-3]):([0-5]\d)/;

  return lines.map((line) => {
    let date = "待确认日期";
    for (const pattern of datePatterns) {
      const match = line.match(pattern);
      if (match) {
        date = match[0];
        break;
      }
    }
    const timeMatch = line.match(timePattern);
    const time = timeMatch ? timeMatch[0] : "待确认时间";
    const title = line.replace(date, "").replace(time, "").trim() || "未命名活动";
    return { date, time, title };
  });
}

function detectConflicts(events) {
  const seen = new Map();
  const conflicts = [];
  events.forEach((event) => {
    const key = `${event.date}-${event.time}`;
    if (seen.has(key)) {
      conflicts.push(key);
    }
    seen.set(key, true);
  });
  return conflicts;
}

function updateMetrics() {
  const rate = (baseMetrics.rate + (Math.random() - 0.5) * 0.02).toFixed(2);
  const queue = Math.max(10, Math.round(baseMetrics.queue + (Math.random() - 0.5) * 10));
  const mtr = Math.max(3, Math.round(baseMetrics.mtr + (Math.random() - 0.5) * 2));

  rateValue.textContent = `1 HKD = ${rate} CNY`;
  queueValue.textContent = `预计 ${queue} 分钟`;
  mtrValue.textContent = `${mtr} 分钟一班`;

  heroRate.textContent = `1 HKD = ${rate} CNY`;
  heroQueue.textContent = `深圳湾 ${queue} 分钟`;
  heroMtr.textContent = `西九龙 ${mtr} 分钟一班`;

  routeSuggestion.textContent =
    queue < 25 ? "推荐路线：深圳湾口岸 → 西九龙高铁站" : "推荐路线：福田口岸 → 港铁东铁线";
}

function searchRag(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    return [];
  }
  return ragDocs.filter((doc) =>
    doc.tags.some((tag) => trimmed.includes(tag) || tag.includes(trimmed))
  );
}

function updateSafety(text) {
  if (!text.trim()) {
    safetyOutput.textContent = "系统将自动标记敏感金融/法律问题。";
    safetyOutput.style.color = "";
    return;
  }

  const hasSensitive = sensitiveTerms.some((term) => text.includes(term));
  if (hasSensitive) {
    safetyOutput.textContent = "检测到敏感问题，请人工复核并遵守合规要求。";
    safetyOutput.style.color = "var(--warning)";
  } else {
    safetyOutput.textContent = "未发现明显敏感词，可继续智能处理。";
    safetyOutput.style.color = "var(--accent)";
  }
}

document.getElementById("planner-run").addEventListener("click", () => {
  const text = plannerInput.value.trim();
  const tasks = generatePlan(text);
  renderList(plannerOutput, tasks);
  updateVerification(text);
});

document.getElementById("event-run").addEventListener("click", () => {
  const events = parseEvents(eventInput.value);
  renderList(
    eventOutput,
    events.map((event) => `${event.date} ${event.time} · ${event.title}`)
  );
  const conflicts = detectConflicts(events);
  eventConflict.textContent = conflicts.length
    ? `检测到 ${conflicts.length} 条时间冲突，请调整日程。`
    : "未检测到明显冲突。";
});

document.getElementById("data-refresh").addEventListener("click", updateMetrics);

document.getElementById("rag-search").addEventListener("click", () => {
  const results = searchRag(ragQuery.value);
  if (!results.length) {
    renderList(ragResults, ["暂无匹配文档，请尝试其它关键词。"]);
    return;
  }
  renderList(
    ragResults,
    results.map((doc) => `${doc.title}：${doc.summary}`)
  );
});

eventFile.addEventListener("change", (event) => {
  if (!event.target.files.length) {
    eventInput.placeholder = "粘贴活动文案，例如：5月18日 10:00 西九龙篮球赛";
    return;
  }
  const name = event.target.files[0].name;
  eventInput.placeholder = `已选择文件：${name}（演示版暂不解析图片内容）`;
});

safetyInput.addEventListener("input", (event) => {
  updateSafety(event.target.value);
});

updateMetrics();
updateSafety("");
