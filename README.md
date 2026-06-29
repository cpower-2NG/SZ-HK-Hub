# SZ-HK Hub · 深港跨境生活 AI 助手

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-0f766e)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

> 一个面向深港双城生活场景的 AI Agent 应用。通过 LangGraph 多智能体协作、高级 RAG 知识库与 MCP 实时工具协议，为跨境出行、金融开户、合规清关等场景提供**可验证、可追踪、可落地**的决策支持。

---

## 目录

- [核心能力](#核心能力)
- [快速开始](#快速开始)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [改进路线图](#改进路线图)
- [开发指南](#开发指南)
- [License](#license)

---

## 核心能力

### 智能跨境规划

用户填写结构化表单（出行目的/目的地/时间/预算）并可选上传海报截图，LangGraph Agent 自动编排全流程：

```
结构化表单 + 📎附件 → route(意图路由) → decompose(拆解子任务)
    → execute(RAG检索+MCP实时工具+Vision日程解析)
    → generate(LLM生成分步计划) → verify(Reflection自审查→自动修正,最多2轮)
    → 输出(规划+日程+冲突提醒+合规校验)
```

### 实时决策支持

| 数据源 | 说明 | 更新频率 |
|--------|------|----------|
| 💱 实时汇率 | HKD ↔ CNY，主备双 API 自动 failover | 实时 |
| 🚆 港铁班次 | 东铁线罗湖/落马洲等 12 站实时到站，data.gov.hk 开放 API | 每 10 秒 |
| 🚌 口岸客流 | 香港入境处每日客流 CSV 解析，三级拥堵分级 | 每日 |

### RAG 知识库

ChromaDB + sentence-transformers 向量检索，覆盖 10 篇语料：

| 类别 | 文档 | 内容 |
|------|------|------|
| 🏦 金融 | `bank-guide.md`、`virtual-bank-guide.md` | 汇丰/渣打/中银/ZA Bank 等 6 家传统银行 + Livi/WeLab/Ant/Mox 4 家虚拟银行 |
| 🛃 海关 | `customs-clearance.md` | 红绿通道、免税额度、货币申报(≥12万HKD)、禁运品 |
| 🚇 交通 | `transport-guide.md`、`mtr-fare.md` | 跨境巴士/港铁/高铁全攻略 + 票价 |
| 🚧 通关 | `border-policy.md` | 8 大口岸开放时间、签注类型、一地两检 |
| 🏫 出行 | `hkust-guide.md`、`hk-tourism.md` | 香港科大 3 条路线 + 20+ 景点 + 半日游推荐 |
| 💰 消费 | `spending-guide.md` | 支付方式对比、餐饮/交通消费参考 |

### 安全护栏

- **规则层**：敏感词过滤（绕过外汇、套现、洗钱、违规开户等）
- **AI 层**：LLM 合规审核 + **Reflection 自审查循环**（规划 → 自我审视 → 自动修正，最多 2 轮）

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- （可选）[Ollama](https://ollama.com) — 无在线 API Key 时的本地 LLM 回退

### 1. 安装依赖

```bash
git clone <repo-url> && cd SZ-HK-Hub
pip install -r requirements.txt
```

### 2. 配置 LLM

```bash
cp .env.example .env
# 编辑 .env，填入 API Key（至少配置 OPENAI_API_KEY）
```

支持的 LLM 提供商（自动优先级探测）：

| 优先级 | 提供商 | 触发条件 |
|--------|--------|----------|
| 1 | DeepSeek / OpenAI 兼容 | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| 2 | Anthropic | `ANTHROPIC_API_KEY` |
| 3 | Ollama 本地 | 以上均未配置 + 本地 Ollama 可达 |

### 3. 构建知识库

```bash
python rag_ingest.py --reset
```

> 首次运行自动从 HuggingFace 镜像下载 embedding 模型（约 90 MB），约 30 秒。

### 4. 启动

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`。

> 无需单独启动 MCP Server。口岸客流、汇率、港铁班次均已内置为直连 API，`MCP_BASE_URL` 仅在需要自定义 MCP 后端时配置。

---

## 技术架构

### 技术栈

| 组件 | 选型 |
|------|------|
| Web UI | Gradio (Soft 主题) |
| AI Agent | LangGraph — StateGraph 多节点编排 |
| LLM | DeepSeek V4 Pro / OpenAI / Anthropic / Ollama 回退 |
| 向量检索 | ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` |
| 实时数据 | data.gov.hk 开放 API + 入境处 CSV + open.er-api.com |
| 多模态 | Vision-LLM（海报/截图日程提取） |
| 工具协议 | MCP (Model Context Protocol) |
| 测试 | pytest + evaluation rubric |

### Agent 工作流

```
                    ┌─────────────┐
                    │  user_query │
                    └──────┬──────┘
                           ▼
               ┌───────────────────────┐
               │  Node 1: route_intent │
               │  意图 + Vision 路由    │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  Node 2: decompose    │
               │  复杂需求 → 子任务拆解  │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  Node 3: execute      │
               │  RAG + MCP + Vision    │
               │  + 日程冲突检测         │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  Node 4: generate     │
               │  LLM 生成分步计划      │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  Node 5: verify       │
               │  Reflection 自审查     │
               │  → 修正(最多2轮)       │
               └───────────┬───────────┘
                           ▼
                    ┌─────────────┐
                    │  统一输出    │
                    │ 规划+日程    │
                    │ +冲突+校验   │
                    └─────────────┘
```

---

## 项目结构

```
SZ-HK-Hub/
├── app.py              # Gradio Web UI 入口（统一表单 + 实时数据仪表盘）
├── config.py           # 环境配置（dataclass）
├── llm_client.py       # LLM 客户端（OpenAI/Anthropic/Ollama 三供应商）
├── planner_agent.py    # LangGraph Agent — 5 节点：路由→拆解→执行→规划→校验(Reflection)
├── mcp_client.py       # 实时数据客户端（汇率双API/口岸客流/港铁实时/Google Maps路线）
├── mcp_server.py       # 自定义 MCP 工具服务（REST 网关，可选启用）
├── rag_store.py        # ChromaDB 向量存储与混合检索
├── rag_ingest.py       # RAG 文档摄入脚本
├── rag_crawler.py      # RAG 动态爬虫（香港政府/银行/港铁官网抓取）
├── vision_client.py    # 多模态 Vision-LLM（日程提取 + 表单填单建议）
├── errors.py           # 自定义异常（ConfigError / ServiceError）
├── requirements.txt    # Python 依赖
├── pytest.ini          # 测试配置
├── .env.example        # 环境变量模板
├── rag_corpus/         # RAG 原始文档（10 篇 .md）
│   ├── bank-guide.md
│   ├── border-policy.md
│   ├── customs-clearance.md
│   ├── hk-tourism.md
│   ├── hkust-guide.md
│   ├── mtr-fare.md
│   ├── spending-guide.md
│   ├── transport-guide.md
│   ├── travel-info.md
│   └── virtual-bank-guide.md
├── rag_db/             # 向量库存储（自动生成）
├── user_data/          # 用户文件存储
└── tests/              # 测试套件（74 用例）+ 评估
    └── evaluation/
        ├── cases_template.jsonl
        ├── manual_judge_template.md
        ├── rag_eval.py        # RAG 自动化评测（Hit Rate + MRR）
        └── rubric_template.md
```

---

## 配置说明

关键环境变量（完整列表见 [`.env.example`](.env.example)）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | LLM API Key（也支持 DeepSeek 等兼容 API） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 端点 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 模型名称 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama 模型名 |
| `MTR_REALTIME_API_URL` | `https://rt.data.gov.hk/v1/transport/mtr/getSchedule.php` | 港铁实时 API |
| `IMMIGRATION_CSV_URL` | `https://www.immd.gov.hk/opendata/...` | 入境处客流 CSV |
| `EXCHANGE_RATE_API_URL` | `https://open.er-api.com/v6/latest/HKD` | 汇率主 API |
| `EXCHANGE_RATE_API_URL_BACKUP` | `https://api.nxvav.cn/api/exchange-rate/` | 汇率备 API |
| `REQUEST_TIMEOUT` | `60` | 请求超时（秒） |

---

## 改进路线图

### ✅ 已完成

- [x] LangGraph 多 Agent 规划流程（5 节点：route → decompose → execute → plan → verify）
- [x] LLM 多供应商回退（DeepSeek / OpenAI / Anthropic / Ollama）
- [x] data.gov.hk 港铁实时到站 API（10 条线，每 10 秒更新）
- [x] 香港入境处每日口岸客流解析（三级拥堵分级）
- [x] 汇率双 API 自动 failover（open.er-api + nxvav.cn）
- [x] RAG 知识库 10 篇深港跨境语料
- [x] Vision-LLM 海报/截图日程提取 + 表单自动填单建议
- [x] 敏感词 + LLM 双重安全护栏
- [x] **统一 Agent Pipeline**：4 Tab → 1 结构化表单 + 附件，AI 自动编排
- [x] **Vision 集成**：附件自动触发 Vision 解析 → 日程注入规划 → 冲突检测
- [x] **Verifier Reflection 循环**：规划 → LLM 自审查 → 自动修正（最多 2 轮）
- [x] **Google Maps 路线规划**：三级 fallback（Google Map API → MCP → 预设）
- [x] **RAG 动态爬虫**：BeautifulSoup 框架，预设 3 个香港政府源
- [x] **数字银行扩充**：Livi Bank / WeLab Bank / Ant Bank / Mox Bank
- [x] **RAG 自动化评测**：Hit Rate 100% / MRR 0.875
- [x] **Multi-Agent 任务拆解**：decompose 节点自动拆解复杂需求
- [x] 74 个 pytest 测试用例全量通过

### 🔮 未来规划

| 任务 | 说明 |
|------|------|
| 多语言支持 | 英文 / 繁体中文界面与输出 |
| 用户历史记忆 | LangGraph Checkpointer 持久化对话上下文 |
| 实时口岸摄像头 | 接入香港运输署 CCTV 快拍（已有 API 端点） |
| 预算自动追踪 | 根据路线 + 消费参考自动计算全程预算 |
| 移动端适配 | PWA 或微信小程序版本 |

---

## 开发指南

### 运行测试

```bash
pytest tests/ -v
```

### RAG 评估

```bash
# 自动化评测（Hit Rate + MRR）
python tests/evaluation/rag_eval.py
python tests/evaluation/rag_eval.py --top-k 5 --json  # JSON 输出
```

评估用例定义在 `tests/evaluation/cases_template.jsonl`（10 条），评分 rubric 在 `tests/evaluation/rubric_template.md`（4 维度：准确性 / 完整性 / 安全性 / 可执行性）。

### RAG 动态更新

```bash
# 爬取预设香港政府源并自动重建向量库
python rag_crawler.py --ingest

# 爬取单个 URL
python rag_crawler.py --url "https://example.com/policy" --output my-policy
```

### Ollama 本地方案

无需任何在线 API Key 即可体验核心功能：

```bash
ollama pull qwen2.5:1.5b   # 986 MB
python app.py               # 自动探测并回退到 Ollama
```

> 注意：小模型对复杂 Planner 提示词响应较慢，建议 `REQUEST_TIMEOUT=120`，或使用在线 API。

---

## License

MIT
