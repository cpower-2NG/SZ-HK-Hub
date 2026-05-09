# SZ-HK Hub · 深港跨境专业生活助手

> 基于 LangGraph + 高级 RAG + MCP 工具链，为深港双城生活提供可验证、可追踪、可落地的决策支持。
> **零 API Key 即可启动**——无任何在线 Key 时自动回退到本地 [Ollama](https://ollama.com)。

---

## 目录

- [快速开始](#快速开始)
- [功能概览](#功能概览)
- [配置说明](#配置说明)
- [Ollama 本地回退](#ollama-本地回退)
- [RAG 知识库](#rag-知识库)
- [MCP 工具约定](#mcp-工具约定)
- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [License](#license)

---

## 快速开始

### 前置条件

- Python ≥ 3.10
- （可选）[Ollama](https://ollama.com) —— 用于无 API Key 时的本地 LLM 回退

### 1. 克隆并安装依赖

```bash
git clone <repo-url> && cd SZ-HK-Hub
pip install -r requirements.txt
```

### 2. （可选）初始化 Ollama 本地模型

```bash
# 安装 Ollama: https://ollama.com/download
ollama pull qwen2.5:1.5b    # 986 MB，默认模型
# 或使用更大模型: ollama pull qwen2.5:7b
```

### 3. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env：不配任何 Key 也能启动（自动 fallback 到本地 Ollama）
```

### 4. 启动应用

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860` 即可体验。

---

## 功能概览

| 模块 | 说明 | 演示模式 |
|------|------|----------|
| **Planner + Verifier** (LangGraph) | 多 Agent 拆解跨境需求 → 规划步骤 → 合规校验 | 使用 Ollama 本地模型 |
| **活动/情报解析** (Vision) | 文本 / 截图解析日程并检测冲突 | 文本解析可用；截图需 API Key |
| **实时决策支持** (MCP) | 汇率、口岸人流、港铁班次 → 推荐路线 | 汇率使用免费 API；口岸/港铁需 MCP |
| **RAG 知识检索** (Vector DB) | 政策/开户指南语义检索 | 需有网环境运行 `rag_ingest.py` |
| **安全护栏检测** | 敏感词过滤 + LLM 合规复核 | 已集成 |

### Planner（需求 → 规划 → 校验）工作流

```
用户输入 → [Route] 路由意图 → [Execute] 检索 RAG + 调用工具
         → [Plan] 生成步骤 → [Verify] 合规校验 → 输出结果
```

---

## 配置说明

所有配置通过环境变量（`.env` 文件）管理，详见 [`.env.example`](.env.example)。

### LLM 提供商优先级

| 优先级 | 提供商 | 触发条件 |
|--------|--------|----------|
| 1 | **OpenAI** | 设置了 `OPENAI_API_KEY` |
| 2 | **Anthropic** | 设置了 `ANTHROPIC_API_KEY`（且无 OpenAI Key） |
| 3 | **Ollama（回退）** | 以上均未设置，但本地 Ollama 服务可达 |

> 无需任何 API Key——安装 Ollama 并拉取一个模型即可全功能体验。

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI API Key |
| `ANTHROPIC_API_KEY` | — | Anthropic API Key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama 模型名称 |
| `MCP_BASE_URL` | — | MCP Server 地址（可选） |
| `EXCHANGE_RATE_API_URL` | `https://open.er-api.com/v6/latest/HKD` | 汇率 API |
| `RAG_CORPUS_PATH` | `./rag_corpus` | 知识库原始文档目录 |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 向量化模型 |

---

## Ollama 本地回退

当未配置 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY` 时，系统自动探测本地 Ollama 服务：

```
LLMClient._resolve_provider()
  ├─ openai?      → 有 OPENAI_API_KEY
  ├─ anthropic?   → 有 ANTHROPIC_API_KEY
  └─ ollama?      → 探测 localhost:11434 是否可达
```

### 支持的模型

| 模型 | 大小 | 推荐场景 |
|------|------|----------|
| `qwen2.5:1.5b` | 986 MB | **默认**，轻量快速 |
| `qwen2.5:7b` | 4.7 GB | 更优规划质量 |
| `deepseek-r1:1.5b` | 1.1 GB | 推理增强 |

切换模型：`export OLLAMA_MODEL=qwen2.5:7b`

---

## RAG 知识库

### 准备资料

将政策文件、开户指南等 `.md`/`.txt` 文件放入 `rag_corpus/`：

```
rag_corpus/
├── za-bank-guide.md
├── mtr-timetable.md
└── cross-border-policy.txt
```

### 构建向量库

```bash
python rag_ingest.py --reset
```

> 首次运行会从 HuggingFace 下载 embedding 模型（需联网）。
> 构建完成后向量库存储在 `rag_db/` 目录。

### 检索示例

在应用界面的 **RAG 知识检索** 面板输入关键词即可查询。

---

## MCP 工具约定

MCP Server 需提供以下工具（可通过 REST 网关转发）：

| 工具 | 输入 | 输出 |
|------|------|------|
| `port_traffic` | `{"port": "深圳湾"}` | `{"queue_minutes": 18}` |
| `mtr_schedule` | `{"station": "西九龙"}` | `{"interval_minutes": 6}` |
| `exchange_rate` | 可选，默认使用 `open.er-api.com` | |

工具名称可通过 `.env` 中的 `MCP_PORT_TOOL`、`MCP_MTR_TOOL`、`MCP_EXCHANGE_TOOL` 自定义。

---

## 项目结构

```
SZ-HK-Hub/
├── app.py              # Gradio 应用入口
├── config.py           # 环境配置（dataclass）
├── llm_client.py       # LLM 客户端（OpenAI / Anthropic / Ollama）
├── planner_agent.py    # LangGraph 多 Agent 规划器
├── mcp_client.py       # MCP 工具调用客户端
├── rag_store.py        # ChromaDB 向量存储与检索
├── rag_ingest.py       # 文档摄入脚本
├── vision_client.py    # 多模态（截图解析）
├── errors.py           # 自定义异常
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量模板
├── README.md           # 本文档
├── rag_corpus/         # RAG 原始文档目录（用户添加）
└── rag_db/             # 向量库目录（自动生成）
```

---

## 技术架构

### 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | Gradio |
| AI Agent | LangGraph (StateGraph) |
| 向量检索 | ChromaDB + sentence-transformers |
| LLM 提供商 | OpenAI / Anthropic / Ollama（回退） |
| 工具协议 | MCP (Model Context Protocol) |
| 多模态 | Vision-LLM（OpenAI / Anthropic） |
| 部署 | 本地 Gradio Server |

### 架构图

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Gradio UI   │────▶│  Planner Agent  │────▶│  Verifier    │
│  (app.py)    │     │  (LangGraph)    │     │  (Reflection)│
└──────────────┘     └────────┬────────┘     └──────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐     ┌──────▼──────┐
              │  RAG      │     │  MCP Tools  │
              │  ChromaDB │     │  汇率/口岸  │
              └───────────┘     └─────────────┘
```

---

## License

MIT
