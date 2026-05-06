# SZ-HK Hub
深港跨境专业生活助手（Gradio 本地演示版）

## 本地运行
1. 安装依赖：`pip install -r requirements.txt`
2. 复制环境变量模板：`cp .env.example .env` 并填写 API Key/MCP 地址
3. （可选）补充 RAG 资料后构建向量库：`python rag_ingest.py --reset`
4. 启动应用：`python app.py`
5. 在浏览器访问提示的本地地址（默认 http://127.0.0.1:7860）

## 配置说明
### 必选/推荐的环境变量
- `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`：用于 LangGraph 规划与 Vision 解析
- `MCP_BASE_URL`：MCP Server 地址（用于口岸人流、港铁班次等实时工具）

### 可选环境变量
- `EXCHANGE_RATE_API_URL`：实时汇率 API（默认使用 open.er-api.com）
- `RAG_CORPUS_PATH`：RAG 知识库原始文档目录（默认 `./rag_corpus`）
- `RAG_DB_PATH`：向量库落盘目录（默认 `./rag_db`）
- `EMBEDDING_MODEL`：sentence-transformers 模型名称

## RAG 知识库
`rag_corpus/` 目录当前留空，请将政策、开户指南、通关指引等资料放入该目录（支持 `.md`/`.txt`）。
完成后运行 `python rag_ingest.py --reset` 生成向量库。

## MCP 工具约定
MCP Server 需要提供以下工具（可通过 REST 网关转发）：
- `port_traffic`：输入 `{ "port": "深圳湾" }` 返回 `{ "queue_minutes": 18 }`
- `mtr_schedule`：输入 `{ "station": "西九龙" }` 返回 `{ "interval_minutes": 6 }`
- `exchange_rate`（可选）：输入 `{ "base": "HKD", "target": "CNY" }` 返回 `{ "rate": 0.92 }`

## 功能亮点
- 跨境金融导航：LangGraph 代理调度检索与工具调用
- 活动/情报解析：Vision 解析截图 + 文本解析并检测冲突
- 实时决策支持：MCP 接入汇率、口岸人流、港铁信息
- RAG 知识检索：Vector DB + sentence-transformers 语义检索
- 安全护栏：敏感词检测与合规复核提示
