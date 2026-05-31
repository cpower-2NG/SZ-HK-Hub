# SZ-HK Hub 测试与评测目录

本目录用于把 LLM 应用测试拆成两类：

1. 工程单元测试（pytest，可自动回归）
2. 评测样本模板（用于后续批量评测与评分）

## 目录规划

- `tests/test_planner_workflow.py`：Planner 工作流与合规校验的典型回归测试
- `tests/test_mcp_client.py`：MCP 客户端解析与错误处理测试
- `tests/test_rag_chunking.py`：RAG 分块函数的确定性测试
- `tests/test_llm_client.py`：LLM 供应商选择、JSON 解析与回退逻辑测试
- `tests/test_vision_client.py`：视觉事件解析与媒体类型识别测试
- `tests/test_config_and_ingest.py`：环境配置与 RAG 入库流程测试
- `tests/test_app_features.py`：应用层的日程解析、RAG 检索、路线决策与安全提示测试
- `tests/conftest.py`：共享 fixture 和配置工厂
- `tests/evaluation/cases_template.jsonl`：评测样本模板
- `tests/evaluation/rubric_template.md`：LLM 输出评分规则模板
- `tests/evaluation/manual_judge_template.md`：人工评分页模板

## 功能覆盖映射

- Planner 路由、计划生成、合规复核、降级路径：`test_planner_workflow.py`
- MCP 调用、汇率读取、异常处理：`test_mcp_client.py`
- RAG 文本分块边界：`test_rag_chunking.py`
- LLM 提供商优先级、无配置报错、JSON 包裹输出解析：`test_llm_client.py`
- Vision 图片类型检测、JSON 抽取、provider 分发：`test_vision_client.py`
- App 配置读取、RAG ingest 文件筛选与入库：`test_config_and_ingest.py`
- 应用层日程解析、RAG 输出格式、路线切换、安全提示：`test_app_features.py`

## 推荐测试方法

### 1) 硬约束测试（必须自动化）

目标：保证结构和规则不被破坏。

- JSON 结构是否满足约束
- 步骤数量是否在 3-5 范围
- 敏感请求是否触发人工复核
- 工具异常是否被优雅降级

这部分全部放到 pytest，作为 CI 阻断项。

### 2) 语义评测（模板化）

目标：判断内容质量而不是字面一致。

- 覆盖关键要点
- 可执行性
- 合规性
- 可追溯性（是否引用检索上下文）

这部分先用模板管理样本和 rubric，后续可接 LLM-as-judge 或人工抽检。

### 3) 人工评分页（建议保留）

目标：当自动评分不可靠、输出存在语义歧义或需要合规判断时，由人工完成最终裁定。

- 每个 case 单独填写基本信息、上下文、评分与结论
- 保留问题、优点和是否需要回归修复的记录
- 适合高风险场景、边界样本和上线前抽检

模板见 `tests/evaluation/manual_judge_template.md`。

## 如何运行

```bash
pip install pytest
pytest
```

如需查看详细失败信息：

```bash
pytest -vv
```

## 扩展建议

- 每新增一个核心能力，至少补 2 个成功用例 + 1 个失败用例
- 对敏感场景优先写回归测试，避免“提示词回归”
- 修改 Planner 提示词或工具逻辑后，先跑本目录全量测试
