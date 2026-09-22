# Research Copilot

**根据本人 RAG 工作方向独立实现的公开演示，非公司原始源码或历史部署证明。** 仓库中的示例、离线回答和验收数据均为合成内容，不包含客户资料，不代表过往业务效果。

一个可运行、可检查的 Qwen RAG + 只读 Agent 全栈原型。浏览器上传资料，后端完成解析、切片、向量化、混合召回与重排，再返回答案、原文引用和执行记录。默认无需密钥；真实模式显式调用 Qwen，出错时不会偷偷改用离线回答。

## 本地启动

需要 Python 3.11+，推荐 3.12。安装依赖需要联网，安装后离线模式不需要外部服务。

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn research_copilot.api:create_app --factory --host 127.0.0.1 --port 8787
```

打开 <http://127.0.0.1:8787>，点击“试用一份合成资料”，提问“北极星索引保存什么？”，再问“它的更新周期是多少？”。也可以上传自己的 UTF-8 `.txt`、`.md` 或带文本层的 `.pdf`。

默认 SQLite 文件为 `.data/copilot.db`。应用不自动加载 `.env`，示例文件仅说明可设置的环境变量。界面没有密钥输入框；不要把密钥写进文档或提交 Git。

## 已实现的闭环

| 环节 | 实现与实际边界 |
| --- | --- |
| 文档入库 | 只解析上传字节；Markdown 标题、段落、PDF 页号随切片保留；600 字符窗口、80 字符重叠；拒绝加密/无文本 PDF；不包含 OCR |
| 向量与存储 | 离线 256 维确定性 feature hashing；真实模式 OpenAI-compatible embeddings，默认 1024 维；SQLite 保存原文、向量与 embedding 配置指纹 |
| 检索 | BM25 + dense 各取最多 12 条，经 RRF（k=60）融合为最多 8 条，再取重排后的 4 条；本地线性检索，上限 5,000 切片，不伪称大规模 ANN 服务 |
| 重排 | 默认可解释词面/覆盖度 baseline；可选 Qwen listwise JSON 评分，校验候选 ID 与 0–1 分数；不是专门训练的 cross-encoder |
| 回答 | 离线只做可检查的原文摘录；Qwen 根据证据输出结构化回答；资料不足、伪造 ID/原文引用/编号均拒答 |
| Agent 工具 | `search_knowledge`、`list_documents`、有界算术 `calculator`；JSON schema 校验；最多 3 轮模型请求、6 次工具调用；无 URL 抓取、任意文件读取、shell 或动态执行 |
| 多轮记忆 | SQLite 保存会话、原文引用、trace；最近 3 轮进入上下文，简单指代检索改写；同一会话版本冲突返回 409，避免静默覆盖 |
| 可检查性 | 执行记录包含召回/重排分数、工具参数和结果、引用校验、阶段耗时、供应商返回的 usage；不存隐藏思维过程或密钥 |
| 浏览器 | 上传/删除、合成示例、对话恢复、引用卡片、执行记录、离线验收；纯静态资源，无 CDN 依赖 |

**引用校验是来源完整性检查，不是语义事实核验。** ID、原文片段和引用编号通过，不能证明答案中的每个判断都被原文支持，也不能声称消除了幻觉。离线哈希向量不等价于深度语义 embedding，固定样例通过不等价于真实业务质量。

## Qwen 真实模式

先在当前终端显式设置以下变量（PowerShell 用 `$env:变量名='值'`）：

```bash
export COPILOT_MODE=qwen
export DASHSCOPE_API_KEY='your-own-key'
export QWEN_BASE_URL='https://your-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export QWEN_MODEL='qwen-plus'
export QWEN_EMBEDDING_MODEL='text-embedding-v4'
export COPILOT_DB='.data/qwen.db'
export COPILOT_RERANKER='lexical'
# 可选额外模型调用：export COPILOT_RERANKER='qwen'
python -m uvicorn research_copilot.api:create_app --factory --host 127.0.0.1 --port 8787
```

把 `your-workspace` 替换为控制台给出的 Workspace ID，区域、模型权限和密钥必须匹配。兼容旧集成的北京/新加坡共享 DashScope 域名也在白名单中；新应用建议使用官方 workspace 域名。应用拒绝任意主机、HTTP、重定向和用户提供的 URL；请求禁用环境代理，设置连接/读取超时。

模型 ID 均可配置。2026-09-23 核验的官方文档还列出 `qwen3.7-text-embedding`，Function Calling 示例使用 `qwen3.8-max`。本仓库保留保守默认值；升级前应按对应区域验证模型可用性、非思考模式、JSON 输出、tools 和 embedding 维度支持。Chat 请求显式传 `enable_thinking=false`。所有真实供应商行为当前仅做 **HTTP 契约测试**，没有使用密钥进行真实推理、成本或质量测量。

切换 embedding 模型、维度或域名时，必须使用新数据库重新入库；不能混用原索引。真实模式会向 Qwen 发送问题、有限历史和相关文档内容。真实调用失败返回 502，非法工具请求/输入被拒绝，不会静默降级。

## 可重复验收

```bash
python -m pytest -q
python -m research_copilot.evaluation --output .data/offline-evaluation.json
```

测试禁止外部网络连接；供应商调用通过 `httpx.MockTransport` 或显式假客户端验证。独立验收使用临时库，覆盖文档入库、原文引用、多轮指代、未知问题拒答、只读工具和持久化。输出标记 `synthetic_offline_acceptance`，不污染用户文档库，也不报告虚构的质量提升。

GitHub Actions 在 Windows/Linux 上执行关键静态检查、测试与验收，并上传 JSON 验收产物。Docker 构建/运行另做 Linux smoke test。

### 带标注的检索与问答评测

```bash
python -m research_copilot.benchmark --output .data/quality-benchmark.json
```

这与上面的 6 项工程验收分开：固定 8 份合成资料和 12 个问题（10 个有答案、2 个无答案），分别运行 BM25、dense、RRF 融合、融合加重排四组配置。按文档去重后计算 Recall@3、MRR@3、nDCG@3；回答侧记录标注词覆盖、拒答、引用来源完整性及逐题耗时。多文档问题以全部相关文档为分母；异常保留在分母内，不当作正确拒答。数据版本、SHA-256、embedding 指纹与每题结果一起输出。

当前合成数据规模小、由开发者编写，不是独立盲测或业务效果证明。离线基线实际记录了无答案问题误答和多文档综合回答不足；来源完整性通过不代表回答正确。完整结果与解释见 [评测说明](docs/quality-evaluation.md)。CI 会运行此评测并保留报告，运行错误使 CI 失败；质量分数偏低作为结果展示，不篡改阈值强行通过。

真实 Qwen 评测沿用前述环境变量，并须显式选择供应商调用：

```bash
python -m research_copilot.benchmark --mode qwen --allow-provider-calls --output .data/qwen-quality.json
```

此命令会向已配置的服务发送合成资料和问题，产生 embedding、重排（如果启用）及回答请求；需自行配置可用账户，可能计费。默认离线命令即使环境中存在密钥也不会外联。公开报告目前只包含离线执行，未将接口契约测试写成真实模型测量。

## API 与 curl 演示

Windows 使用 `curl.exe`。先保存一个 UTF-8 Markdown 文件作为演示资料。

```bash
curl http://127.0.0.1:8787/health
curl -X POST http://127.0.0.1:8787/api/documents -F 'file=@examples/synthetic-manual.md'
curl -X POST http://127.0.0.1:8787/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"北极星索引保存什么？"}'
# 复用响应中的 session_id 进行下一轮：
curl -X POST http://127.0.0.1:8787/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"它的更新周期是多少？","session_id":"replace_with_returned_session_id"}'
curl -X POST http://127.0.0.1:8787/api/evaluations
```

`GET /api/documents`、`GET /api/sessions/{id}`、`GET /api/traces/{id}` 可检查持久化状态。OpenAPI 为 `/openapi.json`；可选 Swagger `/docs` 使用 FastAPI 默认 CDN 资源，只有这个辅助文档页需要网络，主界面不需要。

## Docker

```bash
docker compose up --build
```

同样访问 <http://127.0.0.1:8787>。容器使用非 root 用户和独立数据卷，默认离线；不会包含密钥或本地数据库。真实部署自行通过受控环境注入配置。

## 范围、局限与面试材料

这是 **本机单用户原型**，没有账户/租户鉴权、持久任务队列或多节点服务承诺。不要直接以未鉴权端口部署给不可信用户。同源检查不代替认证。ASGI 层在 multipart 解析前限制总请求体（文档限额 + 64 KiB 元数据余量），流式超限立即停止接收并返回 413；应用只缓存限额内请求。PDF 大小/页数/字符限制不是解析沙箱；复杂或恶意 PDF 在生产环境需要独立受限进程和代理层限速/超时。

删除文档会删除检索索引；历史对话、引用和 trace 仍可能保存摘录，不等于隐私数据全量清除。对话和 trace 当前不自动过期。拒答阈值和词面重排权重是可解释初始值，未经业务集标定。SQLite 保存 float JSON 便于检查，未做压缩/ANN/大规模压测。供应商 usage 尚未覆盖 embedding/重排成本，不能据此算总成本。

架构与取舍见 [docs/architecture.md](docs/architecture.md)，实验设计与已有证据边界见 [docs/experiments.md](docs/experiments.md)，官方 API 核验见 [docs/sources.md](docs/sources.md)。
