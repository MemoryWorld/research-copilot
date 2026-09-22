# 官方资料核验

核验日期：2026-09-23。以下是 API 适配依据，不是实际模型调用证明。版本与区域可用性会变化，运行前以控制台为准。

| 主源 | 本项目采用的契约 |
| --- | --- |
| [Alibaba Cloud Embedding](https://www.alibabacloud.com/help/en/model-studio/embedding) | OpenAI-compatible `/embeddings`；`text-embedding-v4` 支持 1024 维，单批最多 10 条；文档也列出可配置的较新 `qwen3.7-text-embedding` |
| [Function Calling](https://help.aliyun.com/en/model-studio/qwen-function-calling) | `tools`、返回 `tool_calls`、应用执行函数后用 `role=tool` 与 `tool_call_id` 回传；模型选择不等于函数已执行 |
| [Structured output](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen-structured-output) | JSON Object 使用 `response_format`，提示需包含 JSON 字样；格式约束不代替应用层 ID/类型/分数校验 |
| [Thinking mode](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) | 混合思考模型可用请求体 `enable_thinking=false` 关闭思考；本项目显式指定非思考模式，尚未真实调用 |
| [Regions and endpoints](https://help.aliyun.com/en/model-studio/regions/) | Workspace 专属域名与区域绑定；区域密钥不可混用 |
| [Base URL overview](https://help.aliyun.com/en/model-studio/base-url) | 共享 DashScope 域名仍用于旧集成，新配置可使用 `{workspace}.{region}.maas.aliyuncs.com/compatible-mode/v1` |
| [Qwen3 Embedding 官方介绍](https://qwenlm.github.io/blog/qwen3-embedding/) | embedding/retrieval/reranking 的概念背景；本仓库未复现该模型的训练或公开榜单分数 |
| [FastAPI UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/) | 文件上传接口；本应用只处理上传字节，不接受客户端本机路径 |
| [pypdf 文本提取说明](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) | PDF 文本层提取不是 OCR；复杂版式和扫描 PDF 需单独处理 |

默认模型 ID 是便于兼容的配置示例。通过 mocked 请求验证 JSON 字段、维度、分批、工具回传、错误处理，不能推断供应商真实响应、区域权限、费用或效果。可选“Qwen 重排”是 chat-completions 的 listwise 评分，并非声称接入专用 Qwen Rerank 模型 API。
