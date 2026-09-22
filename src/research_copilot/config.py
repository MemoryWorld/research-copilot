from dataclasses import dataclass
import os
import re

DISCLAIMER = "根据本人 RAG 工作方向独立实现的公开演示，非公司原始源码或历史部署证明。"
ALLOWED_BASES = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


@dataclass(frozen=True)
class Settings:
    mode: str = "offline"
    db_path: str = ".data/copilot.db"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    reranker: str = "lexical"
    max_upload_bytes: int = 8 * 1024 * 1024
    max_question_chars: int = 2000
    max_document_chars: int = 200_000
    max_pdf_pages: int = 80
    max_chunks: int = 500
    max_corpus_chunks: int = 5000
    max_tool_calls: int = 6
    max_model_rounds: int = 3

    def __post_init__(self):
        if self.mode not in {"offline", "qwen"}:
            raise ValueError("COPILOT_MODE must be offline or qwen")
        if self.reranker not in {"lexical", "qwen"}:
            raise ValueError("COPILOT_RERANKER must be lexical or qwen")
        if self.mode == "offline" and self.reranker != "lexical":
            raise ValueError("Offline mode cannot use a provider reranker")
        if self.mode == "qwen":
            if not self.api_key.strip():
                raise ValueError("Qwen mode requires DASHSCOPE_API_KEY; no offline fallback")
            workspace = re.fullmatch(r"https://[a-z0-9][a-z0-9-]{0,62}\."
                                     r"(?:cn-beijing|cn-hongkong|ap-southeast-1|ap-northeast-1|eu-central-1|us-east-1)"
                                     r"\.maas\.aliyuncs\.com/compatible-mode/v1", self.base_url.rstrip("/"))
            if self.base_url.rstrip("/") not in ALLOWED_BASES and not workspace:
                raise ValueError("QWEN_BASE_URL must be an allowlisted official DashScope endpoint")

    @classmethod
    def from_env(cls):
        return cls(mode=os.getenv("COPILOT_MODE", "offline"),
                   db_path=os.getenv("COPILOT_DB", ".data/copilot.db"),
                   api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                   base_url=os.getenv("QWEN_BASE_URL", cls.base_url).rstrip("/"),
                   model=os.getenv("QWEN_MODEL", cls.model),
                   embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", cls.embedding_model),
                   reranker=os.getenv("COPILOT_RERANKER", "lexical"))
