"""Small deterministic acceptance suite, not an LLM quality benchmark."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import Settings
from .service import Copilot

SAMPLE = """# 合成项目手册

## 北极星索引
北极星索引保存文档切片和来源信息。索引每隔 24 小时更新一次，向量维度为 256。

## 混合检索
混合检索同时使用 BM25 词面召回和向量召回，再使用 RRF 融合。RRF 的平滑常数为 60。

## 资料边界
本手册全部为公开演示编写的合成资料，不是公司客户文档，不证明任何历史部署成果。
"""


def evaluate():
    with TemporaryDirectory(prefix="research-copilot-eval-") as directory:
        app = Copilot(Settings(db_path=str(Path(directory) / "evaluation.db")))
        app.ingest("合成项目手册.md", SAMPLE.encode())
        results = []

        def record(name, condition, detail):
            results.append({"name": name, "passed": bool(condition), "detail": detail})

        first = app.chat("北极星索引保存什么？")
        record("有依据回答与原文引用", not first["refused"] and bool(first["citations"]), "答案必须包含可校验的原文引用")
        second = app.chat("它的更新周期是多少？", first["session_id"])
        record("多轮指代与记忆", "24" in second["answer"] and "北极星" in second["trace"]["retrieval"]["query"], "上一轮问题进入本轮检索；检查已知合成事实")
        unknown = app.chat("ZXQ999 nebula insurance underwriting premium?")
        record("未知知识拒答", unknown["refused"], "知识库之外的问题不生成事实答案")
        arithmetic = app.chat("计算 (12 + 8) / 4")
        record("只读算术工具", "5.0" in arithmetic["answer"], "固定表达式，未执行 Python 代码")
        restored = Copilot(Settings(db_path=str(Path(directory) / "evaluation.db")))
        record("重建后持久化", len(restored.store.session(first["session_id"])["messages"]) == 4,
               "从同一 SQLite 数据库恢复两轮对话")
        record("trace 落库", restored.store.trace(second["trace"]["trace_id"]) is not None,
               "检索分数、工具调用、引用校验和耗时可检查")
        app.close()
        restored.close()
    return {"data_source": "synthetic_offline_acceptance", "results": results,
            "passed": sum(result["passed"] for result in results), "total": len(results),
            "limitations": "仅验证固定合成任务的工程闭环，不代表真实 Qwen 质量、召回率或收益。"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional JSON evidence artifact")
    args = parser.parse_args()
    result = evaluate()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    # The artifact remains human-readable UTF-8. ASCII-escaped console JSON is
    # valid under Windows cp1252 and redirected non-UTF-8 terminal encodings.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
