from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from research_copilot.benchmark import CASES, ranking_metrics, run_benchmark, validate_dataset
from research_copilot.config import Settings
from research_copilot.service import Copilot


def test_document_metrics_deduplicate_chunks_and_use_all_gold():
    metrics = ranking_metrics(["x", "a", "a", "b"], ["a", "b"], k=2)
    assert metrics["recall_at_k"] == .5
    assert metrics["mrr_at_k"] == .5
    assert metrics["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ranking_metrics(["a"], [], 3) is None
    assert ranking_metrics([], ["a"], 3)["recall_at_k"] == 0


def test_label_validation_rejects_unknown_sources():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    data["queries"][0]["relevant"] = ["missing.md"]
    with pytest.raises(ValueError, match="unknown"):
        validate_dataset(data)


def test_real_pipeline_report_preserves_failure_and_quality_denominators():
    report = run_benchmark()
    assert len(report["answers"]) == 12
    assert report["answer_metrics"]["labelled_term_coverage"]["denominator"] == 10
    assert report["answer_metrics"]["unanswerable_refusal"]["denominator"] == 2
    assert len(report["variants"]) == 4
    for value in report["variants"].values():
        assert value["answerable_count"] == 10
        assert len(value["rows"]) == 12
        assert value["errors"] == 0
    assert report["answer_metrics"]["errors"] == 0


def test_failures_are_not_dropped_or_counted_as_refusals():
    def broken(settings):
        app = Copilot(settings)
        def fail(*args, **kwargs):
            raise RuntimeError("secret provider payload must not appear")
        app.retriever.search = fail
        return app
    report = run_benchmark(factory=broken, variants=["hybrid"])
    assert report["variants"]["hybrid"]["mean"]["recall_at_k"] == 0
    assert report["answer_metrics"]["errors"] == 12
    assert report["answer_metrics"]["unanswerable_refusal"]["rate"] == 0
    assert "secret provider payload" not in json.dumps(report)


def test_bm25_does_not_request_query_embedding(tmp_path):
    app = Copilot(replace(Settings(), db_path=str(tmp_path / "corpus.db")))
    app.ingest("a.md", "北极星索引每隔24小时刷新一次。".encode())
    def forbidden(*args, **kwargs):
        raise AssertionError("BM25 ablation must not use dense embeddings")
    app.embedder.embed = forbidden
    assert app.retriever.search("北极星索引", strategy="bm25", rerank=False)


def test_live_benchmark_requires_explicit_opt_in(tmp_path):
    result = subprocess.run([sys.executable, "-m", "research_copilot.benchmark", "--mode", "qwen",
                             "--output", str(tmp_path / "report.json")], capture_output=True,
                            cwd=Path(__file__).parents[1], env={**os.environ, "PYTHONUTF8": "1"}, timeout=20)
    assert result.returncode == 2
    assert b"--allow-provider-calls" in result.stderr
    assert not (tmp_path / "report.json").exists()
