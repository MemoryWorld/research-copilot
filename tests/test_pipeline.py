from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from research_copilot.config import Settings
from research_copilot.evaluation import SAMPLE, evaluate
from research_copilot.service import Copilot, checked_citations
from research_copilot.store import SessionConflict
from research_copilot.tools import calculate


def copilot(tmp_path):
    service = Copilot(Settings(db_path=str(tmp_path / "app.db")))
    service.ingest("manual.md", SAMPLE.encode())
    return service


def test_end_to_end_evidence_hybrid_ranking_and_persistent_followup(tmp_path):
    service = copilot(tmp_path)
    first = service.chat("北极星索引保存什么？")
    assert not first["refused"]
    assert first["citations"][0]["quote"] in SAMPLE
    assert first["trace"]["retrieval"]["hits"][0]["bm25_score"] > 0
    assert first["trace"]["retrieval"]["hits"][0]["rrf_score"] > 0
    recreated = Copilot(Settings(db_path=str(tmp_path / "app.db")))
    second = recreated.chat("它的更新周期是多少？", first["session_id"])
    assert "24" in second["answer"]
    assert "北极星" in second["trace"]["retrieval"]["query"]
    assert len(recreated.store.session(first["session_id"])["messages"]) == 4
    assert recreated.store.trace(second["trace"]["trace_id"])["status"] == "completed"


def test_missing_evidence_and_document_deletion_refuse(tmp_path):
    service = copilot(tmp_path)
    assert service.chat("ZXQ999 nebula insurance underwriting premium?")["refused"]
    for document in service.store.documents():
        assert service.store.delete_document(document["id"])
    assert service.store.corpus() == []
    assert service.chat("北极星索引保存什么？")["refused"]


@pytest.mark.parametrize("change", [
    {"citations": [{"chunk_id": "forged", "quote": "correct quote"}]},
    {"citations": [{"chunk_id": "chunk", "quote": "made up quote"}]},
    {"answer": "unsupported marker [9]"}, {"answer": "no source marker"},
    {"citations": []}, {"insufficient": "false"},
])
def test_forged_or_malformed_citations_rejected(change):
    chunk = {"id": "chunk", "text": "correct quote from source", "document_name": "test.md", "page": None, "section": "section", "paragraph": 1}
    payload = {"answer": "claim [1]", "insufficient": False, "citations": [{"chunk_id": "chunk", "quote": "correct quote"}], **change}
    with pytest.raises(ValueError):
        checked_citations(payload, {"chunk": chunk})


@pytest.mark.parametrize("expression", ["__import__('os')", "open('/etc/passwd')", "2 ** 10000", "(lambda:1)()", "[1][0]", "1e300", "1/0"])
def test_calculator_cannot_execute_code_or_unbounded_math(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


def test_readonly_tools_and_invalid_extra_arguments(tmp_path):
    service = copilot(tmp_path)
    assert "5.0" in service.chat("计算 (12 + 8) / 4")["answer"]
    assert "manual.md" in service.chat("列出文档")["answer"]
    with pytest.raises(ValueError):
        service.tools.execute("fetch_url", {"url": "http://127.0.0.1/private"})
    with pytest.raises(ValueError):
        service.tools.execute("search_knowledge", {"query": "x", "path": "/etc/passwd"})


def test_computer_question_does_not_enter_arithmetic_route(tmp_path):
    service = copilot(tmp_path)
    service.ingest("computer.md", "计算机系统由硬件和软件组成。硬件包括处理器和内存。".encode())
    result = service.chat("计算机系统由哪些部分组成？")
    assert not result["refused"]
    assert all(tool["name"] != "calculator" for tool in result["trace"]["tools"])
    assert "硬件" in result["answer"]


def test_embedding_identity_prevents_wrong_index_reuse(tmp_path):
    service = copilot(tmp_path)
    with pytest.raises(ValueError, match="不一致"):
        service.store.bind_embedding("different-model")


def test_concurrent_session_turns_cannot_silently_overwrite(tmp_path):
    service = copilot(tmp_path)
    session = service.store.session()
    barrier = Barrier(2)

    def write(index):
        barrier.wait(timeout=5)
        try:
            service.store.finish_turn(session["session_id"], 0, str(index), "answer", {"trace_id": str(index)})
            return "saved"
        except SessionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(write, [0, 1])) == ["conflict", "saved"]
    assert len(service.store.session(session["session_id"])["messages"]) == 2


def test_offline_acceptance_suite():
    result = evaluate()
    assert result["data_source"] == "synthetic_offline_acceptance"
    assert result["passed"] == result["total"] == 6
