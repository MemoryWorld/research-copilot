import json

import httpx
import pytest

from research_copilot.config import Settings
from research_copilot.providers import ProviderError, QwenClient, HashEmbedding
from research_copilot.evaluation import SAMPLE
from research_copilot.service import Copilot


def settings(tmp_path):
    return Settings(mode="qwen", api_key="contract-test-not-a-key", db_path=str(tmp_path / "qwen.db"))


def test_embedding_contract_batches_order_and_dimensions(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/compatible-mode/v1/embeddings"
        assert payload["dimensions"] == 1024
        return httpx.Response(200, json={"data": [{"index": index, "embedding": [1.0] + [0.0] * 1023} for index in reversed(range(len(payload["input"])))]})

    client = QwenClient(settings(tmp_path), httpx.MockTransport(handler))
    vectors = client.embed(["text"] * 12)
    assert [len(item["input"]) for item in requests] == [10, 2]
    assert len(vectors) == 12 and len(vectors[0]) == 1024
    client.close()


@pytest.mark.parametrize("base", ["http://127.0.0.1/v1", "https://dashscope.aliyuncs.com.evil.test/compatible-mode/v1", "https://user:pass@dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1?redirect=x"])
def test_provider_endpoint_allowlist(base):
    with pytest.raises(ValueError):
        Settings(mode="qwen", api_key="test", base_url=base)


def test_workspace_endpoint_allowed():
    Settings(mode="qwen", api_key="test", base_url="https://llm-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")


def test_real_mode_missing_key_is_not_offline():
    with pytest.raises(ValueError, match="no offline fallback"):
        Settings(mode="qwen")


def test_provider_failures_are_sanitized_and_not_offline(tmp_path):
    client = QwenClient(settings(tmp_path), httpx.MockTransport(lambda request: httpx.Response(401, text="sensitive-provider-body")))
    with pytest.raises(ProviderError) as error:
        client.embed(["question"])
    assert "sensitive-provider-body" not in str(error.value)
    assert "contract-test-not-a-key" not in str(error.value)


def test_chat_http_contract_has_tool_fields_and_non_thinking_mode(tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 5}})

    client = QwenClient(settings(tmp_path), httpx.MockTransport(handler))
    message, usage = client.complete([{"role": "user", "content": "Return JSON"}],
                                     tools=[{"type": "function", "function": {"name": "list_documents"}}])
    assert requests[0]["enable_thinking"] is False
    assert requests[0]["tool_choice"] == "auto"
    assert usage["total_tokens"] == 5 and message["content"] == "{}"
    client.complete([{"role": "user", "content": "Return JSON"}], json_mode=True)
    assert requests[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("data", [
    {"data": [{"index": 0, "embedding": [1, 0]}]},
    {"data": [{"index": 1, "embedding": [1] * 1024}]},
    {"data": []},
])
def test_embedding_invalid_contract_is_not_accepted(tmp_path, data):
    client = QwenClient(settings(tmp_path), httpx.MockTransport(lambda request: httpx.Response(200, json=data)))
    with pytest.raises(ProviderError):
        client.embed(["x"])


@pytest.mark.parametrize("malformed", [False, True])
def test_qwen_listwise_reranker_validates_ids_and_scores(tmp_path, malformed):
    from research_copilot.retrieval import Retriever
    app = Copilot(Settings(db_path=str(tmp_path / "ranking.db")))
    app.ingest("manual.md", SAMPLE.encode())

    class RerankModel:
        def complete(self, messages, json_mode=False):
            documents = json.loads(messages[-1]["content"])["documents"]
            scores = [{"id": item["id"], "score": .5} for item in documents]
            if malformed:
                scores[0]["id"] = "forged"
            return {"content": json.dumps({"scores": scores})}, {}

    retriever = Retriever(app.store, app.embedder, RerankModel(), "qwen")
    if malformed:
        with pytest.raises(ProviderError, match="no fallback"):
            retriever.search("北极星索引")
    else:
        assert all(hit["rerank_score"] == .5 for hit in retriever.search("北极星索引"))


class FakeQwen:
    fingerprint = "mock-qwen-contract:v1"

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.calls = []

    def embed(self, texts):
        return HashEmbedding().embed(texts)

    def complete(self, messages, tools=None, json_mode=False):
        self.calls.append(messages.copy())
        if self.behavior == "bad_json":
            return {"content": "not JSON"}, {}
        if self.behavior in {"tool", "forbidden", "loop"} and (len(self.calls) == 1 or self.behavior == "loop"):
            name = "fetch_url" if self.behavior == "forbidden" else "search_knowledge"
            return {"content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps({"query": "北极星索引"})}}]}, {}
        current = next(json.loads(message["content"]) for message in reversed(messages) if message["role"] == "user")
        evidence = current["evidence"][0]
        chunk_id = "invented" if self.behavior == "forged" else evidence["chunk_id"]
        return {"content": json.dumps({"answer": "资料支持的摘录 [1]", "insufficient": False,
                "citations": [{"chunk_id": chunk_id, "quote": evidence["text"][:50]}]}, ensure_ascii=False)}, {"total_tokens": 30}


def test_actual_qwen_adapter_tool_contract_and_citations(tmp_path):
    fake = FakeQwen("tool")
    app = Copilot(settings(tmp_path), fake)
    app.ingest("manual.md", SAMPLE.encode())
    result = app.chat("北极星索引保存什么？")
    assert not result["refused"]
    assert result["trace"]["mode"] == "qwen"
    assert fake.calls[1][-1]["role"] == "tool"
    assert fake.calls[1][-1]["tool_call_id"] == "call-1"
    assert result["trace"]["tools"][-1]["name"] == "search_knowledge"


@pytest.mark.parametrize("behavior,exception", [("bad_json", ProviderError), ("forbidden", ValueError), ("loop", ProviderError)])
def test_malformed_tool_or_budget_failure_is_explicit(tmp_path, behavior, exception):
    app = Copilot(settings(tmp_path), FakeQwen(behavior))
    app.ingest("manual.md", SAMPLE.encode())
    with pytest.raises(exception):
        app.chat("北极星索引保存什么？")


def test_provider_forged_evidence_refuses(tmp_path):
    app = Copilot(settings(tmp_path), FakeQwen("forged"))
    app.ingest("manual.md", SAMPLE.encode())
    result = app.chat("北极星索引保存什么？")
    assert result["refused"] and result["citations"] == []
    assert result["trace"]["citation_validation"].startswith("rejected")
