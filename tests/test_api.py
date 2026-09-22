from fastapi.testclient import TestClient

from research_copilot.api import create_app
from research_copilot.config import Settings
from research_copilot.evaluation import SAMPLE


def test_browser_api_full_flow_and_no_secret_config(tmp_path):
    settings = Settings(db_path=str(tmp_path / "app.db"), api_key="not-exposed")
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert "not-exposed" not in client.get("/api/config").text
        upload = client.post("/api/documents", files={"file": ("source.md", SAMPLE.encode(), "text/markdown")})
        assert upload.status_code == 200
        answer = client.post("/api/chat", json={"question": "北极星索引保存什么？"})
        assert answer.status_code == 200 and not answer.json()["refused"]
        response = answer.json()
        assert client.get(f"/api/traces/{response['trace']['trace_id']}").json()["status"] == "completed"
        restored = client.get(f"/api/sessions/{response['session_id']}").json()["messages"]
        assert len(restored) == 2
        assert restored[1]["citations"] == response["citations"]
        assert restored[1]["trace_id"] == response["trace"]["trace_id"]
        assert client.get("/docs").status_code == 200
        assert "content-security-policy" not in client.get("/docs").headers
        assert "script-src 'self'" in client.get("/").headers["content-security-policy"]
        assert client.delete(f"/api/documents/{upload.json()['id']}").json()["deleted"]
        assert client.post("/api/evaluations").json()["passed"] == 6


def test_api_rejects_invalid_inputs_and_cross_origin_mutations(tmp_path):
    with TestClient(create_app(Settings(db_path=str(tmp_path / "app.db")))) as client:
        assert client.post("/api/documents", files={"file": ("x.exe", b"bad")}).status_code == 400
        assert client.post("/api/chat", json={"question": "x", "session_id": "../../etc"}).status_code == 422
        assert client.post("/api/chat", json={"question": "x", "url": "http://localhost"}).status_code == 422
        assert client.post("/api/chat", json={"question": "x"}, headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.get("/health").headers["x-content-type-options"] == "nosniff"
