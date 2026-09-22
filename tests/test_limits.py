import asyncio

from research_copilot.limits import RequestLimitMiddleware


def test_streamed_body_is_stopped_before_full_read_without_content_length():
    events = [{"type": "http.request", "body": b"x" * 1024, "more_body": True} for _ in range(128)]
    events[-1]["more_body"] = False
    consumed = 0
    sent = []

    async def downstream(scope, receive, send):
        raise AssertionError("The multipart parser must never receive this oversized body")

    async def receive():
        nonlocal consumed
        event = events[consumed]
        consumed += 1
        return event

    async def send(event):
        sent.append(event)

    asyncio.run(RequestLimitMiddleware(downstream, 1024)(
        {"type": "http", "method": "POST", "path": "/api/documents", "headers": []}, receive, send))
    assert sent[0]["status"] == 413
    assert consumed < len(events)
    assert consumed == 66  # 1 KiB document + 64 KiB multipart overhead, then stop


def test_declared_oversized_request_reads_no_body():
    sent = []

    async def forbidden(*args):
        raise AssertionError("Request body/downstream must not be consumed")

    async def send(event):
        sent.append(event)

    asyncio.run(RequestLimitMiddleware(forbidden, 1024)(
        {"type": "http", "method": "POST", "path": "/api/documents",
         "headers": [(b"content-length", b"999999")]}, forbidden, send))
    assert sent[0]["status"] == 413
