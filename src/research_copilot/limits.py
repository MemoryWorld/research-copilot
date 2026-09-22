"""Bound requests before multipart parsing can spool uploaded files."""
from starlette.responses import JSONResponse


class RequestLimitMiddleware:
    def __init__(self, app, max_upload_bytes):
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        # Covers multipart metadata too. Each document retains its tighter limit.
        limit = self.max_upload_bytes + 65536 if scope["path"] == "/api/documents" else 65536
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                length = int(raw_length)
            except ValueError:
                length = -1
            if length < 0:
                return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            if length > limit:
                return await self.reject(scope, receive, send)
        buffered = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > limit:
                return await self.reject(scope, receive, send)
            buffered.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(buffered), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)

    @staticmethod
    async def reject(scope, receive, send):
        await JSONResponse({"detail": "请求体超过限制，上传已停止接收"}, status_code=413)(scope, receive, send)
