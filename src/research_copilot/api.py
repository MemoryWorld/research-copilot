from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .config import DISCLAIMER, Settings
from .evaluation import evaluate, SAMPLE
from .providers import ProviderError
from .service import Copilot
from .store import SessionConflict
from .limits import RequestLimitMiddleware


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def create_app(settings=None, copilot=None):
    settings = settings or Settings.from_env()
    service = copilot or Copilot(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Research Copilot", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestLimitMiddleware, max_upload_bytes=settings.max_upload_bytes)
    app.state.copilot = service

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE", "PUT", "PATCH"} and origin:
            if origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "仅允许同源浏览器操作"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Swagger/ReDoc use their own CDN scripts; keep the app UI policy strict
        # without breaking FastAPI's optional interactive API documentation.
        if request.url.path not in {"/docs", "/redoc", "/docs/oauth2-redirect"}:
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(SessionConflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def bad_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": settings.mode}

    @app.get("/api/config")
    def config():
        return {"mode": settings.mode, "model": settings.model if settings.mode == "qwen" else "deterministic-extractive-demo",
                "embedding_model": settings.embedding_model if settings.mode == "qwen" else "feature-hashing-256",
                "reranker": settings.reranker, "limits": {"max_upload_bytes": settings.max_upload_bytes,
                "max_question_chars": settings.max_question_chars}, "disclaimer": DISCLAIMER}

    @app.get("/api/documents")
    def documents():
        return service.store.documents()

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)):
        try:
            data = await file.read(settings.max_upload_bytes + 1)
            return await run_in_threadpool(service.ingest, file.filename or "document", data)
        finally:
            await file.close()

    @app.delete("/api/documents/{identifier}")
    def delete(identifier: str):
        if not service.store.delete_document(identifier):
            raise HTTPException(404, "文档不存在")
        return {"deleted": True}

    @app.get("/api/sample")
    def sample():
        return {"filename": "合成项目手册.md", "content": SAMPLE, "data_source": "synthetic"}

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        return service.chat(request.question, request.session_id)

    @app.get("/api/sessions/{identifier}")
    def session(identifier: str):
        return service.store.session(identifier)

    @app.get("/api/traces/{identifier}")
    def trace(identifier: str):
        result = service.store.trace(identifier)
        if not result:
            raise HTTPException(404, "Trace 不存在")
        return result

    @app.post("/api/evaluations")
    def evaluations():
        return evaluate()

    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
