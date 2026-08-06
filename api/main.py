"""
Financial PoC REST API — Financial Agent（PDF 财报 + 财经新闻）。

启动:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import add_project_root_to_path, ensure_dirs, setup_logging
from api.dependencies import AppState
from api.routes import chat, health, index, query
from api.schemas import ErrorResponse

add_project_root_to_path()
logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    app.state.app_state = AppState()
    logger.info("Financial PoC API 已启动")
    yield
    logger.info("Financial PoC API 已关闭")


app = FastAPI(
    title="Financial PoC API",
    description=(
        "财报/新闻 RAG 检索与对话 REST API。\n\n"
        "主要接口:\n"
        "- `GET /health` 健康检查\n"
        "- `POST /query` 向量/Hybrid 检索\n"
        "- `POST /chat` RAG 连续对话\n"
        "- `POST /index` 构建向量索引"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="服务器内部错误").model_dump(),
    )


app.include_router(health.router)
app.include_router(query.router)
app.include_router(chat.router)
app.include_router(index.router)


@app.get("/", tags=["root"], summary="API 根路径")
def root() -> dict[str, object]:
    return {
        "name": "Financial PoC API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "chat": "POST /chat",
            "index": "POST /index",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
