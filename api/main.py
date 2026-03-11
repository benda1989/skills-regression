"""FastAPI 应用入口"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes.analyze import router as analyze_router
from .routes.export import router as export_router
from .routes.history import router as history_router

# ── 日志配置 ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("regression_agent")


# ── 生命周期 ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    logger.info("Regression Agent API started")
    yield
    logger.info("Regression Agent API shutting down")


# ── 应用实例 ─────────────────────────────────────────────────────────────────

_is_production = os.getenv("ENV", "development").lower() == "production"

app = FastAPI(
    title="回归分析智能体 API",
    version="1.0.0",
    description="安全、可测试、可上线的回归分析后端服务",
    lifespan=lifespan,
    # 生产环境关闭文档
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────────────

_allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _allowed_origins_env:
    _allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    # 开发模式：允许本地
    _allowed_origins = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── 路由注册 ─────────────────────────────────────────────────────────────────

app.include_router(analyze_router, prefix="/api", tags=["分析"])
app.include_router(history_router, prefix="/api", tags=["历史记录"])
app.include_router(export_router, prefix="/api", tags=["导出"])


# ── 健康检查 ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health() -> dict:
    return {"status": "ok"}
