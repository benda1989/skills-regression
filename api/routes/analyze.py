"""POST /api/analyze        - 上传文件 + 指令 → 完整分析结果（同步）
POST /api/analyze/stream  - 上传文件 + 指令 → SSE 流式分析结果
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from ...core.agent import RegressionAgent, _extract_model_hint  # noqa: F401
from ...core.agent import (
    _extract_features_from_instruction,
    _extract_fit_kwargs,
    _extract_target_from_instruction,
    _resolve_features,
    _resolve_model_type,
    _resolve_target,
)
from ...core.config import LLMConfig
from ...core.data_loader import DataLoadError, load_dataframe_from_bytes
from ..database import save_record
from ..schemas import AnalysisResult

logger = logging.getLogger("regression_agent.api.analyze")
router = APIRouter()

# 文件大小限制：10MB
_MAX_FILE_SIZE = 10 * 1024 * 1024
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ── 公共校验/加载逻辑 ────────────────────────────────────────────────────────

def _validate_and_load(content: bytes, filename: str, features_str: Optional[str]):
    """校验文件并返回 (df, feat_list)，失败抛 HTTPException。"""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式 '{ext}'。仅支持：{', '.join(_ALLOWED_EXTENSIONS)}",
        )
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content) // 1024}KB），上传限制为 {_MAX_FILE_SIZE // 1024 // 1024}MB",
        )
    try:
        df = load_dataframe_from_bytes(content, filename)
    except DataLoadError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to load dataframe from %s", filename)
        raise HTTPException(
            status_code=422,
            detail=f"文件解析失败：{e}",
        )
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上传的文件不包含有效数据",
        )
    feat_list = [f.strip() for f in features_str.split(",") if f.strip()] if features_str else None
    return df, feat_list


# ── SSE 帧工具 ───────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """将 dict 编码为一个 SSE 帧（data: ...\n\n）。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


# ── 流式生成器 ───────────────────────────────────────────────────────────────

async def _stream_analysis(
    content: bytes,
    filename: str,
    instruction: Optional[str],
    target: Optional[str],
    features_str: Optional[str],
    auto_detect: bool,
) -> AsyncGenerator[str, None]:
    """异步生成器：按阶段 yield SSE 帧。

    事件 type 说明：
      stage_start  - 某阶段开始（无 data）
      stage_done   - 某阶段完成，data 字段为该阶段完整结果
      done         - 全流程结束，data 含 id / analysis_context / warnings
      error        - 发生错误，message 字段为错误描述
    """
    # ── 预处理 ───────────────────────────────────────────────────────────────
    yield _sse({"type": "stage_start", "stage": "preprocessing"})
    try:
        df, feat_list = _validate_and_load(content, filename, features_str)
    except HTTPException as e:
        yield _sse({"type": "error", "message": e.detail})
        return

    yield _sse({"type": "stage_done", "stage": "preprocessing", "data": {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }})

    # ── 初始化 Agent ─────────────────────────────────────────────────────────
    try:
        config = LLMConfig.from_env()
        agent = RegressionAgent(config)
    except Exception as e:
        yield _sse({"type": "error", "message": f"LLM 配置错误：{e}"})
        return

    warnings_list: list = []
    understanding: dict = {}

    # ── 指令解析（同步，快速）────────────────────────────────────────────────
    model_hint = _extract_model_hint(instruction)
    target_resolved = target or _extract_target_from_instruction(df, instruction) or None
    features_resolved = feat_list or _extract_features_from_instruction(df, instruction, target_resolved) or None

    # ── Step 1: 数据理解（LLM）──────────────────────────────────────────────
    if auto_detect:
        yield _sse({"type": "stage_start", "stage": "understanding"})
        try:
            understanding = await asyncio.to_thread(
                agent.understand, df, instruction=instruction
            )
        except Exception as e:
            yield _sse({"type": "error", "message": f"数据理解失败：{e}"})
            return
        # LLM 结果优先
        llm_target = understanding.get("target_variable")
        llm_features = understanding.get("features", [])
        if llm_target:
            target_resolved = llm_target
        if llm_features:
            features_resolved = llm_features
        yield _sse({"type": "stage_done", "stage": "understanding", "data": understanding})

    # ── 列名校验 ─────────────────────────────────────────────────────────────
    if not target_resolved or not features_resolved:
        yield _sse({"type": "error",
                    "message": "无法确定目标变量或特征变量，请在 instruction 中指定或手动传入 target/features"})
        return
    try:
        target_resolved, warnings_list = _resolve_target(df, str(target_resolved), warnings_list)
        features_resolved, warnings_list = _resolve_features(
            df, list(features_resolved), target_resolved, warnings_list
        )
    except (KeyError, ValueError) as e:
        yield _sse({"type": "error", "message": str(e)})
        return

    if not features_resolved:
        yield _sse({"type": "error",
                    "message": f"未找到可用的自变量列。可用列: {', '.join(df.columns.astype(str))}"})
        return

    # ── Step 2: 模型选择（LLM）──────────────────────────────────────────────
    yield _sse({"type": "stage_start", "stage": "model_selection"})
    try:
        model_selection = await asyncio.to_thread(
            agent.select_model, df, target_resolved, features_resolved,
            instruction=instruction, model_hint=model_hint,
        )
    except Exception as e:
        yield _sse({"type": "error", "message": f"模型选择失败：{e}"})
        return
    model_type = _resolve_model_type(
        model_selection.get("recommended_model", "linear"),
        model_hint or "",
        instruction or "",
    )
    yield _sse({"type": "stage_done", "stage": "model_selection", "data": {
        **model_selection,
        "resolved_model": model_type,
    }})

    # ── Step 3: 模型拟合（CPU）──────────────────────────────────────────────
    yield _sse({"type": "stage_start", "stage": "fitting"})
    fit_kwargs = _extract_fit_kwargs(model_type, instruction or "")
    try:
        model_results = await asyncio.to_thread(
            agent.fit, df, target_resolved, features_resolved,
            model_type=model_type, **fit_kwargs,
        )
    except Exception as e:
        yield _sse({"type": "error", "message": f"模型拟合失败：{e}"})
        return
    yield _sse({"type": "stage_done", "stage": "fitting", "data": model_results})

    # ── Step 4: 结果解释（LLM）──────────────────────────────────────────────
    yield _sse({"type": "stage_start", "stage": "interpretation"})
    try:
        interpretation = await asyncio.to_thread(
            agent.interpret, model_results, target_resolved, features_resolved,
            instruction=instruction,
        )
    except Exception as e:
        yield _sse({"type": "error", "message": f"结果解释失败：{e}"})
        return
    yield _sse({"type": "stage_done", "stage": "interpretation", "data": interpretation})

    # ── 持久化 ───────────────────────────────────────────────────────────────
    record_id = str(uuid.uuid4())
    full_result = {
        "data_understanding": understanding if auto_detect else {},
        "model_selection": model_selection,
        "model_results": model_results,
        "interpretation": interpretation,
        "analysis_context": {"target": target_resolved, "features": features_resolved},
        "warnings": warnings_list,
    }
    try:
        save_record(
            record_id=record_id,
            filename=filename,
            instruction=instruction,
            model_type=model_results.get("model_type"),
            result=full_result,
        )
    except Exception:
        logger.warning("Failed to save analysis record to DB", exc_info=True)

    # ── 完成 ─────────────────────────────────────────────────────────────────
    yield _sse({"type": "done", "data": {
        "id": record_id,
        "analysis_context": {"target": target_resolved, "features": features_resolved},
        "warnings": warnings_list,
    }})


# ── 端点：非流式（保留兼容）─────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="上传数据文件并执行回归分析（同步）",
    description="上传 CSV 或 Excel 文件，可附加自然语言指令，返回完整回归分析结果。",
)
async def analyze(
    file: UploadFile = File(..., description="数据文件（CSV/XLSX/XLS，最大 10MB）"),
    instruction: Optional[str] = Form(None, description="分析指令，例如：'用线性回归分析年龄对收入的影响'"),
    target: Optional[str] = Form(None, description="目标变量列名（可选）"),
    features: Optional[str] = Form(None, description="特征变量列名，英文逗号分隔（可选）"),
    auto_detect: bool = Form(True, description="是否自动识别目标变量和特征"),
) -> AnalysisResult:
    filename = file.filename or "upload"
    content = await file.read()
    df, feat_list = _validate_and_load(content, filename, features)

    try:
        config = LLMConfig.from_env()
        agent = RegressionAgent(config)
        result = agent.run(
            df=df,
            instruction=instruction,
            target=target or None,
            features=feat_list,
            auto_detect=auto_detect,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析过程中发生错误：{e}",
        )

    record_id = str(uuid.uuid4())
    try:
        save_record(
            record_id=record_id,
            filename=filename,
            instruction=instruction,
            model_type=result.get("model_results", {}).get("model_type"),
            result=result,
        )
    except Exception:
        logger.warning("Failed to save analysis record to DB", exc_info=True)

    return AnalysisResult(id=record_id, status="success", **result)


# ── 端点：流式 SSE ───────────────────────────────────────────────────────────

@router.post(
    "/analyze/stream",
    summary="上传数据文件并流式返回回归分析结果（SSE）",
    description=(
        "与 /api/analyze 相同的参数，但以 Server-Sent Events 格式逐阶段推送结果。\n\n"
        "**事件格式**（每行 `data: <JSON>\\n\\n`）：\n"
        "- `{\"type\":\"stage_start\",\"stage\":\"<阶段名>\"}` — 阶段开始\n"
        "- `{\"type\":\"stage_done\",\"stage\":\"<阶段名>\",\"data\":{...}}` — 阶段完成，含完整数据\n"
        "- `{\"type\":\"done\",\"data\":{\"id\":\"...\",\"analysis_context\":{...},\"warnings\":[...]}}` — 全流程结束\n"
        "- `{\"type\":\"error\",\"message\":\"...\"}` — 错误终止\n\n"
        "**阶段顺序**：`preprocessing` → `understanding`* → `model_selection` → `fitting` → `interpretation`\n"
        "（*`auto_detect=false` 时跳过 `understanding`）"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE 流，每个事件为 `data: <JSON>\\n\\n`",
        }
    },
)
async def analyze_stream(
    file: UploadFile = File(..., description="数据文件（CSV/XLSX/XLS，最大 10MB）"),
    instruction: Optional[str] = Form(None, description="分析指令"),
    target: Optional[str] = Form(None, description="目标变量列名（可选）"),
    features: Optional[str] = Form(None, description="特征变量列名，英文逗号分隔（可选）"),
    auto_detect: bool = Form(True, description="是否自动识别目标变量和特征"),
) -> StreamingResponse:
    filename = file.filename or "upload"
    content = await file.read()

    return StreamingResponse(
        _stream_analysis(
            content=content,
            filename=filename,
            instruction=instruction,
            target=target or None,
            features_str=features,
            auto_detect=auto_detect,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )
