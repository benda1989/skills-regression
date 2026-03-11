"""POST /api/analyze - 上传文件 + 指令 → 完整分析结果"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ...core.agent import RegressionAgent
from ...core.config import LLMConfig
from ...core.data_loader import DataLoadError, load_dataframe_from_bytes
from ..database import save_record
from ..schemas import AnalysisResult

logger = logging.getLogger("regression_agent.api.analyze")
router = APIRouter()

# 文件大小限制：10MB
_MAX_FILE_SIZE = 10 * 1024 * 1024
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="上传数据文件并执行回归分析",
    description="上传 CSV 或 Excel 文件，可附加自然语言指令，返回完整回归分析结果。",
)
async def analyze(
    file: UploadFile = File(..., description="数据文件（CSV/XLSX/XLS，最大 10MB）"),
    instruction: Optional[str] = Form(None, description="分析指令，例如：'用线性回归分析年龄对收入的影响'"),
    target: Optional[str] = Form(None, description="目标变量列名（可选）"),
    features: Optional[str] = Form(None, description="特征变量列名，英文逗号分隔（可选）"),
    auto_detect: bool = Form(True, description="是否自动识别目标变量和特征"),
) -> AnalysisResult:
    # ── 文件校验 ─────────────────────────────────────────────────────────────

    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式 '{ext}'。仅支持：{', '.join(_ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大（{len(content) // 1024}KB），上传限制为 {_MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    # ── 加载数据 ─────────────────────────────────────────────────────────────

    try:
        df = load_dataframe_from_bytes(content, filename)
    except DataLoadError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to load dataframe from %s", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"文件解析失败：{e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上传的文件不包含有效数据",
        )

    # ── 解析参数 ─────────────────────────────────────────────────────────────

    feat_list = [f.strip() for f in features.split(",") if f.strip()] if features else None

    # ── 执行分析 ─────────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析过程中发生错误：{e}",
        )

    # ── 持久化 ───────────────────────────────────────────────────────────────

    record_id = str(uuid.uuid4())
    model_type = result.get("model_results", {}).get("model_type")
    try:
        save_record(
            record_id=record_id,
            filename=filename,
            instruction=instruction,
            model_type=model_type,
            result=result,
        )
    except Exception:
        logger.warning("Failed to save analysis record to DB", exc_info=True)

    return AnalysisResult(
        id=record_id,
        status="success",
        **result,
    )
