"""Pydantic 请求/响应模型"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── 请求模型 ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """文件上传 + 指令的元数据（实际文件通过 Form/File 传入）"""
    instruction: Optional[str] = Field(None, description="用户分析指令，如'分析收入对消费的影响'")
    target: Optional[str] = Field(None, description="目标变量列名（可选，不指定则自动识别）")
    features: Optional[List[str]] = Field(None, description="特征变量列名列表（可选）")
    auto_detect: bool = Field(True, description="是否自动识别目标变量和特征")


# ── 响应模型 ─────────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """分析结果响应"""
    id: str = Field(..., description="分析记录唯一ID")
    status: str = Field("success", description="分析状态")
    data_understanding: Dict[str, Any] = Field(default_factory=dict)
    model_selection: Dict[str, Any] = Field(default_factory=dict)
    model_results: Dict[str, Any] = Field(default_factory=dict)
    interpretation: Dict[str, Any] = Field(default_factory=dict)
    analysis_context: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class HistoryItem(BaseModel):
    """历史记录列表项"""
    id: str
    filename: str
    instruction: Optional[str]
    model_type: Optional[str]
    created_at: str
    status: str


class HistoryDetail(HistoryItem):
    """历史记录详情"""
    result: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str
    code: Optional[str] = None
