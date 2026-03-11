"""历史记录端点（需 Bearer token 鉴权）"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import require_auth
from ..database import delete_record, get_record, list_records
from ..schemas import HistoryDetail, HistoryItem

logger = logging.getLogger("regression_agent.api.history")
router = APIRouter(dependencies=[Depends(require_auth)])


@router.get(
    "/history",
    response_model=List[HistoryItem],
    summary="获取分析历史列表",
)
async def get_history(
    limit: int = Query(50, ge=1, le=200, description="返回条数"),
    offset: int = Query(0, ge=0, description="分页偏移"),
) -> List[HistoryItem]:
    records = list_records(limit=limit, offset=offset)
    return [
        HistoryItem(
            id=r["id"],
            filename=r["filename"] or "",
            instruction=r.get("instruction"),
            model_type=r.get("model_type"),
            created_at=r["created_at"],
            status=r.get("status", "success"),
        )
        for r in records
    ]


@router.get(
    "/history/{record_id}",
    response_model=HistoryDetail,
    summary="获取单条历史记录详情",
)
async def get_history_detail(record_id: str) -> HistoryDetail:
    record = get_record(record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")
    return HistoryDetail(
        id=record["id"],
        filename=record["filename"] or "",
        instruction=record.get("instruction"),
        model_type=record.get("model_type"),
        created_at=record["created_at"],
        status=record.get("status", "success"),
        result=record.get("result"),
    )


@router.delete(
    "/history/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除历史记录",
)
async def delete_history(record_id: str) -> None:
    deleted = delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")
