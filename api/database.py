"""数据库模块：SQLite 存储分析历史记录"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("regression_agent.database")

_db_path: Optional[Path] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id          TEXT PRIMARY KEY,
    filename    TEXT,
    instruction TEXT,
    model_type  TEXT,
    result      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'success',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_history(created_at DESC);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    global _db_path
    if db_path is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "history.db"
    _db_path = db_path
    conn = sqlite3.connect(str(_db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", _db_path)


def _get_conn() -> sqlite3.Connection:
    if _db_path is None:
        init_db()
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def save_record(
    record_id: str,
    filename: str,
    instruction: Optional[str],
    model_type: Optional[str],
    result: Dict[str, Any],
    status: str = "success",
) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO analysis_history (id, filename, instruction, model_type, result, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            filename,
            instruction,
            model_type,
            json.dumps(result, ensure_ascii=False, default=str),
            status,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_records(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, filename, instruction, model_type, status, created_at "
        "FROM analysis_history ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_record(record_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM analysis_history WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["result"] = json.loads(d["result"])
    return d


def delete_record(record_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0
