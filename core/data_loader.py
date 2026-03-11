"""数据加载模块：支持 CSV/XLS/XLSX，返回 DataFrame 供回归分析使用"""

from __future__ import annotations

import logging
import os
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, cast

import pandas as pd

logger = logging.getLogger("regression_agent.data_loader")


class DataLoadError(Exception):
    """数据加载错误"""
    pass


def _normalize_column_name(name: Any) -> str:
    """标准化列名文本，移除多余空白和控制字符。"""
    return (
        str(name)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )


def _deduplicate_columns(columns: Iterable[str]) -> List[str]:
    """确保列名唯一，重复名称追加计数后缀。"""
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for col in columns:
        base = col or "Unnamed"
        count = seen.get(base, 0)
        if count == 0:
            unique.append(base)
        else:
            unique.append(f"{base}_{count}")
        seen[base] = count + 1
    return unique


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    """尝试多种编码读取 CSV，支持多种分隔符"""
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1")
    separators = (",", ";", "\t", "|")

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

    try:
        return pd.read_csv(path, encoding=None, sep=None, engine="python")
    except Exception as e:
        logger.error("无法读取CSV文件 %s: %s", path, e)
        raise DataLoadError(f"无法读取文件: {path}，请检查文件格式和编码。错误详情: {e}") from e


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名和数据"""
    cleaned = df.copy()
    normalized = [_normalize_column_name(col) for col in cleaned.columns]
    cleaned.columns = _deduplicate_columns(normalized)
    cleaned.columns = [
        name if name else f"Unnamed_{idx}" for idx, name in enumerate(cleaned.columns)
    ]
    cleaned = cleaned.dropna(how="all").dropna(axis=1, how="all")
    return cleaned


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    """
    从文件或目录加载数据为 DataFrame。支持 .csv, .xls, .xlsx。
    如果是目录，递归加载所有支持的文件并合并。
    """
    paths: List[str] = []
    if os.path.isdir(path):
        for ext in ("*.csv", "*.xls", "*.xlsx"):
            paths.extend(glob(os.path.join(path, "**", ext), recursive=True))
    else:
        paths = [path]

    dfs: List[pd.DataFrame] = []
    for p in paths:
        lower = p.lower()
        try:
            if lower.endswith(".csv"):
                df = _read_csv_with_fallback(p)
            elif lower.endswith(".xls") or lower.endswith(".xlsx"):
                try:
                    df = pd.read_excel(p, sheet_name=0, engine="openpyxl")
                except Exception:
                    try:
                        df = pd.read_excel(p, sheet_name=0, engine="xlrd")
                    except Exception as e:
                        logger.warning("无法读取Excel文件 %s: %s", p, e)
                        continue
            else:
                continue
            df = _clean_dataframe(df)
            dfs.append(df)
        except Exception as e:
            logger.warning("跳过文件 %s: %s", p, e)

    if not dfs:
        error_msg = (
            f"在路径 {path} 中未找到有效的数据文件。请检查：\n"
            f"  1. 文件路径是否正确\n"
            f"  2. 文件格式是否为 .csv, .xls, 或 .xlsx\n"
            f"  3. 文件是否可读"
        )
        logger.error(error_msg)
        raise DataLoadError(error_msg)

    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


def load_dataframe_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """从字节内容加载 DataFrame（用于 API 文件上传）"""
    import io

    lower = filename.lower()
    if lower.endswith(".csv"):
        # 尝试多种编码
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
            for sep in (",", ";", "\t", "|"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep)
                    if df.shape[1] > 1:
                        return _clean_dataframe(df)
                except Exception:
                    continue
        # 最后尝试自动推断
        df = pd.read_csv(io.BytesIO(content), encoding=None, sep=None, engine="python")
        return _clean_dataframe(df)
    elif lower.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(content), sheet_name=0, engine="openpyxl")
        return _clean_dataframe(df)
    elif lower.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content), sheet_name=0, engine="xlrd")
        return _clean_dataframe(df)
    else:
        raise DataLoadError(f"不支持的文件格式: {filename}。仅支持 .csv, .xls, .xlsx")


def get_data_summary(df: pd.DataFrame, max_rows: int = 5) -> dict:
    """生成数据摘要，用于 LLM 理解数据结构"""
    df_summary = df.copy()

    for col in df_summary.columns:
        series = cast(pd.Series, df_summary[col])
        if series.dtype == "object":
            try:
                numeric_series = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
                if numeric_series.notna().sum() / len(series) > 0.8:
                    df_summary[col] = numeric_series
            except Exception:
                pass

    numeric_cols = df_summary.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_summary.select_dtypes(exclude=["number"]).columns.tolist()

    summary: Dict[str, Any] = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "sample_rows": df.head(max_rows).fillna("").to_dict(orient="records"),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": {
            col: float(df[col].isnull().sum() / len(df) * 100) for col in df.columns
        },
    }

    if numeric_cols:
        numeric_df = df_summary[numeric_cols]
        summary["numeric_summary"] = numeric_df.describe().to_dict()
        numeric_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            series = cast(pd.Series, numeric_df[col])
            numeric_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "unique_count": int(series.nunique()),
            }
        summary["numeric_stats"] = numeric_stats
    else:
        summary["numeric_summary"] = {}
        summary["numeric_stats"] = {}

    if categorical_cols:
        categorical_stats: Dict[str, Dict[str, Any]] = {}
        for col in categorical_cols:
            series = cast(pd.Series, df[col])
            stats: Dict[str, Any] = {"unique_count": int(series.nunique())}
            if series.dtype == "object":
                stats["top_values"] = series.value_counts().head(5).to_dict()
            else:
                stats["top_values"] = {}
            categorical_stats[col] = stats
        summary["categorical_stats"] = categorical_stats
    else:
        summary["categorical_stats"] = {}

    return summary
