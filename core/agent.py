"""回归分析智能体：编排 understand → select → fit → interpret 流程"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import LLMConfig
from .data_loader import get_data_summary
from .llm import build_llm, call_json
from .models import (
    fit_lasso,
    fit_linear,
    fit_logistic,
    fit_negative_binomial,
    fit_poisson,
    fit_rdd,
    fit_ridge,
)
from .prompts import (
    DATA_UNDERSTANDING_PROMPT,
    MODEL_SELECTION_PROMPT,
    RESULT_INTERPRETATION_PROMPT,
)

logger = logging.getLogger("regression_agent.agent")

# 模型类型注册表
_MODEL_REGISTRY: Dict[str, Any] = {
    "linear": fit_linear,
    "ridge": fit_ridge,
    "lasso": fit_lasso,
    "logistic": fit_logistic,
    "poisson": fit_poisson,
    "negative_binomial": fit_negative_binomial,
    "rdd": fit_rdd,
}

# 中文关键字 → 模型 key 映射
_CN_MODEL_KEYWORDS: Dict[str, str] = {
    "线性回归": "linear",
    "岭回归": "ridge",
    "lasso回归": "lasso",
    "拉索回归": "lasso",
    "逻辑回归": "logistic",
    "泊松回归": "poisson",
    "负二项回归": "negative_binomial",
    "负二项": "negative_binomial",
    "断点回归": "rdd",
}


class RegressionAgent:
    """回归分析智能体：四步流程封装

    Usage::

        agent = RegressionAgent()
        result = agent.run(df, instruction="分析收入对消费的影响")
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._llm = build_llm(self.config)

    # ── 内部 LLM 调用 ────────────────────────────────────────────────────────

    def _call_json(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        return call_json(self._llm, self.config, system_prompt, user_content)

    # ── Step 1: 数据理解 ─────────────────────────────────────────────────────

    def understand(
        self,
        df: pd.DataFrame,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """理解数据集结构，识别目标变量与特征变量"""
        summary = get_data_summary(df)
        user_content = (
            f"Dataset summary (JSON):\n\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        if instruction:
            user_content += f"\n\n用户额外指示：\n{instruction}\n请优先遵循上述需求。"
        return self._call_json(DATA_UNDERSTANDING_PROMPT, user_content)

    # ── Step 2: 模型选择 ─────────────────────────────────────────────────────

    def select_model(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        instruction: Optional[str] = None,
        model_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """使用 LLM 推荐最合适的回归模型"""
        info: Dict[str, Any] = {
            "target": target,
            "features": features,
            "n_samples": len(df),
            "target_type": str(df[target].dtype),
            "target_unique": int(df[target].nunique()),
        }
        if model_hint:
            info["preferred_model"] = model_hint
        user_content = (
            f"Regression setup:\n\n{json.dumps(info, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        if instruction:
            user_content += f"\n\n用户额外指示：\n{instruction}\n请优先遵循上述需求。"
        return self._call_json(MODEL_SELECTION_PROMPT, user_content)

    # ── Step 3: 模型拟合 ─────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        model_type: str = "linear",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """根据 model_type 分发到对应的 fit 函数"""
        key = model_type.lower().strip()
        fit_fn = _MODEL_REGISTRY.get(key)
        if fit_fn is None:
            logger.warning("未知模型类型 '%s'，回退到线性回归", model_type)
            fit_fn = fit_linear
        logger.info("拟合模型: %s (target=%s, features=%s)", key, target, features)
        return fit_fn(df, target, features, **kwargs)

    # ── Step 4: 结果解释 ─────────────────────────────────────────────────────

    def interpret(
        self,
        model_results: Dict[str, Any],
        target: str,
        features: List[str],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """使用 LLM 生成专业的分析报告"""
        user_content = (
            f"Regression results:\n\n"
            f"Target: {target}\n"
            f"Features: {features}\n"
            f"Results (JSON):\n{json.dumps(model_results, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON only."
        )
        if instruction:
            user_content += f"\n\n用户额外指示：\n{instruction}\n请结合上述需求给出解释。"
        return self._call_json(RESULT_INTERPRETATION_PROMPT, user_content)

    # ── 完整流程编排 ─────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        instruction: Optional[str] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        auto_detect: bool = True,
    ) -> Dict[str, Any]:
        """执行完整的回归分析流程

        流程：understand → select_model → fit → interpret

        Args:
            df: 待分析数据集
            instruction: 用户自然语言指令
            target: 目标变量列名（可选，auto_detect=True 时自动识别）
            features: 特征列名列表（可选，auto_detect=True 时自动识别）
            auto_detect: 是否自动识别目标变量和特征

        Returns:
            包含 data_understanding / model_selection / model_results /
            interpretation / analysis_context / warnings 的字典
        """
        warnings_list: List[str] = []
        understanding: Dict[str, Any] = {}

        # 解析指令中的模型提示
        model_hint = _extract_model_hint(instruction)
        target_from_instruction = _extract_target_from_instruction(df, instruction)
        features_from_instruction = _extract_features_from_instruction(df, instruction, target)

        if target_from_instruction and not target:
            target = target_from_instruction
        if features_from_instruction and not features:
            features = features_from_instruction

        # Step 1: 数据理解（LLM 结果优先级高于指令解析）
        if auto_detect:
            understanding = self.understand(df, instruction=instruction)
            llm_target = understanding.get("target_variable")
            llm_features = understanding.get("features", [])
            if llm_target:
                target = llm_target
            if llm_features:
                features = llm_features

        if not target or not features:
            raise ValueError("无法确定目标变量或特征变量，请在 instruction 中指定或手动传入 target/features")

        # 验证列名
        target, warnings_list = _resolve_target(df, str(target), warnings_list)
        features, warnings_list = _resolve_features(df, list(features), target, warnings_list)

        if not features:
            available = ", ".join(df.columns.astype(str))
            raise ValueError(f"未找到可用的自变量列。可用列: {available}")

        # Step 2: 模型选择
        model_selection = self.select_model(
            df, target, features,
            instruction=instruction,
            model_hint=model_hint,
        )

        # 确定最终模型类型
        model_type = _resolve_model_type(
            model_selection.get("recommended_model", "linear"),
            model_hint or "",
            instruction or "",
        )

        # 提取模型参数
        fit_kwargs = _extract_fit_kwargs(model_type, instruction or "")

        # Step 3: 拟合
        model_results = self.fit(df, target, features, model_type=model_type, **fit_kwargs)

        # Step 4: 解释
        interpretation = self.interpret(model_results, target, features, instruction=instruction)

        return {
            "data_understanding": understanding if auto_detect else {},
            "model_selection": model_selection,
            "model_results": model_results,
            "interpretation": interpretation,
            "analysis_context": {"target": target, "features": features},
            "warnings": warnings_list,
        }


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _extract_model_hint(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None
    lower = instruction.lower()
    for cn_key, en_key in _CN_MODEL_KEYWORDS.items():
        if cn_key in lower or en_key in lower:
            return en_key
    return None


def _resolve_model_type(
    llm_recommendation: str,
    model_hint: str,
    instruction: str,
) -> str:
    """从 LLM 推荐 + 用户提示 + 指令中确定最终模型类型"""
    combined = f"{llm_recommendation} {model_hint} {instruction}".lower()

    # 优先级：明确指令 > LLM 推荐
    if "lasso" in combined or "拉索" in combined:
        return "lasso"
    if "ridge" in combined or "岭回归" in combined:
        return "ridge"
    if "logistic" in combined or "逻辑回归" in combined:
        return "logistic"
    if "poisson" in combined or "泊松回归" in combined:
        return "poisson"
    if "negative_binomial" in combined or "负二项" in combined:
        return "negative_binomial"
    if "rdd" in combined or "断点回归" in combined:
        return "rdd"
    if "linear" in combined or "线性回归" in combined:
        return "linear"
    return "linear"


def _extract_fit_kwargs(model_type: str, instruction: str) -> Dict[str, Any]:
    """从指令中提取模型参数（如 alpha）"""
    kwargs: Dict[str, Any] = {}
    if model_type in ("ridge", "lasso"):
        alpha_match = re.search(r"alpha[=：:]\s*([0-9.]+)", instruction, re.IGNORECASE)
        if not alpha_match:
            alpha_match = re.search(r"正则化[系数参数]?[=：:]\s*([0-9.]+)", instruction)
        kwargs["alpha"] = float(alpha_match.group(1)) if alpha_match else 1.0
        kwargs["standardize"] = True
    elif model_type == "linear":
        kwargs["standardize"] = False
    elif model_type in ("logistic", "poisson", "negative_binomial"):
        kwargs["standardize"] = True
    return kwargs


def _extract_target_from_instruction(
    df: pd.DataFrame, instruction: Optional[str]
) -> Optional[str]:
    if not instruction:
        return None
    for col in df.columns:
        col_str = str(col)
        if col_str in instruction:
            return col_str
    return None


def _extract_features_from_instruction(
    df: pd.DataFrame,
    instruction: Optional[str],
    exclude: Optional[str] = None,
) -> List[str]:
    if not instruction:
        return []
    found = [
        str(col) for col in df.columns
        if str(col) in instruction and str(col) != (exclude or "")
    ]
    return found


def _resolve_target(
    df: pd.DataFrame, target: str, warnings_list: List[str]
) -> tuple[str, List[str]]:
    if target in df.columns:
        return target, warnings_list
    # 模糊匹配（忽略大小写）
    for col in df.columns:
        if col.lower() == target.lower():
            warnings_list.append(f'因变量"{target}"大小写匹配到列 {col}')
            return col, warnings_list
    # 回退到第一个数值列
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        fallback = numeric_cols[-1]
        warnings_list.append(f'因变量"{target}"未在数据中找到，已自动选择列 {fallback}')
        return fallback, warnings_list
    available = ", ".join(df.columns.astype(str))
    raise KeyError(f"无法在数据中找到因变量列，请确认名称。可用列: {available}")


def _resolve_features(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    warnings_list: List[str],
) -> tuple[List[str], List[str]]:
    resolved: List[str] = []
    missing: List[str] = []
    for feat in features:
        if feat in df.columns and feat != target:
            resolved.append(feat)
        else:
            # 模糊匹配
            match = next(
                (col for col in df.columns if col.lower() == feat.lower() and col != target),
                None,
            )
            if match:
                resolved.append(match)
            else:
                missing.append(feat)

    if missing:
        warnings_list.append(f"以下自变量在数据集中不存在: {missing}，将尝试自动选择。")

    if not resolved:
        # 自动选择数值列
        auto = [
            col for col in df.select_dtypes(include=["number"]).columns
            if col != target
        ]
        if auto:
            warnings_list.append("已自动选择数值型列作为自变量。")
        resolved = auto

    return resolved, warnings_list
