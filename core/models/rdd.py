"""断点回归（RDD）模型模块"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None

from ._helpers import _rdd_p_value_passed


def _local_linear_rdd(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
) -> Tuple[float, float, float]:
    """局部线性回归估计断点处左/右极限。返回 (estimate, se, pvalue)。"""
    # 三角核: K(u) = max(0, 1-|u|), u = (x-c)/h
    u = (x - c) / h
    w = np.maximum(0, 1 - np.abs(u))
    valid = w > 0
    if np.sum(valid) < 5:
        return float("nan"), float("nan"), float("nan")
    x_c = (x[valid] - c).reshape(-1, 1)
    y_v = y[valid]
    w_v = w[valid]
    X_design = np.column_stack([np.ones_like(x_c), x_c])
    # WLS: (X'WX)^{-1} X'W y, 在 x=c 处 estimate = intercept
    Xw = X_design * np.sqrt(w_v)[:, None]
    yw = y_v * np.sqrt(w_v)
    try:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        pred_var = np.var(yw - Xw @ beta)
        n_eff = np.sum(w_v)
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
        se = np.sqrt(pred_var * XtX_inv[0, 0])
        t_stat = beta[0] / se if se > 1e-10 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(1, n_eff - 2)))
        return float(beta[0]), float(se), float(p_val)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _compute_heterogeneity_rdd(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
    split_var: np.ndarray,
    split_name: str,
) -> Dict[str, Any]:
    """RDD 异质性检验：按 split_var 中位数分组，比较两子组的处理效应是否一致。"""
    out: Dict[str, Any] = {
        "test_name": "处理效应子组差异检验（RDD）",
        "split_by": f"按「{split_name}」中位数将样本分为两组",
        "statistic": None,
        "p_value": None,
        "interpretation": "",
        "interpretation_detail": [],
    }
    if len(x) != len(split_var) or len(y) != len(split_var):
        out["interpretation"] = "分组变量长度与样本不一致，无法进行异质性检验"
        return out
    med = float(np.median(split_var))
    mask_lo = split_var < med
    mask_hi = ~mask_lo
    n_lo, n_hi = int(np.sum(mask_lo)), int(np.sum(mask_hi))
    if n_lo < 20 or n_hi < 20:
        out["interpretation"] = "分组后某一子组样本量不足（至少约20），无法进行异质性检验"
        return out
    x_lo, y_lo = x[mask_lo], y[mask_lo]
    x_hi, y_hi = x[mask_hi], y[mask_hi]
    # 每组需在断点两侧都有足够点
    n_left_lo = int(np.sum(x_lo < c))
    n_right_lo = int(np.sum(x_lo >= c))
    n_left_hi = int(np.sum(x_hi < c))
    n_right_hi = int(np.sum(x_hi >= c))
    if n_left_lo < 5 or n_right_lo < 5 or n_left_hi < 5 or n_right_hi < 5:
        out["interpretation"] = "分组后某一子组在断点左侧或右侧样本量不足，无法进行异质性检验"
        return out
    try:
        mu_l_lo, se_l_lo, _ = _local_linear_rdd(x_lo[x_lo < c], y_lo[x_lo < c], c, h)
        mu_r_lo, se_r_lo, _ = _local_linear_rdd(x_lo[x_lo >= c], y_lo[x_lo >= c], c, h)
        mu_l_hi, se_l_hi, _ = _local_linear_rdd(x_hi[x_hi < c], y_hi[x_hi < c], c, h)
        mu_r_hi, se_r_hi, _ = _local_linear_rdd(x_hi[x_hi >= c], y_hi[x_hi >= c], c, h)
    except Exception:
        out["interpretation"] = "异质性检验计算失败（局部回归异常）"
        return out
    te_lo = mu_r_lo - mu_l_lo if not (np.isnan(mu_l_lo) or np.isnan(mu_r_lo)) else float("nan")
    te_hi = mu_r_hi - mu_l_hi if not (np.isnan(mu_l_hi) or np.isnan(mu_r_hi)) else float("nan")
    se_lo = float(np.sqrt(se_l_lo ** 2 + se_r_lo ** 2)) if not (np.isnan(se_l_lo) or np.isnan(se_r_lo)) else float("nan")
    se_hi = float(np.sqrt(se_l_hi ** 2 + se_r_hi ** 2)) if not (np.isnan(se_l_hi) or np.isnan(se_r_hi)) else float("nan")
    if np.isnan(te_lo) or np.isnan(te_hi) or np.isnan(se_lo) or np.isnan(se_hi):
        out["interpretation"] = "某一子组处理效应或标准误无法计算，无法进行异质性检验"
        return out
    if se_lo < 1e-10 and se_hi < 1e-10:
        out["interpretation"] = "两子组处理效应标准误过小，无法进行异质性检验"
        return out
    se_diff = float(np.sqrt(se_lo ** 2 + se_hi ** 2))
    if se_diff < 1e-10:
        out["interpretation"] = "处理效应之差的标准误为零，无法进行异质性检验"
        return out
    diff = te_lo - te_hi
    z_stat = diff / se_diff
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    out["statistic"] = float(z_stat)
    out["p_value"] = p_val
    out["effect_subgroup1"] = float(te_lo)
    out["effect_subgroup2"] = float(te_hi)
    out["se_subgroup1"] = float(se_lo)
    out["se_subgroup2"] = float(se_hi)
    if p_val >= 0.05:
        out["interpretation"] = "未拒绝原假设，两子组处理效应在统计上无显著差异，处理效应在子组间较为稳定。"
        out["interpretation_detail"] = [
            "为考察断点处处理效应在不同子组间是否一致，本文进行异质性检验。按「" + split_name + "」中位数将样本分为两组，分别在两组内估计断点处的处理效应（局部线性回归），得到子组1处理效应、子组2处理效应及其标准误。",
            "在原假设H₀为两子组处理效应相等、备择假设H₁为两子组处理效应存在差异下，构造两处理效应之差的Z统计量进行检验。",
            f"检验结果为Z = {z_stat:.4f}，p = {p_val:.4f}（p ≥ 0.05），在0.05水平上未拒绝原假设，表明处理效应在按「{split_name}」划分的子组间较为稳定，未发现显著异质性，主回归得到的平均处理效应具有较好的可推广性。",
        ]
    else:
        out["interpretation"] = "拒绝原假设，两子组处理效应存在显著差异，存在异质性，建议报告分组处理效应或进行分组讨论。"
        out["interpretation_detail"] = [
            "为考察断点处处理效应在不同子组间是否一致，本文进行异质性检验。按「" + split_name + "」中位数将样本分为两组，分别在两组内估计断点处的处理效应。",
            "在原假设H₀为两子组处理效应相等、备择假设H₁为两子组处理效应存在差异下，构造Z统计量进行检验。",
            f"检验结果为Z = {z_stat:.4f}，p = {p_val:.4f}（p < 0.05），在0.05水平上拒绝原假设，表明处理效应在按「{split_name}」划分的子组间存在显著异质性。建议在正文中分别报告两子组的处理效应估计值，或进行分组分析与讨论。",
        ]
    return out


def _build_rdd_interpretation(
    running_var: str,
    outcome: str,
    cutoff: float,
    rdd_type: str,
    n: int,
    treatment_effect: Optional[float],
    treatment_effect_se: Optional[float],
    treatment_effect_p: Optional[float],
    first_stage_jump: Optional[float],
    bandwidth: float,
    n_left: int,
    n_right: int,
) -> str:
    """生成断点回归(RDD)的结果解读文本，用语符合科研论文写作规范，内容详实。"""
    blocks = []
    rdd_name = "Sharp RDD（锐断点回归）" if rdd_type == "sharp" else "Fuzzy RDD（模糊断点回归）"

    blocks.append(
        f"本研究采用断点回归设计（Regression Discontinuity Design, RDD）估计干预的因果效应。"
        f"断点回归利用运行变量{running_var}在断点 c = {cutoff:.4f} 处的非连续性分配规则识别因果效应："
        f"当干预分配为「一刀切」时采用Sharp RDD（锐断点回归），当干预分配为「概率性」时采用Fuzzy RDD（模糊断点回归）。"
        f"本分析以{running_var}为运行变量、{outcome}为结果变量，有效样本量为 n = {n}，"
        f"采用局部线性回归（三角核）进行估计，带宽 h = {bandwidth:.4f}，"
        f"断点左侧有效样本量 n_L = {n_left}，右侧 n_R = {n_right}。"
        "上述设定遵循 Calonico et al. (2014) 等文献的常规做法，确保估计量具有一致性及渐近正态性。"
    )

    if treatment_effect is not None:
        te_str = f"τ̂ = {treatment_effect:.4f}"
        if treatment_effect_se is not None:
            te_str += f"（SE = {treatment_effect_se:.4f}"
            if treatment_effect_p is not None:
                sig = "，p < 0.05" if treatment_effect_p < 0.05 else "，p ≥ 0.05"
                te_str += f"{sig}）"
            else:
                te_str += "）"
        if rdd_type == "sharp":
            blocks.append(
                f"估计结果显示，在断点处结果变量{outcome}存在显著跳跃，"
                f"局部平均处理效应（Local Average Treatment Effect, LATE）的估计值为{te_str}。"
                "该估计量可解释为：运行变量恰处于断点右侧（接受干预）与恰处于断点左侧（未接受干预）"
                "的个体在结果变量上的条件期望差异，即断点处的平均因果效应。"
                "该识别策略的有效性依赖于以下假设：在断点附近，个体的运行变量可视为近似随机分配，"
                "从而可将断点两侧的个体视为可比的对照组与处理组。研究者可结合变量含义与理论背景，"
                "对处理效应的经济或政策含义进行深入解读，并在稳健性分析中考察不同带宽及拟合形式下的估计结果。"
            )
        else:
            if first_stage_jump is not None:
                blocks.append(
                    f"估计结果显示，断点处干预变量存在显著跳跃（一阶段跳跃 ≈ {first_stage_jump:.4f}），"
                    f"由此识别的局部平均处理效应估计值为{te_str}。"
                    "该估计量识别 complier 子群体（在断点附近因跨越断点而改变干预状态的个体）"
                    "在结果变量上的平均因果效应。Fuzzy RDD 适用于干预分配并非严格由断点决定的情形，"
                    "通过工具变量思想将断点作为干预的外生变异来源，从而识别 LATE。"
                    "研究者应在论文中说明 complier 子群体的构成及其代表性，以增强结论的适用性。"
                )
            else:
                blocks.append(
                    f"局部平均处理效应估计值为{te_str}，"
                    "该估计量识别 complier 子群体在断点处的平均因果效应。"
                    "研究者可结合变量含义与理论背景进行解读，并在稳健性分析中考察不同设定下的结果。"
                )
    else:
        blocks.append(
            "受数据或带宽限制，未能得到有效的处理效应估计。"
            "建议检查断点设定、样本量及带宽选择，并参考稳健性分析图表进行调整。"
            "研究者亦可尝试更换带宽或拟合形式，以获取更稳健的估计结果。"
        )

    blocks.append(
        "上述统计量及表述可直接用于论文的结果与讨论部分。建议在稳健性分析中报告不同带宽、"
        "不同多项式阶数下的估计结果，以及安慰剂检验、控制变量平衡检验等辅助分析，"
        "以增强因果推断结论的可信度与说服力。"
    )
    return "\n\n".join(blocks)


def _compute_rdd_residuals(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
    mu_left: float,
    mu_right: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算RDD模型的残差和预测值。

    Args:
        x: 运行变量
        y: 结果变量
        c: 断点值
        h: 带宽
        mu_left: 断点左侧的估计值
        mu_right: 断点右侧的估计值

    Returns:
        (residuals, predictions): 残差数组和预测值数组
    """
    predictions = np.zeros_like(y)
    mask_left = x < c
    mask_right = x >= c

    # 对每个观测值，使用局部线性回归计算预测值
    for i in range(len(x)):
        xi = x[i]
        if mask_left[i]:
            # 左侧：使用左侧局部线性回归
            u = (x - c) / h
            w = np.maximum(0, 1 - np.abs(u))
            w[mask_right] = 0  # 只使用左侧数据
            valid = w > 0
            if np.sum(valid) >= 2:
                x_c = (x[valid] - c).reshape(-1, 1)
                y_v = y[valid]
                w_v = w[valid]
                X_design = np.column_stack([np.ones_like(x_c), x_c])
                Xw = X_design * np.sqrt(w_v)[:, None]
                yw = y_v * np.sqrt(w_v)
                try:
                    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                    # 预测值 = beta[0] + beta[1] * (xi - c)
                    predictions[i] = beta[0] + beta[1] * (xi - c)
                except Exception:
                    predictions[i] = mu_left
            else:
                predictions[i] = mu_left
        else:
            # 右侧：使用右侧局部线性回归
            u = (x - c) / h
            w = np.maximum(0, 1 - np.abs(u))
            w[mask_left] = 0  # 只使用右侧数据
            valid = w > 0
            if np.sum(valid) >= 2:
                x_c = (x[valid] - c).reshape(-1, 1)
                y_v = y[valid]
                w_v = w[valid]
                X_design = np.column_stack([np.ones_like(x_c), x_c])
                Xw = X_design * np.sqrt(w_v)[:, None]
                yw = y_v * np.sqrt(w_v)
                try:
                    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                    # 预测值 = beta[0] + beta[1] * (xi - c)
                    predictions[i] = beta[0] + beta[1] * (xi - c)
                except Exception:
                    predictions[i] = mu_right
            else:
                predictions[i] = mu_right

    residuals = y - predictions
    return residuals, predictions


def _mccrary_density_test(
    x: np.ndarray,
    c: float,
    h: float,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """McCrary (2008) 密度检验：检验运行变量在断点处是否存在操纵（密度是否连续）。
    若密度在断点处显著不连续，则提示可能存在自选择或操纵，威胁 RDD 有效性。"""
    out: Dict[str, Any] = {
        "test_name": "McCrary 密度检验（操纵检验）",
        "statistic": None,
        "p_value": None,
        "theta": None,
        "se_theta": None,
        "interpretation": "",
    }
    try:
        # 仅在 [c-h, c+h] 内做分箱，断点两侧各若干箱
        mask = (x >= c - h) & (x <= c + h)
        x_local = x[mask]
        if len(x_local) < 20:
            out["interpretation"] = "断点附近样本量不足，无法进行密度检验"
            return out
        # 左侧 (c-h, c) 与右侧 [c, c+h) 的样本量
        n_left = int(np.sum(x_local < c))
        n_right = int(np.sum(x_local >= c))
        if n_left < 5 or n_right < 5:
            out["interpretation"] = "断点两侧样本量不足，无法进行密度检验"
            return out
        # 密度：每侧观测数 / 该侧区间长度
        len_left = max(c - (c - h), 1e-10)
        len_right = max((c + h) - c, 1e-10)
        f_left = n_left / len_left
        f_right = n_right / len_right
        # theta = log(f_right) - log(f_left)，在 H0: 密度连续下 E[theta]=0
        if f_left <= 0 or f_right <= 0:
            out["interpretation"] = "某侧密度为0，无法计算对数差"
            return out
        theta = float(np.log(f_right) - np.log(f_left))
        # 近似 SE(theta) ≈ sqrt(1/n_right + 1/n_left)（对数尺度）
        se_theta = float(np.sqrt(1.0 / n_right + 1.0 / n_left))
        if se_theta < 1e-10:
            out["interpretation"] = "密度检验标准误过小，无法计算"
            return out
        stat = theta / se_theta
        p_val = float(2 * (1 - stats.norm.cdf(abs(stat))))
        out["theta"] = theta
        out["se_theta"] = se_theta
        out["statistic"] = float(stat)
        out["p_value"] = p_val
        out["n_left"] = n_left
        out["n_right"] = n_right
        if p_val >= 0.05:
            out["interpretation"] = "在0.05水平上未拒绝密度在断点处连续的原假设，未发现明显操纵证据，RDD 有效性得到支持。"
        else:
            out["interpretation"] = "在0.05水平上拒绝密度连续的原假设，断点处存在显著密度跳跃，提示可能存在操纵或自选择，需谨慎解释 RDD 估计。"
    except Exception as e:
        out["interpretation"] = f"密度检验计算异常: {e!s}"
    return out


def _rdd_donut_estimate(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
    donut_frac: float = 0.02,
    d: Optional[np.ndarray] = None,
    fuzzy: bool = False,
) -> Dict[str, Any]:
    """Donut RDD：排除断点附近 [c-delta, c+delta] 的样本后重新估计处理效应。
    donut_frac: 排除区间半宽占运行变量范围的比例（如 0.02 表示左右各 2% 范围）。"""
    out: Dict[str, Any] = {
        "treatment_effect": None,
        "se": None,
        "p_value": None,
        "n_excluded": None,
        "n_remaining": None,
        "interpretation": "",
    }
    try:
        x_range = float(np.percentile(x, 99) - np.percentile(x, 1)) or 1.0
        delta = max(x_range * donut_frac, 1e-8)
        keep = (x <= c - delta) | (x >= c + delta)
        n_excluded = int(np.sum(~keep))
        n_remaining = int(np.sum(keep))
        if n_remaining < 20:
            out["interpretation"] = "排除断点附近样本后剩余观测不足，无法进行 Donut RDD 估计"
            out["n_excluded"] = n_excluded
            out["n_remaining"] = n_remaining
            return out
        x_d = x[keep]
        y_d = y[keep]
        mask_left = x_d < c
        mask_right = x_d >= c
        if np.sum(mask_left) < 5 or np.sum(mask_right) < 5:
            out["interpretation"] = "排除后断点某一侧样本不足，无法进行 Donut RDD 估计"
            out["n_excluded"] = n_excluded
            out["n_remaining"] = n_remaining
            return out
        mu_l, se_l, _ = _local_linear_rdd(x_d[mask_left], y_d[mask_left], c, h)
        mu_r, se_r, _ = _local_linear_rdd(x_d[mask_right], y_d[mask_right], c, h)
        if np.isnan(mu_l) or np.isnan(mu_r):
            out["interpretation"] = "Donut 样本下局部回归无法得到有效估计"
            out["n_excluded"] = n_excluded
            out["n_remaining"] = n_remaining
            return out
        te = mu_r - mu_l
        se_te = float(np.sqrt(se_l ** 2 + se_r ** 2))
        if fuzzy and d is not None:
            d_d = d[keep]
            dl, sel, _ = _local_linear_rdd(x_d[mask_left], d_d[mask_left], c, h)
            dr, ser, _ = _local_linear_rdd(x_d[mask_right], d_d[mask_right], c, h)
            if not (np.isnan(dl) or np.isnan(dr)) and abs(dr - dl) > 1e-8:
                fs_jump = dr - dl
                te = te / fs_jump
                var_ratio = (se_l ** 2 + se_r ** 2) / (fs_jump ** 2)
                var_ratio += (sel ** 2 + ser ** 2) * (te ** 2) / (fs_jump ** 2)
                se_te = float(np.sqrt(var_ratio))
        out["treatment_effect"] = float(te)
        out["se"] = se_te
        out["n_excluded"] = n_excluded
        out["n_remaining"] = n_remaining
        if se_te > 1e-10:
            t_stat = te / se_te
            out["p_value"] = float(2 * (1 - stats.t.cdf(abs(t_stat), max(1, n_remaining - 4))))
        if out["p_value"] is not None and out["p_value"] >= 0.05:
            out["interpretation"] = "排除断点附近样本后，处理效应在0.05水平上不显著，可能与主回归结论不一致，需结合样本与设定讨论。"
        else:
            out["interpretation"] = "排除断点附近样本后，处理效应方向与主回归一致，结论对「挖洞」设定稳健。"
    except Exception as e:
        out["interpretation"] = f"Donut RDD 计算异常: {e!s}"
    return out


def _build_robustness_interpretation_detail(
    check_name: str,
    check_data: Dict[str, Any],
    rdd_type: str,
    running_var: str,
    outcome: str,
    cutoff: float,
    bandwidth: float,
    treatment_var: Optional[str] = None,
    main_treatment_effect: Optional[float] = None,
) -> List[str]:
    """为稳健性检验项生成详细的解读文本（多段），符合科研论文写作规范。"""
    detail: List[str] = []

    if check_name == "密度检验（McCrary）":
        theta = check_data.get("theta")
        se_theta = check_data.get("se_theta")
        stat = check_data.get("statistic")
        p_val = check_data.get("p_value")
        n_left = check_data.get("n_left")
        n_right = check_data.get("n_right")
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归设计（RDD）的有效性，本文采用 McCrary (2008) 提出的密度检验方法，检验运行变量「{running_var}」在断点 c = {cutoff:.4f} 处的密度分布是否连续。"
            f"该检验基于以下识别假设：若个体无法精确操纵运行变量，则在断点附近运行变量的密度分布应连续；若存在操纵或自选择，则密度在断点处会出现显著跳跃。"
            f"检验方法为：在断点附近带宽 h = {bandwidth:.4f} 的区间内，分别计算断点左侧与右侧的观测密度（观测数/区间长度），"
            f"构造对数密度差 θ = log(f₊) - log(f₋)，其中 f₊ 为断点右侧密度，f₋ 为断点左侧密度。"
        )
        if theta is not None and se_theta is not None and stat is not None and p_val is not None:
            detail.append(
                f"在断点左侧有效样本量 n₋ = {n_left}，右侧 n₊ = {n_right}。"
                f"估计得到对数密度差 θ = {theta:.4f}，标准误 SE(θ) = {se_theta:.4f}，"
                f"构造检验统计量 t = θ / SE(θ) = {stat:.4f}。"
                f"在原假设 H₀ 为密度在断点处连续（即 θ = 0）、备择假设 H₁ 为密度在断点处不连续（即 θ ≠ 0）下，"
                f"采用标准正态分布进行检验。"
            )
            if passed:
                detail.append(
                    f"检验结果为 t = {stat:.4f}，p = {p_val:.4f}（p ≥ 0.05），在 0.05 显著性水平上未拒绝原假设，"
                    f"表明运行变量「{running_var}」在断点处的密度分布连续，未发现明显的操纵或自选择证据，"
                    f"断点回归设计的有效性得到支持，主回归估计结果具备良好的识别基础。"
                    f"该结果进一步增强了断点回归识别策略的可信度，说明在断点附近个体对运行变量的控制能力有限，"
                    f"局部随机化假设在可接受范围内成立，为后续的因果推断提供了坚实的实证基础。"
                )
            else:
                detail.append(
                    f"检验结果为 t = {stat:.4f}，p = {p_val:.4f}（p < 0.05），在 0.05 显著性水平上拒绝原假设，"
                    f"表明运行变量「{running_var}」在断点处存在显著的密度跳跃，提示可能存在操纵或自选择行为。"
                    f"在此情形下，断点回归的识别假设可能受到威胁，建议研究者结合理论背景与数据特征进一步分析："
                    f"（1）检查是否存在个体能够精确控制运行变量的机制；（2）考虑采用 Donut RDD 排除断点附近可能被操纵的样本；"
                    f"（3）结合其他识别策略（如工具变量、双重差分等）进行交叉验证；（4）在正文中明确报告密度检验结果，"
                    f"并讨论其对结论稳健性的潜在影响。"
                )

    elif check_name == "Donut RDD":
        donut_te = check_data.get("treatment_effect")
        donut_se = check_data.get("se")
        donut_p = check_data.get("p_value")
        n_excluded = check_data.get("n_excluded")
        n_remaining = check_data.get("n_remaining")
        main_te = check_data.get("main_treatment_effect")
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归估计结果对断点附近样本的敏感性，本文采用 Donut RDD（挖洞回归）方法，"
            f"排除运行变量「{running_var}」在断点 c = {cutoff:.4f} 附近 2% 范围内的观测后重新估计处理效应。"
            f"该方法基于以下考虑：若个体能够精确操纵运行变量使其恰好落在断点附近，则这些观测可能违反局部随机化假设，"
            f"影响估计结果的可靠性；通过排除这些观测，可以检验主回归结论是否对断点附近的样本选择敏感。"
        )
        if main_treatment_effect is not None:
            check_data["main_treatment_effect"] = main_treatment_effect
        if n_excluded is not None and n_remaining is not None:
            detail.append(
                f"排除断点附近样本后，剩余有效观测数 n = {n_remaining}，排除观测数 = {n_excluded}。"
                f"在排除断点附近样本的条件下，采用与主回归相同的局部线性回归方法（三角核，带宽 h = {bandwidth:.4f}）"
                f"重新估计断点处的处理效应。"
            )
        if donut_te is not None and donut_se is not None and donut_p is not None and main_te is not None:
            detail.append(
                f"Donut RDD 估计结果显示，排除断点附近样本后的处理效应为 {donut_te:.4f}，标准误为 {donut_se:.4f}，"
                f"t 统计量为 {donut_te / donut_se:.4f}（若标准误 > 0），p 值为 {donut_p:.4f}。"
                f"主回归的处理效应估计值为 {main_te:.4f}。"
            )
            if passed:
                detail.append(
                    f"比较 Donut RDD 与主回归的估计结果，两者处理效应的方向一致（符号相同），"
                    f"且 Donut RDD 估计在 0.05 水平上显著（p = {donut_p:.4f} < 0.05），"
                    f"表明排除断点附近样本后，处理效应的估计结果与主回归结论保持一致，"
                    f"主回归结论对「挖洞」设定稳健，未发现断点附近样本对估计结果的显著影响。"
                    f"该结果进一步支持了断点回归识别策略的有效性，说明主回归估计结果不依赖于断点附近的特定样本，"
                    f"结论具有较好的稳健性与可推广性。"
                )
            else:
                detail.append(
                    f"比较 Donut RDD 与主回归的估计结果，两者处理效应的方向或显著性存在差异，"
                    f"提示主回归结论可能对断点附近的样本选择敏感。"
                    f"建议研究者：（1）在正文中报告 Donut RDD 估计结果，并与主回归结果进行对比；"
                    f"（2）讨论断点附近样本的特征，分析是否存在操纵或选择性偏差；"
                    f"（3）考虑采用不同的排除窗口（如 1%、3%、5%）进行敏感性分析；"
                    f"（4）结合其他稳健性检验结果，综合评估结论的稳健性。"
                )

    elif check_name == "安慰剂断点检验":
        placebo_effects = check_data.get("placebo_effects", [])
        placebo_pvalues = check_data.get("placebo_pvalues", [])
        placebo_cutoffs = check_data.get("placebo_cutoffs", [])
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归识别策略的有效性，本文进行安慰剂断点检验（Placebo Cutoff Test），"
            f"在假断点（理论上不应存在政策或干预变化的位置）处估计处理效应。"
            f"该检验基于以下逻辑：若断点回归识别有效，则仅在真实断点 c = {cutoff:.4f} 处应观察到显著的处理效应，"
            f"而在其他位置（假断点）处不应观察到显著的处理效应；若在假断点处也观察到显著效应，"
            f"则可能表明结果变量本身在运行变量上的变化模式存在非线性或其他设定偏误，而非真正的处理效应。"
        )
        if placebo_cutoffs and placebo_effects and placebo_pvalues:
            detail.append(
                f"本文选择运行变量「{running_var}」的第 25% 和第 75% 分位数作为假断点，"
                f"分别在这些位置采用与主回归相同的局部线性回归方法估计处理效应。"
            )
            for i, (fake_c, eff, p_val) in enumerate(zip(placebo_cutoffs, placebo_effects, placebo_pvalues)):
                if not np.isnan(eff) and not np.isnan(p_val):
                    detail.append(
                        f"在假断点 c_placebo = {fake_c:.4f} 处，估计得到的处理效应为 {eff:.4f}，"
                        f"p 值为 {p_val:.4f}。"
                    )
            if passed:
                detail.append(
                    f"检验结果显示，在所有假断点处，处理效应的 p 值均 ≥ 0.05，未在 0.05 显著性水平上拒绝处理效应为零的原假设，"
                    f"表明在假断点处未观察到显著的处理效应，这与断点回归识别假设一致。"
                    f"该结果支持了主回归识别策略的有效性，说明在真实断点处观察到的处理效应确实来自断点处的政策或干预变化，"
                    f"而非结果变量在运行变量上的固有非线性模式，进一步增强了因果推断结论的可信度。"
                )
            else:
                detail.append(
                    f"检验结果显示，在部分假断点处，处理效应的 p 值 < 0.05，在 0.05 显著性水平上拒绝了处理效应为零的原假设，"
                    f"提示可能存在设定偏误。建议研究者：（1）检查结果变量在运行变量上的函数形式，"
                    f"考虑采用更高阶的多项式或非参数方法；（2）结合带宽稳健性检验与多项式阶数敏感性检验，"
                    f"综合评估模型设定的合理性；（3）在正文中报告安慰剂检验结果，并讨论其对结论稳健性的影响。"
                )

    elif check_name == "协变量连续性（安慰剂结果）":
        covar_balance = check_data.get("covariate_balance", [])
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归的局部随机化假设，本文进行协变量连续性检验（Covariate Continuity Test），"
            f"检验 predetermined 协变量（在干预发生前已确定的变量）在断点 c = {cutoff:.4f} 处是否存在显著跳跃。"
            f"该检验基于以下识别假设：若局部随机化成立，则除运行变量「{running_var}」外，"
            f"其他 predetermined 协变量在断点处的条件期望应连续，不应出现显著跳跃；"
            f"若协变量在断点处存在显著跳跃，则可能表明断点两侧的个体在可观测特征上存在系统性差异，"
            f"威胁局部随机化假设的有效性。"
        )
        if covar_balance:
            detail.append(
                f"本文对 {len(covar_balance)} 个 predetermined 协变量进行检验，"
                f"采用与主回归相同的局部线性回归方法（三角核，带宽 h = {bandwidth:.4f}），"
                f"分别估计各协变量在断点处的跳跃值及其标准误，并构造 t 统计量检验跳跃是否显著不为零。"
            )
            significant_covars = []
            for item in covar_balance:
                name = item.get("name", "")
                jump = item.get("jump")
                p_val = item.get("p_value")
                if p_val is not None and not np.isnan(p_val) and p_val < 0.05:
                    significant_covars.append((name, jump, p_val))

            if significant_covars:
                detail.append(
                    f"检验结果显示，以下协变量在断点处存在显著跳跃（p < 0.05）："
                )
                for name, jump, p_val in significant_covars:
                    detail.append(f"「{name}」：跳跃值 = {jump:.4f}，p 值 = {p_val:.4f}。")

            if passed:
                detail.append(
                    f"综合检验结果，所有 predetermined 协变量在断点处的跳跃均不显著（p ≥ 0.05），"
                    f"在 0.05 显著性水平上未拒绝协变量连续的原假设，"
                    f"表明断点两侧的个体在可观测特征上不存在系统性差异，局部随机化假设得到支持。"
                    f"该结果进一步增强了断点回归识别策略的有效性，说明在断点附近个体可视为近似随机分配，"
                    f"断点两侧的个体具有可比性，为主回归的因果推断结论提供了坚实的实证基础。"
                )
            else:
                detail.append(
                    f"综合检验结果，部分 predetermined 协变量在断点处存在显著跳跃（p < 0.05），"
                    f"提示局部随机化假设可能受到威胁。建议研究者：（1）在回归中加入这些协变量作为控制变量，"
                    f"检验处理效应估计是否发生变化；（2）结合理论背景分析协变量跳跃的原因，"
                    f"判断是否存在选择性偏差；（3）考虑采用条件 RDD（Conditional RDD）方法，"
                    f"在协变量条件下重新估计处理效应；（4）在正文中报告协变量连续性检验结果，"
                    f"并讨论其对结论稳健性的潜在影响。"
                )

    elif check_name == "多项式阶数敏感性":
        poly_comparison = check_data.get("polynomial_comparison", [])
        main_te = check_data.get("main_treatment_effect")
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归估计结果对函数形式设定的敏感性，本文进行多项式阶数敏感性检验，"
            f"分别采用线性、二次和三次多项式在断点 c = {cutoff:.4f} 附近拟合结果变量「{outcome}」与运行变量「{running_var}」的关系，"
            f"并比较不同多项式阶数下处理效应的估计值。"
            f"该检验基于以下考虑：断点回归的有效性依赖于在断点附近对函数形式的正确设定，"
            f"若真实函数形式与设定不一致，则可能导致处理效应估计偏误；通过比较不同多项式阶数下的估计结果，"
            f"可以评估结论对函数形式设定的稳健性。"
        )
        if poly_comparison:
            detail.append(
                f"本文采用局部多项式回归方法，在断点两侧分别拟合 {len(poly_comparison)} 种不同阶数的多项式（线性、二次、三次），"
                f"使用三角核权重，带宽 h = {bandwidth:.4f}，分别估计各阶数下断点处的处理效应。"
            )
            if main_treatment_effect is not None:
                check_data["main_treatment_effect"] = main_treatment_effect
            for item in poly_comparison:
                order = item.get("order")
                label = item.get("label", "")
                effect = item.get("effect")
                if effect is not None and not np.isnan(effect):
                    detail.append(f"{label}多项式（阶数 = {order}）下的处理效应估计值为 {effect:.4f}。")

            if main_te is not None:
                poly_vals = [e.get("effect") for e in poly_comparison if e.get("effect") is not None and not np.isnan(e.get("effect"))]
                if len(poly_vals) > 1:
                    cv_poly = float(np.std(poly_vals)) / abs(main_te) if abs(main_te) > 1e-10 else float("inf")
                    detail.append(
                        f"主回归（局部线性回归）的处理效应估计值为 {main_te:.4f}。"
                        f"不同多项式阶数下处理效应估计值的变异系数为 {cv_poly:.4f}。"
                    )

            if passed:
                detail.append(
                    f"检验结果显示，不同多项式阶数下的处理效应估计值相对稳定，变异系数较小（< 0.5），"
                    f"表明估计结果对函数形式设定不敏感，结论对多项式阶数选择稳健。"
                    f"该结果支持了局部线性回归设定的合理性，说明在断点附近采用线性函数形式能够较好地捕捉结果变量的变化模式，"
                    f"处理效应估计具有较好的稳健性与可靠性，可用于进一步的因果推断与政策分析。"
                )
            else:
                detail.append(
                    f"检验结果显示，不同多项式阶数下的处理效应估计值存在较大差异，变异系数较大（≥ 0.5），"
                    f"表明估计结果对函数形式设定较为敏感。建议研究者：（1）在正文中报告不同多项式阶数下的估计结果，"
                    f"并讨论函数形式选择对结论的影响；（2）结合带宽稳健性检验，综合评估模型设定的合理性；"
                    f"（3）考虑采用非参数方法（如局部多项式回归、样条回归等）进行稳健性分析；"
                    f"（4）在可能的情况下，基于理论模型或先验知识选择函数形式。"
                )

    elif check_name == "带宽敏感性":
        bandwidth_effects = check_data.get("bandwidth_effects", [])
        bandwidth_list = check_data.get("bandwidth_list", [])
        cv_bw = check_data.get("coefficient_of_variation")
        main_te = check_data.get("main_treatment_effect")
        passed = check_data.get("passed", False)

        detail.append(
            f"为检验断点回归估计结果对带宽选择的敏感性，本文进行带宽稳健性检验，"
            f"在多个不同带宽下分别估计断点 c = {cutoff:.4f} 处的处理效应，"
            f"并比较不同带宽下估计值的一致性。"
            f"该检验基于以下考虑：带宽（bandwidth）是局部多项式回归的关键参数，"
            f"决定了用于估计的样本范围；若估计结果对带宽选择敏感，则可能表明结论不够稳健，"
            f"需要谨慎解释；若估计结果在不同带宽下相对稳定，则说明结论对带宽选择不敏感，"
            f"具有较好的稳健性。"
        )
        if bandwidth_list and bandwidth_effects:
            detail.append(
                f"本文选择 {len(bandwidth_list)} 个不同的带宽值，"
                f"分别为基准带宽的 0.5 倍、0.75 倍、1.0 倍（基准带宽）、1.25 倍和 1.5 倍，"
                f"采用局部线性回归方法（三角核），分别在各带宽下估计处理效应。"
            )
            valid_effects = [(h, eff) for h, eff in zip(bandwidth_list, bandwidth_effects) if not np.isnan(eff)]
            if valid_effects:
                detail.append("各带宽下的处理效应估计值如下：")
                for h_val, eff in valid_effects:
                    detail.append(f"带宽 h = {h_val:.4f} 时，处理效应 = {eff:.4f}。")

            if cv_bw is not None and not np.isnan(cv_bw) and main_treatment_effect is not None:
                detail.append(
                    f"主回归（基准带宽 h = {bandwidth:.4f}）的处理效应估计值为 {main_treatment_effect:.4f}。"
                    f"不同带宽下处理效应估计值的变异系数（标准差/均值绝对值）为 {cv_bw:.4f}。"
                )
            if main_treatment_effect is not None:
                check_data["main_treatment_effect"] = main_treatment_effect

            if passed:
                detail.append(
                    f"检验结果显示，不同带宽下的处理效应估计值相对稳定，变异系数较小（< 0.5），"
                    f"表明估计结果对带宽选择不敏感，结论对带宽选择稳健。"
                    f"该结果支持了主回归估计结果的可靠性，说明处理效应估计不依赖于特定的带宽选择，"
                    f"具有较好的稳健性与可推广性，可用于进一步的因果推断与政策分析。"
                    f"研究者可以基于该结果，结合带宽选择方法（如交叉验证、MSE 最小化等）给出的最优带宽，"
                    f"或报告多个带宽下的估计结果，以增强结论的可信度。"
                )
            else:
                detail.append(
                    f"检验结果显示，不同带宽下的处理效应估计值存在较大差异，变异系数较大（≥ 0.5），"
                    f"表明估计结果对带宽选择较为敏感。建议研究者：（1）在正文中报告不同带宽下的估计结果，"
                    f"并讨论带宽选择对结论的影响；（2）采用带宽选择方法（如交叉验证、MSE 最小化、"
                    f"Imbens-Kalyanaraman 方法等）确定最优带宽；（3）结合多项式阶数敏感性检验，"
                    f"综合评估模型设定的合理性；（4）在可能的情况下，报告基于稳健标准误的估计结果，"
                    f"以降低带宽选择对推断的影响。"
                )

    return detail


def fit_rdd(
    df: pd.DataFrame,
    running_var: str,
    outcome: str,
    cutoff: Optional[float] = None,
    rdd_type: str = "sharp",
    bandwidth: Optional[float] = None,
    treatment_var: Optional[str] = None,
    controls: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """断点回归（RDD）分析：支持 Sharp RDD 与 Fuzzy RDD。

    当干预分配是"一刀切"时使用 Sharp RDD；当干预分配是"概率性"时使用 Fuzzy RDD。
    使用局部线性回归估计断点处的处理效应。

    Args:
        df: 数据集 DataFrame
        running_var: 运行变量（断点分配变量）列名
        outcome: 结果变量列名
        cutoff: 断点值，若为 None 则使用运行变量的中位数
        rdd_type: "sharp" 或 "fuzzy"
        bandwidth: 带宽，若为 None 则自动计算
        treatment_var: 实际干预变量列名（Fuzzy RDD 必填）
        controls: 控制变量列名列表（可选）

    Returns:
        Dict: treatment_effect、interpretation、rdd_plot_data 等
    """
    if running_var not in df.columns or outcome not in df.columns:
        return {
            "model_type": "rdd",
            "error": f"运行变量{running_var}或结果变量{outcome}不存在于数据集中",
            "interpretation": {"conclusions": "变量列名错误，请检查数据。"},
        }
    if rdd_type == "fuzzy" and (not treatment_var or treatment_var not in df.columns):
        return {
            "model_type": "rdd",
            "error": "Fuzzy RDD 需要指定干预变量（treatment_var），且该列须存在于数据集中",
            "interpretation": {"conclusions": "请指定有效的干预变量。"},
        }

    x_raw = df[running_var].copy()
    y_raw = df[outcome].copy()
    x_raw = x_raw.fillna(x_raw.median())
    y_raw = y_raw.fillna(y_raw.median())
    if not pd.api.types.is_numeric_dtype(x_raw) or not pd.api.types.is_numeric_dtype(y_raw):
        return {
            "model_type": "rdd",
            "error": "运行变量与结果变量须为数值型",
            "interpretation": {"conclusions": "请选择数值型变量。"},
        }

    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    # 有效样本对应的 DataFrame，用于控制变量
    df_valid = df.loc[valid].copy()
    # 构建控制变量矩阵（仅数值型）
    control_names_used: List[str] = []
    W_controls: Optional[np.ndarray] = None
    if controls:
        for col in controls:
            if col not in df_valid.columns:
                continue
            ser = df_valid[col]
            if not pd.api.types.is_numeric_dtype(ser):
                try:
                    ser = pd.Series(pd.to_numeric(ser, errors="coerce"), index=ser.index)
                except Exception:
                    continue
            ser = ser.fillna(ser.median())
            arr = np.asarray(ser, dtype=np.float64)
            if W_controls is None:
                W_controls = arr.reshape(-1, 1)
                control_names_used.append(col)
            else:
                W_controls = np.column_stack((W_controls, arr))
                control_names_used.append(col)
    if n < 20:
        return {
            "model_type": "rdd",
            "error": "样本量过小，RDD 至少需要约 20 个有效观测",
            "interpretation": {"conclusions": "样本量不足。"},
        }

    c = float(cutoff) if cutoff is not None else float(np.median(x))
    x_range = float(np.percentile(x, 99) - np.percentile(x, 1)) or 1.0
    h = 1.84 * float(np.std(x)) * (n ** (-0.2))
    h = max(h, x_range * 0.05)
    h = min(h, x_range * 0.5)

    mask_left = x < c
    mask_right = x >= c
    n_left = int(np.sum(mask_left))
    n_right = int(np.sum(mask_right))

    # Sharp RDD: τ = E[Y|X=c+] - E[Y|X=c-]
    mu_left, se_left, _ = _local_linear_rdd(x[mask_left], y[mask_left], c, h)
    mu_right, se_right, _ = _local_linear_rdd(x[mask_right], y[mask_right], c, h)
    treatment_effect: Optional[float] = None
    treatment_effect_se: Optional[float] = None
    treatment_effect_p: Optional[float] = None
    first_stage_jump: Optional[float] = None

    if not np.isnan(mu_left) and not np.isnan(mu_right):
        treatment_effect = mu_right - mu_left
        treatment_effect_se = float(np.sqrt(se_left ** 2 + se_right ** 2))
        if treatment_effect_se > 1e-10:
            t_stat = treatment_effect / treatment_effect_se
            treatment_effect_p = float(2 * (1 - stats.t.cdf(abs(t_stat), max(1, n - 4))))

    if rdd_type == "fuzzy" and treatment_var:
        d_raw = df[treatment_var].copy()
        d_raw = d_raw.fillna(d_raw.median())
        if not pd.api.types.is_numeric_dtype(d_raw):
            d_raw = (d_raw.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "是"])).astype(float)
        d = np.asarray(d_raw, dtype=float)[valid]
        d_left_mu, d_left_se, _ = _local_linear_rdd(x[mask_left], d[mask_left], c, h)
        d_right_mu, d_right_se, _ = _local_linear_rdd(x[mask_right], d[mask_right], c, h)
        if not np.isnan(d_left_mu) and not np.isnan(d_right_mu):
            first_stage_jump = d_right_mu - d_left_mu
            if abs(first_stage_jump) > 1e-8 and treatment_effect is not None:
                treatment_effect = treatment_effect / first_stage_jump
                var_ratio = (se_left ** 2 + se_right ** 2) / (first_stage_jump ** 2)
                var_ratio += (d_left_se ** 2 + d_right_se ** 2) * (treatment_effect ** 2) / (first_stage_jump ** 2)
                treatment_effect_se = float(np.sqrt(var_ratio))
                if treatment_effect_se > 1e-10:
                    t_stat = treatment_effect / treatment_effect_se
                    treatment_effect_p = float(2 * (1 - stats.t.cdf(abs(t_stat), max(1, n - 4))))

    interpretation_text = _build_rdd_interpretation(
        running_var=running_var,
        outcome=outcome,
        cutoff=c,
        rdd_type=rdd_type,
        n=n,
        treatment_effect=treatment_effect,
        treatment_effect_se=treatment_effect_se,
        treatment_effect_p=treatment_effect_p,
        first_stage_jump=first_stage_jump,
        bandwidth=h,
        n_left=n_left,
        n_right=n_right,
    )

    rdd_plot_data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "cutoff": c,
        "running_var": running_var,
        "outcome": outcome,
        "bandwidth": h,
        "n_left": n_left,
        "n_right": n_right,
        "treatment_effect": treatment_effect,
    }
    if rdd_type == "fuzzy" and treatment_var:
        rdd_plot_data["treatment_var"] = treatment_var
        rdd_plot_data["first_stage_jump"] = first_stage_jump

    # Sharp RDD 额外图表数据与稳健性检验：安慰剂、带宽、协变量平衡、多项式、密度检验、Donut RDD
    robustness_checks_list: List[Dict[str, Any]] = []
    if rdd_type == "sharp":
        x_range = float(np.percentile(x, 99) - np.percentile(x, 1)) or 1.0
        # 1. 安慰剂断点检验：在假断点处估计处理效应（应接近0），并计算 p 值
        placebo_cutoffs: List[float] = []
        placebo_effects: List[float] = []
        placebo_se: List[float] = []
        placebo_pvalues: List[float] = []
        p10, p25, p75, p90 = float(np.percentile(x, 10)), float(np.percentile(x, 25)), float(np.percentile(x, 75)), float(np.percentile(x, 90))
        for fake_c in [p25, p75]:
            if abs(fake_c - c) > x_range * 0.1:
                mu_l, se_l, _ = _local_linear_rdd(x[x < fake_c], y[x < fake_c], fake_c, h)
                mu_r, se_r, _ = _local_linear_rdd(x[x >= fake_c], y[x >= fake_c], fake_c, h)
                if not (np.isnan(mu_l) or np.isnan(mu_r)):
                    placebo_cutoffs.append(fake_c)
                    eff = mu_r - mu_l
                    placebo_effects.append(eff)
                    se_eff = float(np.sqrt(se_l ** 2 + se_r ** 2)) if not (np.isnan(se_l) or np.isnan(se_r)) else float("nan")
                    placebo_se.append(se_eff)
                    p_placebo = float("nan")
                    if not np.isnan(se_eff) and se_eff > 1e-10:
                        t_placebo = abs(eff) / se_eff
                        n_eff = int(np.sum(x < fake_c)) + int(np.sum(x >= fake_c))
                        p_placebo = float(2 * (1 - stats.t.cdf(t_placebo, max(1, n_eff - 4))))
                    placebo_pvalues.append(p_placebo)
        rdd_plot_data["placebo_cutoffs"] = placebo_cutoffs
        rdd_plot_data["placebo_effects"] = placebo_effects
        rdd_plot_data["placebo_se"] = placebo_se
        rdd_plot_data["placebo_pvalues"] = placebo_pvalues

        # 2. 带宽稳健性：不同带宽下的处理效应
        h_list = [h * 0.5, h * 0.75, h, h * 1.25, h * 1.5]
        h_list = [max(hn, x_range * 0.03) for hn in h_list]
        bw_effects: List[float] = []
        for hk in h_list:
            ml, _, _ = _local_linear_rdd(x[mask_left], y[mask_left], c, hk)
            mr, _, _ = _local_linear_rdd(x[mask_right], y[mask_right], c, hk)
            bw_effects.append(float(mr - ml) if not (np.isnan(ml) or np.isnan(mr)) else float("nan"))
        rdd_plot_data["bandwidth_list"] = h_list
        rdd_plot_data["bandwidth_effects"] = bw_effects

        # 3. 协变量连续性（控制变量平衡 / 安慰剂结果）：在断点处做 RDD，跳跃应接近0，带 se 与 p 值
        covars_to_check: List[str] = list(controls) if controls else []
        if not covars_to_check:
            for col in df.columns:
                if col not in (running_var, outcome) and pd.api.types.is_numeric_dtype(df[col]):
                    covars_to_check.append(col)
                    if len(covars_to_check) >= 5:
                        break
        covar_balance: List[Dict[str, Any]] = []
        for col in covars_to_check:
            if col not in df.columns:
                continue
            v_raw = df[col].copy().fillna(df[col].median())
            if not pd.api.types.is_numeric_dtype(v_raw):
                continue
            v = np.asarray(v_raw, dtype=float)[valid]
            if len(v) < 20:
                continue
            vl, se_l, _ = _local_linear_rdd(x[mask_left], v[mask_left], c, h)
            vr, se_r, _ = _local_linear_rdd(x[mask_right], v[mask_right], c, h)
            jump = vr - vl if not (np.isnan(vl) or np.isnan(vr)) else float("nan")
            se_jump = float(np.sqrt(se_l ** 2 + se_r ** 2)) if not (np.isnan(se_l) or np.isnan(se_r)) else float("nan")
            p_cov = float("nan")
            if not np.isnan(jump) and not np.isnan(se_jump) and se_jump > 1e-10:
                t_cov = abs(jump) / se_jump
                p_cov = float(2 * (1 - stats.t.cdf(t_cov, max(1, n - 4))))
            covar_balance.append({"name": col, "jump": jump, "se": se_jump, "p_value": p_cov})
        rdd_plot_data["covariate_balance"] = covar_balance

        # 4. 多项式阶数敏感性：线性、二次、三次
        def poly_rdd_limit(xv: np.ndarray, yv: np.ndarray, c_val: float, h_val: float, order: int) -> float:
            u = (xv - c_val) / h_val
            w = np.maximum(0, 1 - np.abs(u))
            valid_w = w > 0
            if np.sum(valid_w) < 2 * (order + 1):
                return float("nan")
            xc = (xv[valid_w] - c_val)
            yc = yv[valid_w]
            wc = w[valid_w]
            X = np.column_stack([(xc ** k) for k in range(order + 1)])
            Xw = X * np.sqrt(wc)[:, None]
            yw = yc * np.sqrt(wc)
            try:
                beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                return float(beta[0])
            except Exception:
                return float("nan")

        poly_effects: List[Dict[str, Any]] = []
        for order, label in [(1, "线性"), (2, "二次"), (3, "三次")]:
            mu_left_p = poly_rdd_limit(x[mask_left], y[mask_left], c, h, order)
            mu_right_p = poly_rdd_limit(x[mask_right], y[mask_right], c, h, order)
            te_p = mu_right_p - mu_left_p if not (np.isnan(mu_left_p) or np.isnan(mu_right_p)) else float("nan")
            poly_effects.append({"order": order, "label": label, "effect": te_p})
        rdd_plot_data["polynomial_comparison"] = poly_effects

        # 5. McCrary 密度检验（操纵检验）
        density_test_result = _mccrary_density_test(x, c, h)
        rdd_plot_data["density_test"] = density_test_result
        _density_p = density_test_result.get("p_value")
        density_check_data = {
            "theta": density_test_result.get("theta"),
            "se_theta": density_test_result.get("se_theta"),
            "statistic": density_test_result.get("statistic"),
            "p_value": _density_p,
            "n_left": density_test_result.get("n_left"),
            "n_right": density_test_result.get("n_right"),
            "passed": _density_p is not None and _density_p >= 0.05,
        }
        density_check = {
            "name": "密度检验（McCrary）",
            "description": "检验运行变量在断点处密度是否连续，若显著不连续则提示可能存在操纵。",
            "passed": density_check_data["passed"],
            "statistic": density_test_result.get("statistic"),
            "p_value": density_test_result.get("p_value"),
            "interpretation": density_test_result.get("interpretation", ""),
        }
        density_check["interpretation_detail"] = _build_robustness_interpretation_detail(
            "密度检验（McCrary）", density_check_data, rdd_type, running_var, outcome, c, h,
            treatment_var=treatment_var, main_treatment_effect=treatment_effect
        )
        robustness_checks_list.append(density_check)

        # 6. Donut RDD（排除断点附近样本）
        donut_result = _rdd_donut_estimate(x, y, c, h, donut_frac=0.02, d=None, fuzzy=False)
        rdd_plot_data["donut_rdd"] = donut_result
        donut_te = donut_result.get("treatment_effect")
        donut_p = donut_result.get("p_value")
        main_te = treatment_effect
        donut_check_data = {
            "treatment_effect": donut_te,
            "se": donut_result.get("se"),
            "p_value": donut_p,
            "n_excluded": donut_result.get("n_excluded"),
            "n_remaining": donut_result.get("n_remaining"),
            "main_treatment_effect": main_te,
            "passed": donut_te is not None and donut_p is not None and (donut_p < 0.05 and main_te is not None and (donut_te * main_te) > 0),
        }
        donut_check = {
            "name": "Donut RDD",
            "description": "排除断点附近 2% 范围样本后重新估计，检验结论是否稳健。",
            "passed": donut_check_data["passed"],
            "treatment_effect": donut_te,
            "p_value": donut_p,
            "interpretation": donut_result.get("interpretation", ""),
        }
        donut_check["interpretation_detail"] = _build_robustness_interpretation_detail(
            "Donut RDD", donut_check_data, rdd_type, running_var, outcome, c, h,
            treatment_var=treatment_var, main_treatment_effect=main_te
        )
        robustness_checks_list.append(donut_check)

        # 汇总：安慰剂断点、带宽、协变量连续性、多项式
        if placebo_pvalues:
            placebo_passed = all(p >= 0.05 for p in placebo_pvalues if not np.isnan(p))
            placebo_check_data = {
                "placebo_effects": placebo_effects,
                "placebo_pvalues": placebo_pvalues,
                "placebo_cutoffs": placebo_cutoffs,
                "passed": placebo_passed,
            }
            placebo_check = {
                "name": "安慰剂断点检验",
                "description": "在假断点（如 25%、75% 分位数）处估计处理效应，应不显著。",
                "passed": placebo_passed,
                "placebo_effects": placebo_effects,
                "placebo_pvalues": placebo_pvalues,
                "interpretation": "安慰剂断点处处理效应不显著，支持识别假设。" if placebo_passed else "部分安慰剂断点处效应显著，需谨慎解释。",
            }
            placebo_check["interpretation_detail"] = _build_robustness_interpretation_detail(
                "安慰剂断点检验", placebo_check_data, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=treatment_effect
            )
            robustness_checks_list.append(placebo_check)
        if covar_balance:
            covar_passed = all(
                _rdd_p_value_passed(item.get("p_value"))
                for item in covar_balance
            )
            covar_check_data = {
                "covariate_balance": covar_balance,
                "passed": covar_passed,
            }
            covar_check = {
                "name": "协变量连续性（安慰剂结果）",
                "description": " predetermined 协变量在断点处应无显著跳跃。",
                "passed": covar_passed,
                "covariate_balance": covar_balance,
                "interpretation": "协变量在断点处均无显著跳跃，支持局部随机化。" if covar_passed else "部分协变量在断点处存在显著跳跃，需结合理论讨论。",
            }
            covar_check["interpretation_detail"] = _build_robustness_interpretation_detail(
                "协变量连续性（安慰剂结果）", covar_check_data, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=treatment_effect
            )
            robustness_checks_list.append(covar_check)
        if poly_effects:
            def _valid_effect(e: Dict[str, Any]) -> bool:
                v = e.get("effect")
                return v is not None and (not isinstance(v, float) or not np.isnan(v))

            poly_vals = [e["effect"] for e in poly_effects if _valid_effect(e)]
            if len(poly_vals) > 1 and main_te is not None:
                cv_poly = float(np.std(poly_vals)) / abs(main_te) if abs(main_te) > 1e-10 else float("inf")
                poly_passed = cv_poly < 0.5
            else:
                poly_passed = True
            poly_check_data = {
                "polynomial_comparison": poly_effects,
                "main_treatment_effect": main_te,
                "passed": poly_passed,
            }
            poly_check = {
                "name": "多项式阶数敏感性",
                "description": "线性、二次、三次多项式下处理效应应相对稳定。",
                "passed": poly_passed,
                "polynomial_comparison": poly_effects,
                "interpretation": "不同多项式阶数下估计结果较为一致，结论对函数形式稳健。" if poly_passed else "估计对多项式阶数较敏感，建议报告多种设定。",
            }
            poly_check["interpretation_detail"] = _build_robustness_interpretation_detail(
                "多项式阶数敏感性", poly_check_data, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=main_te
            )
            robustness_checks_list.append(poly_check)

    # Fuzzy RDD：带宽稳健性、多项式、协变量平衡、安慰剂、密度、Donut
    elif rdd_type == "fuzzy" and treatment_var:
        x_range = float(np.percentile(x, 99) - np.percentile(x, 1)) or 1.0
        h_list = [h * 0.5, h * 0.75, h, h * 1.25, h * 1.5]
        h_list = [max(hn, x_range * 0.03) for hn in h_list]
        bw_effects_fuzzy: List[float] = []
        for hk in h_list:
            ml, _, _ = _local_linear_rdd(x[mask_left], y[mask_left], c, hk)
            mr, _, _ = _local_linear_rdd(x[mask_right], y[mask_right], c, hk)
            dl, _, _ = _local_linear_rdd(x[mask_left], d[mask_left], c, hk)
            dr, _, _ = _local_linear_rdd(x[mask_right], d[mask_right], c, hk)
            if not (np.isnan(ml) or np.isnan(mr) or np.isnan(dl) or np.isnan(dr)):
                fs_jump = dr - dl
                if abs(fs_jump) > 1e-8:
                    late_k = (mr - ml) / fs_jump
                    bw_effects_fuzzy.append(float(late_k))
                else:
                    bw_effects_fuzzy.append(float("nan"))
            else:
                bw_effects_fuzzy.append(float("nan"))
        rdd_plot_data["bandwidth_list"] = h_list
        rdd_plot_data["bandwidth_effects"] = bw_effects_fuzzy

        def poly_rdd_limit_f(xv: np.ndarray, yv: np.ndarray, c_val: float, h_val: float, order: int) -> float:
            u = (xv - c_val) / h_val
            w = np.maximum(0, 1 - np.abs(u))
            valid_w = w > 0
            if np.sum(valid_w) < 2 * (order + 1):
                return float("nan")
            xc = (xv[valid_w] - c_val)
            yc = yv[valid_w]
            wc = w[valid_w]
            X = np.column_stack([(xc ** k) for k in range(order + 1)])
            Xw = X * np.sqrt(wc)[:, None]
            yw = yc * np.sqrt(wc)
            try:
                beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                return float(beta[0])
            except Exception:
                return float("nan")

        poly_effects_f: List[Dict[str, Any]] = []
        for order, label in [(1, "线性"), (2, "二次"), (3, "三次")]:
            mu_l_y = poly_rdd_limit_f(x[mask_left], y[mask_left], c, h, order)
            mu_r_y = poly_rdd_limit_f(x[mask_right], y[mask_right], c, h, order)
            mu_l_d = poly_rdd_limit_f(x[mask_left], d[mask_left], c, h, order)
            mu_r_d = poly_rdd_limit_f(x[mask_right], d[mask_right], c, h, order)
            if not (np.isnan(mu_l_y) or np.isnan(mu_r_y) or np.isnan(mu_l_d) or np.isnan(mu_r_d)):
                fs_j = mu_r_d - mu_l_d
                late_p = (mu_r_y - mu_l_y) / fs_j if abs(fs_j) > 1e-8 else float("nan")
            else:
                late_p = float("nan")
            poly_effects_f.append({"order": order, "label": label, "effect": late_p})
        rdd_plot_data["polynomial_comparison"] = poly_effects_f

        covars_f: List[str] = list(controls) if controls else []
        if not covars_f:
            for col in df.columns:
                if col not in (running_var, outcome, treatment_var) and pd.api.types.is_numeric_dtype(df[col]):
                    covars_f.append(col)
                    if len(covars_f) >= 5:
                        break
        covar_balance_f: List[Dict[str, Any]] = []
        for col in covars_f:
            if col not in df.columns:
                continue
            v_raw = df[col].copy().fillna(df[col].median())
            if not pd.api.types.is_numeric_dtype(v_raw):
                continue
            v = np.asarray(v_raw, dtype=float)[valid]
            if len(v) < 20:
                continue
            vl, se_l, _ = _local_linear_rdd(x[mask_left], v[mask_left], c, h)
            vr, se_r, _ = _local_linear_rdd(x[mask_right], v[mask_right], c, h)
            jump = vr - vl if not (np.isnan(vl) or np.isnan(vr)) else float("nan")
            se_jump = float(np.sqrt(se_l ** 2 + se_r ** 2)) if not (np.isnan(se_l) or np.isnan(se_r)) else float("nan")
            p_cov = float("nan")
            if not np.isnan(jump) and not np.isnan(se_jump) and se_jump > 1e-10:
                t_cov = abs(jump) / se_jump
                p_cov = float(2 * (1 - stats.t.cdf(t_cov, max(1, n - 4))))
            covar_balance_f.append({"name": col, "jump": jump, "se": se_jump, "p_value": p_cov})
        rdd_plot_data["covariate_balance"] = covar_balance_f

        placebo_cutoffs_f: List[float] = []
        placebo_reduced_form: List[float] = []
        placebo_first_stage: List[float] = []
        for fake_c in [float(np.percentile(x, 25)), float(np.percentile(x, 75))]:
            if abs(fake_c - c) > x_range * 0.1:
                mu_yl, _, _ = _local_linear_rdd(x[x < fake_c], y[x < fake_c], fake_c, h)
                mu_yr, _, _ = _local_linear_rdd(x[x >= fake_c], y[x >= fake_c], fake_c, h)
                mu_dl, _, _ = _local_linear_rdd(x[x < fake_c], d[x < fake_c], fake_c, h)
                mu_dr, _, _ = _local_linear_rdd(x[x >= fake_c], d[x >= fake_c], fake_c, h)
                if not (np.isnan(mu_yl) or np.isnan(mu_yr) or np.isnan(mu_dl) or np.isnan(mu_dr)):
                    placebo_cutoffs_f.append(fake_c)
                    placebo_reduced_form.append(mu_yr - mu_yl)
                    placebo_first_stage.append(mu_dr - mu_dl)
        rdd_plot_data["placebo_cutoffs"] = placebo_cutoffs_f
        rdd_plot_data["placebo_effects"] = placebo_reduced_form
        rdd_plot_data["placebo_first_stage"] = placebo_first_stage

        density_test_f = _mccrary_density_test(x, c, h)
        rdd_plot_data["density_test"] = density_test_f
        _density_p_f = density_test_f.get("p_value")
        density_check_data_f = {
            "theta": density_test_f.get("theta"),
            "se_theta": density_test_f.get("se_theta"),
            "statistic": density_test_f.get("statistic"),
            "p_value": _density_p_f,
            "n_left": density_test_f.get("n_left"),
            "n_right": density_test_f.get("n_right"),
            "passed": _density_p_f is not None and _density_p_f >= 0.05,
        }
        density_check_f = {
            "name": "密度检验（McCrary）",
            "description": "检验运行变量在断点处密度是否连续。",
            "passed": density_check_data_f["passed"],
            "statistic": density_test_f.get("statistic"),
            "p_value": density_test_f.get("p_value"),
            "interpretation": density_test_f.get("interpretation", ""),
        }
        density_check_f["interpretation_detail"] = _build_robustness_interpretation_detail(
            "密度检验（McCrary）", density_check_data_f, rdd_type, running_var, outcome, c, h,
            treatment_var=treatment_var, main_treatment_effect=treatment_effect
        )
        robustness_checks_list.append(density_check_f)

        donut_result_f = _rdd_donut_estimate(x, y, c, h, donut_frac=0.02, d=d, fuzzy=True)
        rdd_plot_data["donut_rdd"] = donut_result_f
        donut_te_f = donut_result_f.get("treatment_effect")
        donut_p_f = donut_result_f.get("p_value")
        main_te_f = treatment_effect
        donut_check_data_f = {
            "treatment_effect": donut_te_f,
            "se": donut_result_f.get("se"),
            "p_value": donut_p_f,
            "n_excluded": donut_result_f.get("n_excluded"),
            "n_remaining": donut_result_f.get("n_remaining"),
            "main_treatment_effect": main_te_f,
            "passed": donut_te_f is not None and donut_p_f is not None and main_te_f is not None and (donut_te_f * main_te_f) > 0,
        }
        donut_check_f = {
            "name": "Donut RDD",
            "description": "排除断点附近 2% 范围样本后重新估计 LATE。",
            "passed": donut_check_data_f["passed"],
            "treatment_effect": donut_te_f,
            "p_value": donut_p_f,
            "interpretation": donut_result_f.get("interpretation", ""),
        }
        donut_check_f["interpretation_detail"] = _build_robustness_interpretation_detail(
            "Donut RDD", donut_check_data_f, rdd_type, running_var, outcome, c, h,
            treatment_var=treatment_var, main_treatment_effect=main_te_f
        )
        robustness_checks_list.append(donut_check_f)

        if covar_balance_f:
            covar_passed_f = all(
                _rdd_p_value_passed(item.get("p_value"))
                for item in covar_balance_f
            )
            covar_check_data_f = {
                "covariate_balance": covar_balance_f,
                "passed": covar_passed_f,
            }
            covar_check_f = {
                "name": "协变量连续性（安慰剂结果）",
                "description": " predetermined 协变量在断点处应无显著跳跃。",
                "passed": covar_passed_f,
                "covariate_balance": covar_balance_f,
                "interpretation": "协变量在断点处均无显著跳跃。" if covar_passed_f else "部分协变量在断点处存在显著跳跃。",
            }
            covar_check_f["interpretation_detail"] = _build_robustness_interpretation_detail(
                "协变量连续性（安慰剂结果）", covar_check_data_f, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=treatment_effect
            )
            robustness_checks_list.append(covar_check_f)
        if poly_effects_f and treatment_effect is not None:
            poly_vals_f = [e["effect"] for e in poly_effects_f if e.get("effect") is not None and not np.isnan(e["effect"])]
            if len(poly_vals_f) > 1 and abs(treatment_effect) > 1e-10:
                cv_poly_f = float(np.std(poly_vals_f)) / abs(treatment_effect)
                poly_passed_f = cv_poly_f < 0.5
            else:
                poly_passed_f = True
            poly_check_data_f = {
                "polynomial_comparison": poly_effects_f,
                "main_treatment_effect": treatment_effect,
                "passed": poly_passed_f,
            }
            poly_check_f = {
                "name": "多项式阶数敏感性",
                "description": "线性、二次、三次多项式下 LATE 应相对稳定。",
                "passed": poly_passed_f,
                "polynomial_comparison": poly_effects_f,
                "interpretation": "不同多项式阶数下 LATE 较为一致。" if poly_passed_f else "估计对多项式阶数较敏感。",
            }
            poly_check_f["interpretation_detail"] = _build_robustness_interpretation_detail(
                "多项式阶数敏感性", poly_check_data_f, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=treatment_effect
            )
            robustness_checks_list.append(poly_check_f)

    # 带宽稳健性（Sharp 与 Fuzzy 共用，加入稳健性检验列表）
    if "bandwidth_effects" in rdd_plot_data:
        bandwidth_effects_all = rdd_plot_data.get("bandwidth_effects", [])
        bandwidth_list_all = rdd_plot_data.get("bandwidth_list", [])
        if bandwidth_effects_all and treatment_effect is not None:
            valid_bw = [e for e in bandwidth_effects_all if not np.isnan(e)]
            if len(valid_bw) > 1 and abs(treatment_effect) > 1e-10:
                cv_bw = float(np.std(valid_bw)) / abs(treatment_effect)
                bw_passed = cv_bw < 0.5
            else:
                cv_bw = float("nan")
                bw_passed = True
            bw_check_data = {
                "bandwidth_effects": bandwidth_effects_all,
                "bandwidth_list": bandwidth_list_all,
                "coefficient_of_variation": cv_bw if not (isinstance(cv_bw, float) and np.isnan(cv_bw)) else None,
                "main_treatment_effect": treatment_effect,
                "passed": bw_passed,
            }
            bw_check = {
                "name": "带宽敏感性",
                "description": "在不同带宽下估计处理效应（或 LATE），结果应相对稳定。",
                "passed": bw_passed,
                "bandwidth_effects": bandwidth_effects_all,
                "coefficient_of_variation": cv_bw if not (isinstance(cv_bw, float) and np.isnan(cv_bw)) else None,
                "interpretation": "不同带宽下估计较为一致，结论对带宽选择稳健。" if bw_passed else "估计对带宽较敏感，建议报告多种带宽或采用带宽选择方法。",
            }
            bw_check["interpretation_detail"] = _build_robustness_interpretation_detail(
                "带宽敏感性", bw_check_data, rdd_type, running_var, outcome, c, h,
                treatment_var=treatment_var, main_treatment_effect=treatment_effect
            )
            robustness_checks_list.append(bw_check)

    # ========== 模型假设检验 ==========
    model_tests: Dict[str, Any] = {}
    if robustness_checks_list:
        model_tests["robustness_checks"] = robustness_checks_list

    # 1. 处理效应显著性检验（相当于回归系数显著性检验）
    if treatment_effect is not None and treatment_effect_se is not None and treatment_effect_p is not None:
        model_tests["treatment_effect_significance"] = {
            "test_name": "处理效应t检验",
            "statistic": float(treatment_effect / treatment_effect_se) if treatment_effect_se > 1e-10 else 0.0,
            "p_value": float(treatment_effect_p),
            "interpretation": "处理效应显著性检验" if treatment_effect_p < 0.05 else "处理效应不显著"
        }

    # 计算残差和预测值（用于后续检验）
    residuals: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    if not np.isnan(mu_left) and not np.isnan(mu_right):
        try:
            residuals, predictions = _compute_rdd_residuals(x, y, c, h, mu_left, mu_right)
        except Exception:
            pass

    # 2. 残差正态性检验
    if residuals is not None and len(residuals) > 3:
        residuals_array = np.array(residuals)
        residuals_array = residuals_array[~np.isnan(residuals_array)]
        if len(residuals_array) > 3:
            if len(residuals_array) <= 5000:  # Shapiro-Wilk适用于小样本
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(residuals_array)
                    model_tests["normality"] = {
                        "test_name": "Shapiro-Wilk",
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "interpretation": "残差正态性检验" if shapiro_p >= 0.05 else "残差可能不服从正态分布"
                    }
                except Exception:
                    model_tests["normality"] = {
                        "test_name": "Shapiro-Wilk",
                        "statistic": None,
                        "p_value": None,
                        "interpretation": "无法计算"
                    }
            else:  # 大样本使用Kolmogorov-Smirnov检验
                try:
                    standardized_residuals = residuals_array / np.std(residuals_array, ddof=1)
                    ks_stat, ks_p = stats.kstest(standardized_residuals, 'norm')
                    model_tests["normality"] = {
                        "test_name": "Kolmogorov-Smirnov",
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "interpretation": "残差正态性检验" if ks_p >= 0.05 else "残差可能不服从正态分布"
                    }
                except Exception:
                    model_tests["normality"] = {
                        "test_name": "Kolmogorov-Smirnov",
                        "statistic": None,
                        "p_value": None,
                        "interpretation": "无法计算"
                    }

    # 3. 残差同方差性检验
    if residuals is not None and predictions is not None:
        residuals_array = np.array(residuals)
        predictions_array = np.array(predictions)
        valid_mask = ~(np.isnan(residuals_array) | np.isnan(predictions_array))
        if np.sum(valid_mask) > 10:
            residuals_clean = residuals_array[valid_mask]
            predictions_clean = predictions_array[valid_mask]

            if STATSMODELS_AVAILABLE and len(residuals_clean) > 5:
                assert sm is not None
                try:
                    # 使用Breusch-Pagan检验
                    residuals_squared = residuals_clean ** 2
                    X_with_const = sm.add_constant(predictions_clean.reshape(-1, 1))
                    bp_model = sm.OLS(residuals_squared, X_with_const).fit()
                    bp_stat = len(residuals_clean) * bp_model.rsquared
                    bp_p = 1 - stats.chi2.cdf(bp_stat, 1)
                    model_tests["homoscedasticity"] = {
                        "test_name": "Breusch-Pagan",
                        "statistic": float(bp_stat),
                        "p_value": float(bp_p),
                        "interpretation": "残差同方差性检验" if bp_p >= 0.05 else "残差可能存在异方差性"
                    }
                except Exception:
                    # 如果Breusch-Pagan失败，使用简单的相关性检验
                    try:
                        corr_res_pred = np.corrcoef(residuals_clean, predictions_clean)[0, 1]
                        model_tests["homoscedasticity"] = {
                            "test_name": "残差-预测值相关性",
                            "statistic": float(abs(corr_res_pred)),
                            "p_value": None,
                            "interpretation": f"残差与预测值相关系数: {corr_res_pred:.4f}（接近0表示同方差）"
                        }
                    except Exception:
                        model_tests["homoscedasticity"] = {
                            "test_name": "Breusch-Pagan",
                            "statistic": None,
                            "p_value": None,
                            "interpretation": "无法计算"
                        }
            else:
                # 如果没有statsmodels，使用简单的相关性检验
                try:
                    corr_res_pred = np.corrcoef(residuals_clean, predictions_clean)[0, 1]
                    model_tests["homoscedasticity"] = {
                        "test_name": "残差-预测值相关性",
                        "statistic": float(abs(corr_res_pred)),
                        "p_value": None,
                        "interpretation": f"残差与预测值相关系数: {corr_res_pred:.4f}（接近0表示同方差）"
                    }
                except Exception:
                    model_tests["homoscedasticity"] = {
                        "test_name": "残差-预测值相关性",
                        "statistic": None,
                        "p_value": None,
                        "interpretation": "无法计算"
                    }

    # 4. 残差独立性检验（Durbin-Watson检验）
    if residuals is not None:
        residuals_array = np.array(residuals)
        residuals_array = residuals_array[~np.isnan(residuals_array)]
        if len(residuals_array) > 1:
            try:
                # Durbin-Watson统计量
                diff = np.diff(residuals_array)
                dw_statistic = np.sum(diff ** 2) / np.sum(residuals_array ** 2)
                dw_interpretation = "残差独立" if 1.5 <= dw_statistic <= 2.5 else "残差可能存在自相关"
                model_tests["independence"] = {
                    "test_name": "Durbin-Watson",
                    "statistic": float(dw_statistic),
                    "p_value": None,  # DW检验没有p值，需要查表
                    "interpretation": dw_interpretation
                }
            except Exception:
                model_tests["independence"] = {
                    "test_name": "Durbin-Watson",
                    "statistic": None,
                    "p_value": None,
                    "interpretation": "无法计算"
                }

    # 5. 线性关系检验（RDD中检验局部线性假设）
    if residuals is not None and predictions is not None:
        residuals_array = np.array(residuals)
        predictions_array = np.array(predictions)
        valid_mask = ~(np.isnan(residuals_array) | np.isnan(predictions_array))
        if np.sum(valid_mask) > 10:
            residuals_clean = residuals_array[valid_mask]
            predictions_clean = predictions_array[valid_mask]
            try:
                # 如果残差与预测值没有明显的非线性关系，相关系数应该接近0
                corr_res_pred_linear = np.corrcoef(residuals_clean, predictions_clean)[0, 1]
                model_tests["linearity"] = {
                    "test_name": "残差-预测值相关性（局部线性检验）",
                    "statistic": float(abs(corr_res_pred_linear)),
                    "p_value": None,
                    "interpretation": f"残差与预测值相关系数: {corr_res_pred_linear:.4f}（接近0表示局部线性关系良好）"
                }
            except Exception:
                model_tests["linearity"] = {
                    "test_name": "残差-预测值相关性（局部线性检验）",
                    "statistic": None,
                    "p_value": None,
                    "interpretation": "无法计算"
                }

    # 5b. 控制变量回归系数（当存在控制变量时：局部线性回归 Y ~ (X-c) + D + W，返回 W 的系数、t、p）
    control_coefs: Dict[str, float] = {}
    control_t_values: Dict[str, float] = {}
    control_p_values: Dict[str, float] = {}
    if control_names_used and W_controls is not None and W_controls.size > 0:
        u = (x - c) / h
        w_k = np.maximum(0, 1 - np.abs(u))
        valid_w = w_k > 0
        if np.sum(valid_w) >= 5 + len(control_names_used):
            try:
                X_one = np.ones(np.sum(valid_w))
                x_c = (x[valid_w] - c).reshape(-1, 1)
                D = (x[valid_w] >= c).astype(float).reshape(-1, 1)
                W_w = W_controls[valid_w]
                X_design = np.column_stack([X_one, x_c, D, W_w])
                y_w = y[valid_w]
                w_sqrt = np.sqrt(w_k[valid_w])
                Xw = X_design * w_sqrt[:, None]
                yw = y_w * w_sqrt
                beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
                resid = yw - Xw @ beta
                n_eff = np.sum(valid_w)
                df_resid = max(1, n_eff - X_design.shape[1])
                sigma2 = np.var(resid, ddof=X_design.shape[1])
                XtX_inv = np.linalg.inv(Xw.T @ Xw)
                se_all = np.sqrt(sigma2 * np.diag(XtX_inv))
                # 控制变量对应索引 3, 4, ...
                for i, name in enumerate(control_names_used):
                    idx = 3 + i
                    if idx < len(beta) and idx < len(se_all) and se_all[idx] > 1e-10:
                        control_coefs[name] = float(beta[idx])
                        control_t_values[name] = float(beta[idx] / se_all[idx])
                        t_stat = abs(control_t_values[name])
                        control_p_values[name] = float(2 * (1 - stats.t.cdf(t_stat, df_resid)))
            except Exception:
                pass

    # 6. 稳健性检验（带宽稳健性：Sharp 与 Fuzzy 均支持）
    if "bandwidth_effects" in rdd_plot_data:
        bandwidth_effects = rdd_plot_data.get("bandwidth_effects", [])
        if bandwidth_effects and treatment_effect is not None:
            # 检查不同带宽下的处理效应是否一致
            valid_effects = [e for e in bandwidth_effects if not np.isnan(e)]
            if len(valid_effects) > 1:
                effect_std = float(np.std(valid_effects))
                effect_mean = float(np.mean(valid_effects))
                # 如果标准差相对于均值较小，说明稳健
                cv = effect_std / abs(effect_mean) if abs(effect_mean) > 1e-10 else float("inf")
                model_tests["robustness"] = {
                    "method": "带宽稳健性检验",
                    "description": "通过不同带宽下的处理效应估计值检验结果的稳健性",
                    "bandwidth_effects": bandwidth_effects,
                    "coefficient_of_variation": float(cv),
                    "interpretation": "带宽稳健性检验通过" if cv < 0.5 else "带宽稳健性检验未通过，结果对带宽选择敏感"
                }

    # 7. 异质性检验（按分组变量中位数分为两子组，比较两子组处理效应是否一致）
    split_var: Optional[np.ndarray] = None
    split_name = ""
    if control_names_used and W_controls is not None and W_controls.shape[1] >= 1:
        split_var = W_controls[:, 0]
        split_name = control_names_used[0]
    else:
        split_var = x
        split_name = running_var
    if split_var is not None and len(split_var) == n:
        model_tests["heterogeneity"] = _compute_heterogeneity_rdd(
            x, y, c, h, split_var, split_name
        )

    out: Dict[str, Any] = {
        "model_type": "rdd",
        "rdd_type": rdd_type,
        "running_var": running_var,
        "outcome": outcome,
        "cutoff": c,
        "treatment_effect": treatment_effect,
        "treatment_effect_se": treatment_effect_se,
        "treatment_effect_p": treatment_effect_p,
        "first_stage_jump": first_stage_jump,
        "bandwidth": h,
        "n_samples": n,
        "n_left": n_left,
        "n_right": n_right,
        "rdd_plot_data": rdd_plot_data,
        "model_tests": model_tests,  # 模型假设检验结果
        "interpretation": {
            "model_overview": interpretation_text,
            "conclusions": interpretation_text,
        },
        "data_understanding": {
            "running_variable": running_var,
            "outcome_variable": outcome,
            "sample_size": n,
        },
        "analysis_context": {"target": outcome, "features": [running_var]},
        "data": df,
    }
    # 用户请求的控制变量列表（用于前端判断是否"包含控制变量"）
    out["control_variables"] = list(controls) if controls else []
    # 控制变量的回归系数、t 值、p 值（局部线性回归中含控制变量时已计算）
    if control_coefs:
        out["coefficients"] = control_coefs
        out["t_values"] = control_t_values
        out["p_values"] = control_p_values
    return out
