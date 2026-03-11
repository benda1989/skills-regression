"""Logistic regression standalone function."""
from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None

from ._helpers import _compute_robustness_logit, _compute_heterogeneity_lr_logit


def fit_logistic(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    standardize: bool = True,
    max_iter: int = 1000,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合逻辑回归模型。

    Args:
        df: 数据集 DataFrame
        target: 目标变量（因变量）列名，应为二分类变量
        features: 特征（自变量）列名列表
        standardize: 是否对特征进行标准化，默认 True（逻辑回归通常需要标准化）
        max_iter: 最大迭代次数，默认 1000

    Returns:
        Dict[str, Any]: 包含 model_type、coefficients、intercept、odds_ratios、p_values、accuracy 等
    """
    X: pd.DataFrame = pd.DataFrame(df[features].copy())
    y: pd.Series = pd.Series(df[target].copy())

    # 处理缺失值：数值列用均值，分类列用众数
    for col in X.columns:
        col_series = cast(pd.Series, X[col])
        if pd.api.types.is_numeric_dtype(col_series):
            X[col] = col_series.fillna(col_series.mean())
        else:
            # 分类变量用众数填充
            mode_val = col_series.mode()
            if len(mode_val) > 0:
                X[col] = col_series.fillna(mode_val.iloc[0])
            else:
                fallback = col_series.iloc[0] if len(col_series) > 0 else ""
                X[col] = col_series.fillna(fallback)

    # 处理目标变量的缺失值
    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0])
    else:
        mode_val = y.mode()
        if len(mode_val) > 0:
            y = cast(pd.Series, y.fillna(mode_val.iloc[0]))

    # 编码分类变量为数值
    label_encoders: Dict[str, LabelEncoder] = {}
    X_encoded = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # 确保目标变量是数值型（二分类：0和1）
    if not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
    else:
        # 如果是数值型，确保是二分类（0和1）
        unique_values = y.unique()
        if len(unique_values) != 2:
            raise ValueError(f"逻辑回归要求目标变量必须是二分类，但发现 {len(unique_values)} 个类别: {unique_values}")
        # 将目标变量转换为0和1
        y_min, y_max = y.min(), y.max()
        y = (y - y_min) / (y_max - y_min) if y_max > y_min else y
        y = y.round().astype(int)

    X = X_encoded
    y = y.astype(int)

    # 保存原始数据用于计算非标准化系数（与SPSS对齐）
    X_original = X.to_numpy()
    y_original = y.to_numpy()

    # 计算原始数据的均值和标准差（用于系数转换）
    X_mean = np.mean(X_original, axis=0)
    X_std = np.std(X_original, axis=0, ddof=0)  # 使用总体标准差（与SPSS一致）

    # 逻辑回归通常需要对特征进行标准化（内部拟合使用标准化数据）
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.to_numpy()
    X_scaled = np.asarray(X_scaled, dtype=float)

    # 拟合逻辑回归模型（使用标准化数据）
    model = LogisticRegression(max_iter=max_iter, solver='lbfgs', random_state=42)
    model.fit(X_scaled, y)

    # 预测（使用标准化数据）
    y_pred_proba_scaled = model.predict_proba(X_scaled)[:, 1]  # 正类概率
    y_pred = model.predict(X_scaled)  # 分类预测

    # 计算分类指标
    # zero_division 参数在运行时可以是 float (0.0, 1.0) 或 str ("warn")
    # 但类型检查器的类型定义不完整，使用类型忽略注释
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0.0)  # pyright: ignore[reportArgumentType]
    recall = recall_score(y, y_pred, zero_division=0.0)  # pyright: ignore[reportArgumentType]
    f1 = f1_score(y, y_pred, zero_division=0.0)  # pyright: ignore[reportArgumentType]

    # ROC AUC（如果只有两个类别）
    try:
        roc_auc = roc_auc_score(y, y_pred_proba_scaled)
    except Exception:
        roc_auc = None

    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # 将标准化系数转换回原始尺度（非标准化系数，与SPSS对齐）
    # 对于逻辑回归，系数转换：B_unstandardized = B_standardized / std_x
    coefficients = {}
    for i, (feat, coef_scaled) in enumerate(zip(features, model.coef_[0])):
        if standardize:
            if X_std[i] > 0:
                coef_unstandardized = coef_scaled / X_std[i]
            else:
                coef_unstandardized = 0.0
        else:
            coef_unstandardized = coef_scaled
        coefficients[feat] = float(coef_unstandardized)

    # 截距转换：intercept_unstandardized = intercept_standardized - sum(coef_unstandardized * mean_x)
    # model.intercept_ 是一个一维数组，取第一个元素
    intercept_scaled = float(model.intercept_[0])  # type: ignore[index]
    if standardize:
        intercept = float(intercept_scaled - np.sum([coefficients[feat] * X_mean[i] for i, feat in enumerate(features)]))
    else:
        intercept = float(intercept_scaled)

    # 计算优势比（OR = exp(coefficient)）
    odds_ratios = {feat: float(np.exp(coef)) for feat, coef in coefficients.items()}
    intercept_or = float(np.exp(intercept))

    # 计算统计量（使用原始数据，与SPSS对齐）
    n = len(y)
    p = len(features)

    # 使用原始数据计算标准误（与SPSS对齐）
    # 逻辑回归的标准误基于Hessian矩阵的逆
    Z = np.column_stack([np.ones(n), X_original])

    # 使用非标准化系数计算预测概率
    logit = intercept + X_original @ np.array([coefficients[feat] for feat in features])
    prob = 1 / (1 + np.exp(-logit))

    # 重新计算预测概率（用于后续计算）
    y_pred_proba = prob

    # 计算权重矩阵（对角矩阵，元素为 p(1-p)）
    W = np.diag(prob * (1 - prob))

    # 计算Hessian矩阵的负值：-X'*W*X
    hessian = -Z.T @ W @ Z

    # 协方差矩阵 = -Hessian的逆
    try:
        cov_beta = -np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆，使用伪逆
        cov_beta = -np.linalg.pinv(hessian)

    # 确保 cov_beta 是 numpy 数组，并且 diag 返回的是数组
    cov_beta = np.asarray(cov_beta, dtype=float)
    diag_values = np.diag(cov_beta)
    # 确保 diag_values 是数组格式
    if np.isscalar(diag_values):
        diag_values = np.array([float(diag_values) if not isinstance(diag_values, complex) else float(diag_values.real)], dtype=float)
    else:
        diag_values = np.asarray(diag_values, dtype=float)
    se_all = np.sqrt(diag_values)

    # 截距在第 0 个位置，其余是各自变量
    intercept_se = float(se_all[0])
    se_coef_arr = se_all[1:]

    # 计算Wald统计量（类似t统计量）和p值（与SPSS一致）
    coef_array = np.array([coefficients[feat] for feat in features])
    wald_stats = coef_array / se_coef_arr

    p_values = {
        feat: float(2 * (1 - stats.norm.cdf(abs(w))))  # 使用正态分布（大样本，与SPSS一致）
        for feat, w in zip(features, wald_stats)
    }
    wald_values = {feat: float(w) for feat, w in zip(features, wald_stats)}
    se_values = {feat: float(se) for feat, se in zip(features, se_coef_arr)}

    # 截距的 Wald 和 p
    intercept_wald = intercept / intercept_se if intercept_se > 0 else 0.0
    intercept_p = float(2 * (1 - stats.norm.cdf(abs(intercept_wald))))

    # 计算95%置信区间（与SPSS一致）
    z_critical = stats.norm.ppf(0.975)  # 95%置信区间的Z临界值
    intercept_ci = [
        float(intercept - z_critical * intercept_se),
        float(intercept + z_critical * intercept_se)
    ]
    ci_values = {
        feat: [
            float(coefficients[feat] - z_critical * se_values[feat]),
            float(coefficients[feat] + z_critical * se_values[feat])
        ]
        for feat in features
    }

    # 计算OR的95%置信区间（与SPSS一致）
    intercept_or_ci = [
        float(np.exp(intercept_ci[0])),
        float(np.exp(intercept_ci[1]))
    ]
    or_ci_values = {
        feat: [
            float(np.exp(ci_values[feat][0])),
            float(np.exp(ci_values[feat][1]))
        ]
        for feat in features
    }

    # 计算伪R²（McFadden's R²和Cox & Snell R²）
    # 对数似然
    log_likelihood_model = np.sum(y * np.log(prob + 1e-10) + (1 - y) * np.log(1 - prob + 1e-10))
    # 零模型的对数似然（只有截距）
    y_mean = np.mean(y)
    log_likelihood_null = n * (y_mean * np.log(y_mean + 1e-10) + (1 - y_mean) * np.log(1 - y_mean + 1e-10))

    # McFadden's R²
    mcfadden_r2 = 1 - (log_likelihood_model / log_likelihood_null) if log_likelihood_null != 0 else 0.0

    # Cox & Snell R²
    cox_snell_r2 = 1 - np.exp(2 * (log_likelihood_null - log_likelihood_model) / n)

    # Nagelkerke R²（调整后的Cox & Snell R²）
    max_r2 = 1 - np.exp(2 * log_likelihood_null / n)
    nagelkerke_r2 = cox_snell_r2 / max_r2 if max_r2 > 0 else 0.0

    # 计算VIF（方差膨胀因子）
    vif_values = {}
    if len(features) > 1:
        for i, feat in enumerate(features):
            # 将其他特征作为自变量，当前特征作为因变量
            X_other = np.delete(X_scaled, i, axis=1)
            y_feat = X_scaled[:, i]
            model_vif = LinearRegression()
            model_vif.fit(X_other, y_feat)
            r2_vif = r2_score(y_feat, model_vif.predict(X_other))
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            vif_values[feat] = float(vif)
    else:
        vif_values[features[0]] = 1.0

    # Hosmer-Lemeshow检验（简化版本，用于模型拟合优度）
    # 将预测概率分成10组
    n_groups = min(10, n // 5)
    if n_groups >= 2:
        sorted_indices = np.argsort(prob)
        group_size = n // n_groups
        hl_statistic = 0.0
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else n
            group_indices = sorted_indices[start_idx:end_idx]
            group_observed = np.sum(y_original[group_indices])
            group_expected = np.sum(prob[group_indices])
            group_n = len(group_indices)
            if group_expected > 0 and (group_n - group_expected) > 0:
                hl_statistic += ((group_observed - group_expected)**2) / (group_expected * (group_n - group_expected))
        hl_p_value = 1 - stats.chi2.cdf(hl_statistic, n_groups - 2) if n_groups > 2 else None
    else:
        hl_statistic = None
        hl_p_value = None

    # ========== 模型假设检验（逻辑回归特有）==========
    model_tests = {}

    # 1. 模型整体显著性检验（似然比检验）
    # 似然比统计量 = -2 * (LL_null - LL_model)
    likelihood_ratio_stat = -2 * (log_likelihood_null - log_likelihood_model)
    # 自由度 = 特征数 p
    likelihood_ratio_p = 1 - stats.chi2.cdf(likelihood_ratio_stat, p) if p > 0 else None
    model_tests["model_significance"] = {
        "test_name": "似然比检验",
        "statistic": float(likelihood_ratio_stat),
        "p_value": float(likelihood_ratio_p) if likelihood_ratio_p is not None else None,
        "interpretation": "模型整体显著" if (likelihood_ratio_p is not None and likelihood_ratio_p < 0.05) else "模型整体不显著" if likelihood_ratio_p is not None else "无法判断"
    }

    # 2. 回归系数显著性检验（Wald检验已在上面计算，这里只是汇总）
    # Wald检验结果已经在p_values和wald_values中，这里不需要额外计算

    # 3. 模型拟合优度检验（Hosmer-Lemeshow检验）
    if hl_statistic is not None and hl_p_value is not None:
        model_tests["goodness_of_fit"] = {
            "test_name": "Hosmer-Lemeshow",
            "statistic": float(hl_statistic),
            "p_value": float(hl_p_value),
            "interpretation": "模型拟合良好" if hl_p_value >= 0.05 else "模型拟合不佳"
        }
    else:
        model_tests["goodness_of_fit"] = {
            "test_name": "Hosmer-Lemeshow",
            "statistic": None,
            "p_value": None,
            "interpretation": "无法计算（样本量太小）"
        }

    # 4. 线性关系检验（logit线性）
    # 检查预测概率的logit与自变量的线性关系
    # 通过检查预测概率与自变量的相关性来判断
    try:
        # 计算logit值
        logit_values = np.log(prob / (1 - prob + 1e-10) + 1e-10)
        # 计算logit与第一个主要自变量的相关性（如果有多个自变量，取第一个）
        if p > 0:
            main_feature_idx = 0
            corr_logit_feature = np.corrcoef(logit_values, X_original[:, main_feature_idx])[0, 1]
            model_tests["linearity"] = {
                "test_name": "Logit线性关系检验",
                "statistic": float(abs(corr_logit_feature)),
                "p_value": None,
                "interpretation": f"Logit与自变量相关系数: {corr_logit_feature:.4f}（接近1表示线性关系良好）"
            }
        else:
            model_tests["linearity"] = {
                "test_name": "Logit线性关系检验",
                "statistic": None,
                "p_value": None,
                "interpretation": "无法计算"
            }
    except Exception:
        model_tests["linearity"] = {
            "test_name": "Logit线性关系检验",
            "statistic": None,
            "p_value": None,
            "interpretation": "无法计算"
        }

    # 5. 多重共线性检验（VIF已在上面计算，这里只是汇总）
    # VIF值已经在vif_values中，这里不需要额外计算

    # 稳健性检验：异方差稳健标准误(HC1)
    model_tests["robustness"] = _compute_robustness_logit(X_original, y_original, features, p_values)

    # 异质性检验（似然比，分组系数稳定性）
    model_tests["heterogeneity"] = _compute_heterogeneity_lr_logit(
        X_original, y_original, features, log_likelihood_model, p + 1
    )

    return {
        "model_type": "logistic",
        "coefficients": coefficients,
        "intercept": intercept,
        "intercept_se": float(intercept_se),
        "intercept_wald": float(intercept_wald),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "odds_ratios": odds_ratios,
        "intercept_or": intercept_or,
        "intercept_or_ci": intercept_or_ci,
        "or_ci_values": or_ci_values,
        "p_values": p_values,
        "wald_values": wald_values,
        "se_values": se_values,
        "mcfadden_r_squared": float(mcfadden_r2),
        "cox_snell_r_squared": float(cox_snell_r2),
        "nagelkerke_r_squared": float(nagelkerke_r2),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "vif_values": vif_values,
        "hosmer_lemeshow_statistic": float(hl_statistic) if hl_statistic is not None else None,
        "hosmer_lemeshow_p_value": float(hl_p_value) if hl_p_value is not None else None,
        "n_samples": n,
        "n_features": p,
        "y_pred": y_pred.tolist(),  # 分类预测
        "y_pred_proba": y_pred_proba.tolist(),  # 预测概率
        "log_likelihood_model": float(log_likelihood_model),  # 模型对数似然
        "log_likelihood_null": float(log_likelihood_null),  # 零模型对数似然
        "likelihood_ratio_statistic": float(likelihood_ratio_stat),  # 似然比统计量
        "likelihood_ratio_p_value": float(likelihood_ratio_p) if likelihood_ratio_p is not None else None,  # 似然比检验p值
        "model_tests": model_tests,  # 模型假设检验结果
    }
