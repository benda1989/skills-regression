"""Linear, Ridge, and Lasso regression standalone functions."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None

from ._helpers import _compute_robustness_ols, _compute_heterogeneity_chow_ols, _compute_heterogeneity_chow_ridge


def fit_linear(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    standardize: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合线性回归模型。

    Args:
        df: 数据集 DataFrame
        target: 目标变量（因变量）列名
        features: 特征（自变量）列名列表
        standardize: 是否对特征进行标准化，默认 False（对齐 SPSS 的非标准化系数）

    Returns:
        Dict[str, Any]: 包含 model_type、coefficients、intercept、r_squared、rmse、mse、p_values、n_samples、n_features
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
        y = y.fillna(y.mean())
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

    # 确保目标变量是数值型
    if not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)

    X = X_encoded

    # 标准化
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.to_numpy()
    X_scaled = np.asarray(X_scaled, dtype=float)

    # 拟合模型
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 预测和评估
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    # 确保 mse 是 float 类型
    mse = float(mse) if not isinstance(mse, np.ndarray) else float(mse.item() if mse.size == 1 else mse)
    rmse = float(np.sqrt(mse))

    # 系数和统计量
    coefficients = {feat: float(coef) for feat, coef in zip(features, model.coef_)}
    intercept = float(model.intercept_)

    # 计算统计量（使用原始数据，与SPSS对齐）
    n = len(y)
    p = len(features)
    residuals = y - y_pred
    mse_residual = np.sum(residuals**2) / (n - p - 1)

    # 使用原始数据的设计矩阵 Z=[1, X] 计算协方差矩阵（与SPSS一致）
    # 即使内部使用了标准化，标准误的计算也应该基于原始数据
    X_original_for_se = X.to_numpy() if not standardize else X.to_numpy()
    Z = np.column_stack([np.ones(n), X_original_for_se])
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    cov_beta = mse_residual * XtX_inv
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

    # 计算系数标准误差、t统计量和p值（使用非标准化系数，与SPSS一致）
    coef_array = np.array([coefficients[feat] for feat in features])
    t_stats = coef_array / se_coef_arr
    df_resid = n - p - 1
    p_values = {
        feat: float(2 * (1 - stats.t.cdf(abs(t), df_resid)))
        for feat, t in zip(features, t_stats)
    }
    t_values = {feat: float(t) for feat, t in zip(features, t_stats)}
    se_values = {feat: float(se) for feat, se in zip(features, se_coef_arr)}

    # 截距的 t 和 p（使用非标准化截距，与SPSS对齐）
    intercept_t = intercept / intercept_se if intercept_se > 0 else 0.0
    intercept_p = float(2 * (1 - stats.t.cdf(abs(intercept_t), df_resid)))

    # 计算95%置信区间（与SPSS一致）
    t_critical = stats.t.ppf(0.975, df_resid)  # 95%置信区间的t临界值
    intercept_ci = [
        float(intercept - t_critical * intercept_se),
        float(intercept + t_critical * intercept_se)
    ]
    ci_values = {
        feat: [
            float(coefficients[feat] - t_critical * se_values[feat]),
            float(coefficients[feat] + t_critical * se_values[feat])
        ]
        for feat in features
    }

    # 计算调整R方
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # 计算F统计量
    ssr = np.sum((y_pred - np.mean(y))**2)  # 回归平方和
    sse = np.sum(residuals**2)  # 残差平方和
    f_statistic = (ssr / p) / (sse / (n - p - 1)) if (n - p - 1) > 0 and sse > 0 else 0.0
    f_p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1) if f_statistic > 0 else 1.0

    # 计算标准化回归系数（Beta）
    # Beta = B * (std(X) / std(Y))，使用原始数据的标准差（与SPSS一致）
    # SPSS使用样本标准差（ddof=1），而不是总体标准差
    y_std = np.std(y, ddof=1) if len(y) > 1 else 0.0
    standardized_coef = {}
    for i, (feat, coef) in enumerate(zip(features, model.coef_)):
        x_std = np.std(X.iloc[:, i], ddof=1) if len(X.iloc[:, i]) > 1 else 0.0
        beta = coef * x_std / y_std if y_std > 0 else 0
        standardized_coef[feat] = float(beta)

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
        # 只有一个特征时，VIF为1（无多重共线性）
        vif_values[features[0]] = 1.0

    # 计算Durbin-Watson统计量
    diff_residuals = np.diff(residuals)
    dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2)

    # 计算95%置信区间（与SPSS一致）
    t_critical = stats.t.ppf(0.975, df_resid)  # 95%置信区间的t临界值
    intercept_ci = [
        float(intercept - t_critical * intercept_se),
        float(intercept + t_critical * intercept_se)
    ]
    ci_values = {
        feat: [
            float(coefficients[feat] - t_critical * se_values[feat]),
            float(coefficients[feat] + t_critical * se_values[feat])
        ]
        for feat in features
    }

    # ========== 模型假设检验 ==========
    model_tests = {}

    # 1. 残差正态性检验（Shapiro-Wilk检验，适用于小样本；Kolmogorov-Smirnov适用于大样本）
    residuals_array = np.array(residuals)
    if n <= 5000:  # Shapiro-Wilk适用于小样本
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
            # 标准化残差
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

    # 2. 残差同方差性检验（Breusch-Pagan检验）
    if STATSMODELS_AVAILABLE and n > p + 1:
        assert sm is not None
        try:
            # 使用statsmodels进行Breusch-Pagan检验
            # 构建设计矩阵（包含截距）
            X_with_const = sm.add_constant(X_scaled)
            # 计算残差平方
            residuals_squared = residuals_array ** 2
            # 对残差平方进行回归
            bp_model = sm.OLS(residuals_squared, X_with_const).fit()
            # Breusch-Pagan统计量 = n * R²
            bp_stat = n * bp_model.rsquared
            # 自由度 = p（特征数）
            bp_p = 1 - stats.chi2.cdf(bp_stat, p)
            model_tests["homoscedasticity"] = {
                "test_name": "Breusch-Pagan",
                "statistic": float(bp_stat),
                "p_value": float(bp_p),
                "interpretation": "残差同方差性检验" if bp_p >= 0.05 else "残差可能存在异方差性"
            }
        except Exception:
            # 如果Breusch-Pagan失败，尝试简单的残差图方法
            try:
                # 计算残差与预测值的相关系数（简单检验）
                corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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
            corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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

    # 3. 残差独立性检验（Durbin-Watson检验）
    # DW统计量已经在上面计算了
    # DW值在1.5-2.5之间通常认为残差独立
    dw_interpretation = "残差独立" if 1.5 <= dw_statistic <= 2.5 else "残差可能存在自相关"
    model_tests["independence"] = {
        "test_name": "Durbin-Watson",
        "statistic": float(dw_statistic),
        "p_value": None,  # DW检验没有p值，需要查表
        "interpretation": dw_interpretation
    }

    # 4. 线性关系检验（通过残差与预测值的散点图判断，这里计算相关性）
    try:
        # 如果残差与预测值没有明显的非线性关系，相关系数应该接近0
        corr_res_pred_linear = np.corrcoef(residuals_array, y_pred)[0, 1]
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": float(abs(corr_res_pred_linear)),
            "p_value": None,
            "interpretation": f"残差与预测值相关系数: {corr_res_pred_linear:.4f}（接近0表示线性关系良好）"
        }
    except Exception:
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": None,
            "p_value": None,
            "interpretation": "无法计算"
        }

    # 稳健性检验：异方差稳健标准误(HC1)
    X_robust = X.to_numpy()
    y_robust = np.asarray(y, dtype=float)
    model_tests["robustness"] = _compute_robustness_ols(X_robust, y_robust, features, p_values)

    # 异质性检验（Chow检验）
    pooled_ssr_linear = float(np.sum(np.asarray(residuals) ** 2))
    model_tests["heterogeneity"] = _compute_heterogeneity_chow_ols(
        X_scaled, np.asarray(y, dtype=float), pooled_ssr_linear, p + 1
    )

    return {
        "model_type": "linear",
        "coefficients": coefficients,
        "intercept": intercept,
        "intercept_se": float(intercept_se),
        "intercept_t": float(intercept_t),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "r_squared": float(r2),
        "adjusted_r_squared": float(adjusted_r2),
        "rmse": float(rmse),
        "mse": float(mse),
        "p_values": p_values,
        "t_values": t_values,
        "se_values": se_values,
        "standardized_coefficients": standardized_coef,
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "vif_values": vif_values,
        "durbin_watson": float(dw_statistic),
        "n_samples": n,
        "n_features": p,
        "residuals": residuals.tolist(),  # 残差，用于残差分析
        "y_pred": y_pred.tolist(),  # 预测值，用于残差分析
        "model_tests": model_tests,  # 模型假设检验结果
    }


def fit_ridge(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 1.0,
    standardize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合岭回归模型。

    Args:
        df: 数据集 DataFrame
        target: 目标变量（因变量）列名
        features: 特征（自变量）列名列表
        alpha: 正则化强度，默认 1.0。alpha 越大，正则化越强，系数越趋向于 0
        standardize: 是否对特征进行标准化，默认 True（岭回归通常需要标准化）

    Returns:
        Dict[str, Any]: 包含 model_type、coefficients、intercept、r_squared、rmse、p_values、alpha 等
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
        y = y.fillna(y.mean())
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

    # 确保目标变量是数值型
    if not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)

    X = X_encoded

    # 保存原始数据用于计算非标准化系数（与SPSS对齐）
    X_original = X.to_numpy()
    y_original = y.to_numpy()

    # 岭回归通常需要对特征进行标准化（内部拟合使用标准化数据）
    scaler = None
    if standardize:
        # 计算原始数据的均值和标准差（用于系数转换）
        X_mean = np.mean(X_original, axis=0)
        X_std = np.std(X_original, axis=0, ddof=0)  # 使用总体标准差（与SPSS一致）
        y_mean = np.mean(y_original)
        y_std = np.std(y_original, ddof=0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 对y也进行标准化（与SPSS一致）
        y_scaled = (y_original - y_mean) / y_std if y_std > 0 else y_original
    else:
        X_scaled = X.to_numpy()
        y_scaled = y_original
        X_mean = np.zeros(len(features))
        X_std = np.ones(len(features))
        y_mean = 0.0
        y_std = 1.0
    X_scaled = np.asarray(X_scaled, dtype=float)
    y_scaled = np.asarray(y_scaled, dtype=float)

    # 拟合岭回归模型（使用标准化数据）
    model = Ridge(alpha=alpha, solver='auto')
    model.fit(X_scaled, y_scaled)

    # 预测和评估（使用标准化数据）
    y_pred_scaled = model.predict(X_scaled)
    # 将预测值转换回原始尺度
    y_pred = y_pred_scaled * y_std + y_mean if standardize else y_pred_scaled
    r2 = r2_score(y_original, y_pred)
    mse = mean_squared_error(y_original, y_pred)
    # 确保 mse 是 float 类型
    mse = float(mse) if not isinstance(mse, np.ndarray) else float(mse.item() if mse.size == 1 else mse)
    rmse = float(np.sqrt(mse))

    # 将标准化系数转换回原始尺度（非标准化系数，与SPSS对齐）
    # B_unstandardized = B_standardized * (std_y / std_x)
    coefficients = {}
    for i, (feat, coef_scaled) in enumerate(zip(features, model.coef_)):
        if standardize:
            if X_std[i] > 0:
                coef_unstandardized = coef_scaled * (y_std / X_std[i])
            else:
                coef_unstandardized = 0.0
        else:
            coef_unstandardized = coef_scaled
        coefficients[feat] = float(coef_unstandardized)

    # 截距转换
    if standardize:
        # intercept_unstandardized = intercept_standardized * std_y + mean_y - sum(coef_unstandardized * mean_x)
        intercept_scaled = model.intercept_
        intercept = float(intercept_scaled * y_std + y_mean - np.sum([coefficients[feat] * X_mean[i] for i, feat in enumerate(features)]))
    else:
        intercept = float(model.intercept_)

    # 计算统计量（使用原始数据计算残差和标准误，与SPSS对齐）
    n = len(y)
    p = len(features)
    # 使用原始数据计算预测值
    y_pred = intercept + X_original @ np.array([coefficients[feat] for feat in features])
    residuals = y_original - y_pred
    mse_residual = np.sum(residuals**2) / (n - p - 1)

    # 对于岭回归，标准误的计算需要考虑正则化项
    # 使用原始数据的设计矩阵 Z=[1, X]（与SPSS对齐）
    Z = np.column_stack([np.ones(n), X_original])
    XtX = Z.T @ Z
    # 添加正则化项（不包括截距），但需要调整正则化强度以匹配原始尺度
    # 正则化项需要根据数据尺度调整：alpha_adjusted = alpha * (n / trace(XtX))
    trace_XtX = np.trace(XtX[1:, 1:])
    if trace_XtX > 0:
        alpha_adjusted = alpha * (n / trace_XtX)
    else:
        alpha_adjusted = alpha
    ridge_matrix = XtX.copy()
    ridge_matrix[1:, 1:] += alpha_adjusted * np.eye(p)  # 只对特征部分添加正则化
    XtX_ridge_inv = np.linalg.pinv(ridge_matrix)
    cov_beta = mse_residual * XtX_ridge_inv @ XtX @ XtX_ridge_inv
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

    # 计算系数标准误差、t统计量和p值（使用非标准化系数，与SPSS对齐）
    coef_array = np.array([coefficients[feat] for feat in features])
    t_stats = coef_array / se_coef_arr
    df_resid = n - p - 1
    p_values = {
        feat: float(2 * (1 - stats.t.cdf(abs(t), df_resid)))
        for feat, t in zip(features, t_stats)
    }
    t_values = {feat: float(t) for feat, t in zip(features, t_stats)}
    se_values = {feat: float(se) for feat, se in zip(features, se_coef_arr)}

    # 截距的 t 和 p（使用非标准化截距，与SPSS对齐）
    intercept_t = intercept / intercept_se if intercept_se > 0 else 0.0
    intercept_p = float(2 * (1 - stats.t.cdf(abs(intercept_t), df_resid)))

    # 计算调整R方
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # 计算F统计量（考虑正则化）
    ssr = np.sum((y_pred - np.mean(y))**2)  # 回归平方和
    sse = np.sum(residuals**2)  # 残差平方和
    f_statistic = (ssr / p) / (sse / (n - p - 1))
    f_p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1)

    # 计算标准化回归系数（Beta）
    # Beta = B * (std(X) / std(Y))，使用原始数据的样本标准差（与SPSS一致）
    y_std = np.std(y_original, ddof=1) if len(y_original) > 1 else 0.0
    standardized_coef = {}
    for i, (feat, coef) in enumerate(zip(features, coefficients.values())):
        x_std = np.std(X_original[:, i], ddof=1) if len(X_original[:, i]) > 1 else 0.0
        beta = coef * x_std / y_std if y_std > 0 else 0
        standardized_coef[feat] = float(beta)

    # 计算VIF（方差膨胀因子）
    # 注意：岭回归通过正则化已经处理了多重共线性，VIF 的计算可能不太适用
    # 但为了保持结果格式一致，仍然计算
    vif_values = {}
    if len(features) > 1:
        for i, feat in enumerate(features):
            # 将其他特征作为自变量，当前特征作为因变量
            X_other = np.delete(X_scaled, i, axis=1)
            y_feat = X_scaled[:, i]
            model_vif = Ridge(alpha=alpha)  # 使用相同的 alpha
            model_vif.fit(X_other, y_feat)
            r2_vif = r2_score(y_feat, model_vif.predict(X_other))
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            vif_values[feat] = float(vif)
    else:
        # 只有一个特征时，VIF为1（无多重共线性）
        vif_values[features[0]] = 1.0

    # 计算Durbin-Watson统计量
    diff_residuals = np.diff(residuals)
    dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 0.0

    # 计算95%置信区间（与SPSS一致）
    t_critical = stats.t.ppf(0.975, df_resid)  # 95%置信区间的t临界值
    intercept_ci = [
        float(intercept - t_critical * intercept_se),
        float(intercept + t_critical * intercept_se)
    ]
    ci_values = {
        feat: [
            float(coefficients[feat] - t_critical * se_values[feat]),
            float(coefficients[feat] + t_critical * se_values[feat])
        ]
        for feat in features
    }

    # ========== 模型假设检验（与线性回归相同）==========
    model_tests = {}

    # 1. 残差正态性检验（Shapiro-Wilk检验，适用于小样本；Kolmogorov-Smirnov适用于大样本）
    residuals_array = np.array(residuals)
    if n <= 5000:  # Shapiro-Wilk适用于小样本
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
            # 标准化残差
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

    # 2. 残差同方差性检验（Breusch-Pagan检验）
    if STATSMODELS_AVAILABLE and n > p + 1:
        assert sm is not None
        try:
            # 使用statsmodels进行Breusch-Pagan检验
            # 构建设计矩阵（包含截距）
            X_with_const = sm.add_constant(X_scaled)
            # 计算残差平方
            residuals_squared = residuals_array ** 2
            # 对残差平方进行回归
            bp_model = sm.OLS(residuals_squared, X_with_const).fit()
            # Breusch-Pagan统计量 = n * R²
            bp_stat = n * bp_model.rsquared
            # 自由度 = p（特征数）
            bp_p = 1 - stats.chi2.cdf(bp_stat, p)
            model_tests["homoscedasticity"] = {
                "test_name": "Breusch-Pagan",
                "statistic": float(bp_stat),
                "p_value": float(bp_p),
                "interpretation": "残差同方差性检验" if bp_p >= 0.05 else "残差可能存在异方差性"
            }
        except Exception:
            # 如果Breusch-Pagan失败，尝试简单的残差图方法
            try:
                # 计算残差与预测值的相关系数（简单检验）
                corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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
        # 如果statsmodels不可用或样本量太小，使用简单的相关性检验
        try:
            corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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

    # 3. 残差独立性检验（Durbin-Watson检验）
    # DW统计量已经在上面计算了
    # DW值在1.5-2.5之间通常认为残差独立
    dw_interpretation = "残差独立" if 1.5 <= dw_statistic <= 2.5 else "残差可能存在自相关"
    model_tests["independence"] = {
        "test_name": "Durbin-Watson",
        "statistic": float(dw_statistic),
        "p_value": None,  # DW检验没有p值，需要查表
        "interpretation": dw_interpretation
    }

    # 4. 线性关系检验（通过残差与预测值的散点图判断，这里计算相关性）
    try:
        # 如果残差与预测值没有明显的非线性关系，相关系数应该接近0
        corr_res_pred_linear = np.corrcoef(residuals_array, y_pred)[0, 1]
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": float(abs(corr_res_pred_linear)),
            "p_value": None,
            "interpretation": f"残差与预测值相关系数: {corr_res_pred_linear:.4f}（接近0表示线性关系良好）"
        }
    except Exception:
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": None,
            "p_value": None,
            "interpretation": "无法计算"
        }

    # 稳健性检验：异方差稳健标准误(HC1)
    model_tests["robustness"] = _compute_robustness_ols(X_original, y_original, features, p_values)

    # 异质性检验（Chow检验）
    pooled_ssr_ridge = float(np.sum((y_original - y_pred) ** 2))
    model_tests["heterogeneity"] = _compute_heterogeneity_chow_ridge(
        X_original, y_original, features, alpha, standardize, scaler, pooled_ssr_ridge, p + 1
    )

    return {
        "model_type": "ridge",
        "alpha": float(alpha),
        "coefficients": coefficients,
        "intercept": intercept,
        "intercept_se": float(intercept_se),
        "intercept_t": float(intercept_t),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "r_squared": float(r2),
        "adjusted_r_squared": float(adjusted_r2),
        "rmse": float(rmse),
        "mse": float(mse),
        "p_values": p_values,
        "t_values": t_values,
        "se_values": se_values,
        "standardized_coefficients": standardized_coef,
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "vif_values": vif_values,
        "durbin_watson": float(dw_statistic),
        "n_samples": n,
        "n_features": p,
        "residuals": residuals.tolist(),  # 残差，用于残差分析
        "y_pred": y_pred.tolist(),  # 预测值，用于残差分析
        "model_tests": model_tests,  # 模型假设检验结果
    }


def fit_lasso(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    alpha: float = 1.0,
    standardize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合 Lasso 回归模型。

    Args:
        df: 数据集 DataFrame
        target: 目标变量（因变量）列名
        features: 特征（自变量）列名列表
        alpha: 正则化强度，默认 1.0。alpha 越大，正则化越强，更多系数趋向于 0
        standardize: 是否对特征进行标准化，默认 True（Lasso 回归通常需要标准化）

    Returns:
        Dict[str, Any]: 包含 model_type、coefficients、intercept、r_squared、rmse、p_values、alpha 等
    """
    X: pd.DataFrame = pd.DataFrame(df[features].copy())
    y: pd.Series = pd.Series(df[target].copy())

    # 处理缺失值：数值列用均值，分类列用众数或False（布尔类型）
    for col in X.columns:
        col_series = cast(pd.Series, X[col])
        if pd.api.types.is_numeric_dtype(col_series):
            # 数值列用均值填充
            mean_val = float(col_series.mean())
            if pd.isna(mean_val) or not np.isfinite(mean_val):
                # 如果均值无效（全为NaN），用0填充
                X[col] = col_series.fillna(0.0)
            else:
                X[col] = col_series.fillna(mean_val)
            # 确保是浮点型
            X[col] = cast(pd.Series, pd.to_numeric(X[col], errors='coerce')).fillna(0.0)
        else:
            # 分类变量：检查是否为布尔类型
            if col_series.dtype == bool:
                # 布尔类型：用False填充，然后转换为0/1
                X[col] = col_series.fillna(False).astype(bool).astype(int)
            else:
                # 其他分类变量用众数填充
                mode_val = col_series.mode()
                if len(mode_val) > 0:
                    X[col] = col_series.fillna(mode_val.iloc[0])
                else:
                    fallback = col_series.iloc[0] if len(col_series) > 0 else ""
                    X[col] = col_series.fillna(fallback)

    # 处理目标变量的缺失值
    if pd.api.types.is_numeric_dtype(y):
        mean_val = float(y.mean())
        if pd.isna(mean_val) or not np.isfinite(mean_val):
            y = y.fillna(0.0)
        else:
            y = y.fillna(mean_val)
        y = cast(pd.Series, pd.to_numeric(y, errors='coerce')).fillna(0.0)
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

    # 确保目标变量是数值型
    if not pd.api.types.is_numeric_dtype(y):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)

    X = X_encoded

    # 确保所有值都是有效的数值（最终检查）
    for col in X.columns:
        X[col] = cast(pd.Series, pd.to_numeric(X[col], errors='coerce')).fillna(0.0)
    y = cast(pd.Series, pd.to_numeric(y, errors='coerce')).fillna(0.0)

    # 保存原始数据用于计算非标准化系数（与SPSS对齐）
    X_original = X.to_numpy()
    y_original = y.to_numpy()

    # 最终确保 numpy 数组中没有 NaN 或 Inf
    X_original = np.nan_to_num(X_original, nan=0.0, posinf=0.0, neginf=0.0)
    y_original = np.nan_to_num(y_original, nan=0.0, posinf=0.0, neginf=0.0)

    # Lasso 回归通常需要对特征进行标准化（内部拟合使用标准化数据）
    scaler = None
    if standardize:
        # 计算原始数据的均值和标准差（用于系数转换）
        X_mean = np.mean(X_original, axis=0)
        X_std = np.std(X_original, axis=0, ddof=0)  # 使用总体标准差（与SPSS一致）
        y_mean = np.mean(y_original)
        y_std = np.std(y_original, ddof=0)

        # 确保标准差不为0，避免除零错误
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
        y_std = y_std if y_std > 1e-10 else 1.0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 对y也进行标准化（与SPSS一致）
        if y_std > 1e-10:  # 避免除以接近0的值
            y_scaled = (y_original - y_mean) / y_std
        else:
            y_scaled = y_original - y_mean  # 只减去均值，不除以标准差
    else:
        X_scaled = X.to_numpy()
        y_scaled = y_original
        X_mean = np.zeros(len(features))
        X_std = np.ones(len(features))
        y_mean = 0.0
        y_std = 1.0
    X_scaled = np.asarray(X_scaled, dtype=float)
    y_scaled = np.asarray(y_scaled, dtype=float)

    # 确保标准化后的数据也没有 NaN 或 Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # 拟合 Lasso 回归模型（使用标准化数据）
    model = Lasso(alpha=alpha, max_iter=2000)
    model.fit(X_scaled, y_scaled)

    # 预测和评估（使用标准化数据）
    y_pred_scaled = model.predict(X_scaled)
    # 将预测值转换回原始尺度
    y_pred = y_pred_scaled * y_std + y_mean if standardize else y_pred_scaled
    r2 = r2_score(y_original, y_pred)
    mse = mean_squared_error(y_original, y_pred)
    # 确保 mse 是 float 类型
    mse = float(mse) if not isinstance(mse, np.ndarray) else float(mse.item() if mse.size == 1 else mse)
    rmse = float(np.sqrt(mse))

    # 将标准化系数转换回原始尺度（非标准化系数，与SPSS对齐）
    # B_unstandardized = B_standardized * (std_y / std_x)
    coefficients = {}
    for i, (feat, coef_scaled) in enumerate(zip(features, model.coef_)):
        if standardize:
            if X_std[i] > 0:
                coef_unstandardized = coef_scaled * (y_std / X_std[i])
            else:
                coef_unstandardized = 0.0
        else:
            coef_unstandardized = coef_scaled
        coefficients[feat] = float(coef_unstandardized)

    # 截距转换
    if standardize:
        # intercept_unstandardized = intercept_standardized * std_y + mean_y - sum(coef_unstandardized * mean_x)
        intercept_scaled = model.intercept_
        intercept = float(intercept_scaled * y_std + y_mean - np.sum([coefficients[feat] * X_mean[i] for i, feat in enumerate(features)]))
    else:
        intercept = float(model.intercept_)

    # 计算统计量（使用原始数据计算残差和标准误，与SPSS对齐）
    n = len(y)
    p = len(features)
    # 使用原始数据计算预测值
    y_pred = intercept + X_original @ np.array([coefficients[feat] for feat in features])
    residuals = y_original - y_pred
    # 计算 mse_residual，确保不为负数或 NaN
    if (n - p - 1) > 0:
        mse_residual = float(np.sum(residuals**2) / (n - p - 1))
        # 确保不为负数或 NaN
        if mse_residual < 0 or not np.isfinite(mse_residual):
            mse_residual = float(mse) if np.isfinite(mse) and mse > 0 else 1.0
    else:
        mse_residual = float(mse) if np.isfinite(mse) and mse > 0 else 1.0
    # 确保 mse_residual 是正数
    mse_residual = max(mse_residual, 1e-10)

    # 对于 Lasso 回归，标准误的计算需要考虑正则化项
    # 使用原始数据的设计矩阵 Z=[1, X]（与SPSS对齐）
    Z = np.column_stack([np.ones(n), X_original])
    XtX = Z.T @ Z
    # 对于非零系数，使用近似协方差矩阵；对于零系数，标准误为 0
    # 正则化项需要根据数据尺度调整
    trace_XtX = np.trace(XtX[1:, 1:])
    if trace_XtX > 0:
        alpha_adjusted = alpha * (n / trace_XtX)
    else:
        alpha_adjusted = alpha
    lasso_matrix = XtX.copy()
    # 对于非零系数，添加正则化项；对于零系数，使用大值近似
    for i in range(p):
        if abs(coefficients[features[i]]) < 1e-10:
            lasso_matrix[i+1, i+1] += 1e10  # 近似无穷大，使逆矩阵中对应元素接近0
        else:
            lasso_matrix[i+1, i+1] += alpha_adjusted
    # 计算协方差矩阵（用于标准误）
    try:
        XtX_lasso_inv = np.linalg.pinv(lasso_matrix)
        # 确保所有矩阵都是 numpy 数组
        XtX_lasso_inv = np.array(XtX_lasso_inv, dtype=np.float64)
        XtX = np.array(XtX, dtype=np.float64)
        mse_residual = float(mse_residual)

        # 计算协方差矩阵
        cov_beta = mse_residual * XtX_lasso_inv @ XtX @ XtX_lasso_inv
        # 确保 cov_beta 是 numpy 数组（使用 np.array 确保是真正的 ndarray）
        cov_beta = np.array(cov_beta, dtype=np.float64, copy=False)
        # 确保 cov_beta 是 2D 数组
        if cov_beta.ndim == 0:
            cov_beta = np.array([[float(cov_beta)]], dtype=np.float64)
        elif cov_beta.ndim == 1:
            # 1D 数组，转换为方阵
            size = len(cov_beta)
            cov_beta = np.diag(cov_beta)
        elif cov_beta.ndim > 2:
            cov_beta = cov_beta.reshape(cov_beta.shape[0], -1)
        # 现在提取对角线元素
        diag_values = np.diag(cov_beta)
    except Exception as e:
        # 如果协方差矩阵计算失败，使用简化的方法：只对非零系数计算标准误
        # 对于零系数，标准误设为 0
        diag_values = np.zeros(p + 1, dtype=np.float64)
        # 对于非零系数，使用简化的标准误估计
        for i, feat in enumerate(features):
            if abs(coefficients[feat]) > 1e-10:
                # 使用简化的标准误：基于残差的标准差
                diag_values[i + 1] = max(mse_residual, 1e-10)
        # 截距的标准误
        diag_values[0] = max(mse_residual, 1e-10)
    # 使用 np.atleast_1d 确保至少是 1D 数组（即使是标量也会转换为数组）
    diag_values = np.atleast_1d(diag_values)
    # 明确转换为 float64 类型的数组
    diag_values = np.array(diag_values, dtype=np.float64, copy=False)
    # 确保是 1D 数组
    diag_values = diag_values.flatten()
    # 确保所有值都是非负的（方差不能为负）
    diag_values = np.maximum(diag_values, 0.0)
    # 过滤无效值（NaN 和 Inf），替换为 0
    diag_values = np.nan_to_num(diag_values, nan=0.0, posinf=0.0, neginf=0.0)
    # 确保 diag_values 有足够的元素（至少 p+1 个，包括截距）
    expected_size = p + 1
    if len(diag_values) < expected_size:
        # 如果元素不足，用 0 填充
        diag_values = np.pad(diag_values, (0, expected_size - len(diag_values)), mode='constant', constant_values=0.0)
    elif len(diag_values) > expected_size:
        # 如果元素太多，截取前 expected_size 个
        diag_values = diag_values[:expected_size]
    # 最终类型和有效性检查：确保是 NumPy float64 数组，且所有值都有效
    diag_values = np.array(diag_values, dtype=np.float64)
    diag_values = np.maximum(diag_values, 0.0)  # 再次确保非负
    diag_values = np.nan_to_num(diag_values, nan=0.0, posinf=0.0, neginf=0.0)  # 再次清理无效值

    # 使用最安全的方法计算平方根：确保类型正确后再计算
    # 最终确保 diag_values 是 numpy float64 数组
    if not isinstance(diag_values, np.ndarray):
        diag_values = np.array(diag_values, dtype=np.float64)
    else:
        diag_values = diag_values.astype(np.float64)

    # 确保所有值都是有效的非负数
    diag_values = np.maximum(diag_values, 0.0)
    diag_values = np.nan_to_num(diag_values, nan=0.0, posinf=0.0, neginf=0.0)

    # 现在安全地计算平方根
    # 使用 np.sqrt 对整个数组进行计算（这是最标准的方法）
    se_all = np.sqrt(diag_values)

    # 确保 se_all 也是正确的类型
    if not isinstance(se_all, np.ndarray):
        se_all = np.array(se_all, dtype=np.float64)

    # 截距在第 0 个位置，其余是各自变量
    intercept_se = float(se_all[0])
    se_coef_arr = se_all[1:].copy()  # 使用 copy() 确保是独立的数组

    # 对于被 Lasso 压缩到 0 的系数，标准误和 t 值设为 0
    for i, feat in enumerate(features):
        if i < len(se_coef_arr) and abs(coefficients[feat]) < 1e-10:  # 系数接近 0
            se_coef_arr[i] = 0.0

    # 计算系数标准误差、t统计量和p值（使用非标准化系数，与SPSS对齐）
    coef_array = np.array([coefficients[feat] for feat in features])
    t_stats = np.zeros_like(coef_array)
    for i, (coef, se) in enumerate(zip(coef_array, se_coef_arr)):
        if se > 1e-10:  # 只有当标准误不为 0 时才计算 t 值
            t_stats[i] = coef / se
        else:
            t_stats[i] = 0.0

    df_resid = n - p - 1 if (n - p - 1) > 0 else 1
    p_values = {
        feat: float(2 * (1 - stats.t.cdf(abs(t), df_resid))) if abs(t) > 1e-10 else 1.0
        for feat, t in zip(features, t_stats)
    }
    t_values = {feat: float(t) for feat, t in zip(features, t_stats)}
    se_values = {feat: float(se) for feat, se in zip(features, se_coef_arr)}

    # 截距的 t 和 p
    intercept_t = model.intercept_ / intercept_se if intercept_se > 0 else 0.0
    intercept_p = float(2 * (1 - stats.t.cdf(abs(intercept_t), df_resid))) if abs(intercept_t) > 1e-10 else 1.0

    # 计算调整R方
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2

    # 计算F统计量（考虑正则化）
    ssr = np.sum((y_pred - np.mean(y))**2)  # 回归平方和
    sse = np.sum(residuals**2)  # 残差平方和
    f_statistic = (ssr / p) / (sse / (n - p - 1)) if (n - p - 1) > 0 and sse > 0 else 0.0
    f_p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1) if f_statistic > 0 else 1.0

    # 计算标准化回归系数（Beta）
    # Beta = B * (std(X) / std(Y))，使用原始数据的样本标准差（与SPSS一致）
    y_std = np.std(y_original, ddof=1) if len(y_original) > 1 else 0.0
    standardized_coef = {}
    for i, (feat, coef) in enumerate(zip(features, coefficients.values())):
        x_std = np.std(X_original[:, i], ddof=1) if len(X_original[:, i]) > 1 else 0.0
        beta = coef * x_std / y_std if y_std > 0 else 0
        standardized_coef[feat] = float(beta)

    # 计算VIF（方差膨胀因子）
    # 注意：Lasso 回归通过正则化已经处理了多重共线性，VIF 的计算可能不太适用
    # 但为了完整性，仍然计算（使用非正则化的线性回归）
    vif_values = {}
    if len(features) > 1:
        for i, feat in enumerate(features):
            # 将其他特征作为自变量，当前特征作为因变量
            X_other = np.delete(X_scaled, i, axis=1)
            y_feat = X_scaled[:, i]
            # 对于 Lasso，使用简化的线性回归计算 VIF
            model_vif = LinearRegression()
            model_vif.fit(X_other, y_feat)
            r2_vif = r2_score(y_feat, model_vif.predict(X_other))
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            vif_values[feat] = float(vif)
    else:
        # 只有一个特征时，VIF为1（无多重共线性）
        vif_values[features[0]] = 1.0

    # 计算Durbin-Watson统计量
    diff_residuals = np.diff(residuals)
    dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 0.0

    # 统计被压缩到 0 的特征数量（特征选择效果）
    n_selected_features = sum(1 for coef in model.coef_ if abs(coef) > 1e-10)
    n_zero_features = p - n_selected_features

    # 计算95%置信区间（与SPSS一致）
    t_critical = stats.t.ppf(0.975, df_resid)  # 95%置信区间的t临界值
    intercept_ci = [
        float(intercept - t_critical * intercept_se),
        float(intercept + t_critical * intercept_se)
    ]
    ci_values = {
        feat: [
            float(coefficients[feat] - t_critical * se_values[feat]),
            float(coefficients[feat] + t_critical * se_values[feat])
        ]
        for feat in features
    }

    # ========== 模型假设检验（与线性回归相同）==========
    model_tests = {}

    # 1. 残差正态性检验（Shapiro-Wilk检验，适用于小样本；Kolmogorov-Smirnov适用于大样本）
    residuals_array = np.array(residuals)
    if n <= 5000:  # Shapiro-Wilk适用于小样本
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
            # 标准化残差
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

    # 2. 残差同方差性检验（Breusch-Pagan检验）
    if STATSMODELS_AVAILABLE and n > p + 1:
        assert sm is not None
        try:
            # 使用statsmodels进行Breusch-Pagan检验
            # 构建设计矩阵（包含截距）
            X_with_const = sm.add_constant(X_scaled)
            # 计算残差平方
            residuals_squared = residuals_array ** 2
            # 对残差平方进行回归
            bp_model = sm.OLS(residuals_squared, X_with_const).fit()
            # Breusch-Pagan统计量 = n * R²
            bp_stat = n * bp_model.rsquared
            # 自由度 = p（特征数）
            bp_p = 1 - stats.chi2.cdf(bp_stat, p)
            model_tests["homoscedasticity"] = {
                "test_name": "Breusch-Pagan",
                "statistic": float(bp_stat),
                "p_value": float(bp_p),
                "interpretation": "残差同方差性检验" if bp_p >= 0.05 else "残差可能存在异方差性"
            }
        except Exception:
            # 如果Breusch-Pagan失败，尝试简单的残差图方法
            try:
                # 计算残差与预测值的相关系数（简单检验）
                corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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
        # 如果statsmodels不可用或样本量太小，使用简单的相关性检验
        try:
            corr_res_pred = np.corrcoef(residuals_array, y_pred)[0, 1]
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

    # 3. 残差独立性检验（Durbin-Watson检验）
    # DW统计量已经在上面计算了
    # DW值在1.5-2.5之间通常认为残差独立
    dw_interpretation = "残差独立" if 1.5 <= dw_statistic <= 2.5 else "残差可能存在自相关"
    model_tests["independence"] = {
        "test_name": "Durbin-Watson",
        "statistic": float(dw_statistic),
        "p_value": None,  # DW检验没有p值，需要查表
        "interpretation": dw_interpretation
    }

    # 4. 线性关系检验（通过残差与预测值的散点图判断，这里计算相关性）
    try:
        # 如果残差与预测值没有明显的非线性关系，相关系数应该接近0
        corr_res_pred_linear = np.corrcoef(residuals_array, y_pred)[0, 1]
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": float(abs(corr_res_pred_linear)),
            "p_value": None,
            "interpretation": f"残差与预测值相关系数: {corr_res_pred_linear:.4f}（接近0表示线性关系良好）"
        }
    except Exception:
        model_tests["linearity"] = {
            "test_name": "残差-预测值相关性",
            "statistic": None,
            "p_value": None,
            "interpretation": "无法计算"
        }

    # 稳健性检验：异方差稳健标准误(HC1)
    model_tests["robustness"] = _compute_robustness_ols(X_original, y_original, features, p_values)

    return {
        "model_type": "lasso",
        "alpha": float(alpha),
        "coefficients": coefficients,
        "intercept": intercept,
        "intercept_se": float(intercept_se),
        "intercept_t": float(intercept_t),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "r_squared": float(r2),
        "adjusted_r_squared": float(adjusted_r2),
        "rmse": float(rmse),
        "mse": float(mse),
        "p_values": p_values,
        "t_values": t_values,
        "se_values": se_values,
        "standardized_coefficients": standardized_coef,
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "vif_values": vif_values,
        "durbin_watson": float(dw_statistic),
        "n_samples": n,
        "n_features": p,
        "n_selected_features": n_selected_features,  # Lasso 特有的：被选中的特征数
        "n_zero_features": n_zero_features,  # Lasso 特有的：被压缩到 0 的特征数
        "residuals": residuals.tolist(),  # 残差，用于残差分析
        "y_pred": y_pred.tolist(),  # 预测值，用于残差分析
        "model_tests": model_tests,  # 模型假设检验结果
    }
