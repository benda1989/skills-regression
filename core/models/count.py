"""Poisson and Negative Binomial regression standalone functions."""
from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ._helpers import _compute_heterogeneity_lr_glm


def fit_poisson(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    standardize: bool = True,
    max_iter: int = 100,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合泊松回归模型。

    泊松回归用于建模计数数据（非负整数）。使用statsmodels的Poisson回归实现。

    Args:
        df: 数据框
        target: 目标变量（因变量）列名，必须是计数数据（非负整数）
        features: 特征（自变量）列名列表
        standardize: 是否对特征进行标准化，默认 True
        max_iter: 最大迭代次数，默认 100

    Returns:
        Dict[str, Any]: 包含模型类型、系数、统计量等的字典
    """
    # 运行时动态检查 statsmodels 是否可用
    try:
        import statsmodels.api as sm
        # 设置使用基于对数似然的 BIC，避免 FutureWarning
        try:
            from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
            SET_USE_BIC_LLF(True)
        except (ImportError, AttributeError):
            # 如果设置失败，继续使用默认方法（会有警告但功能正常）
            pass
    except ImportError:
        raise ImportError("statsmodels 未安装，无法使用泊松回归。请运行: pip install statsmodels")

    X: pd.DataFrame = pd.DataFrame(df[features].copy())
    y: pd.Series = pd.Series(df[target].copy())

    # 处理缺失值：数值列用均值，分类列用众数
    for col in X.columns:
        col_series = cast(pd.Series, X[col])
        if pd.api.types.is_numeric_dtype(col_series):
            X[col] = col_series.fillna(col_series.mean())
        else:
            mode_val = col_series.mode()
            if len(mode_val) > 0:
                X[col] = col_series.fillna(mode_val.iloc[0])
            else:
                fallback = col_series.iloc[0] if len(col_series) > 0 else ""
                X[col] = col_series.fillna(fallback)

    # 处理目标变量的缺失值
    if pd.api.types.is_numeric_dtype(y):
        y = cast(pd.Series, y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]))
    else:
        mode_val = y.mode()
        if len(mode_val) > 0:
            y = cast(pd.Series, y.fillna(mode_val.iloc[0]))

    # 确保目标变量是非负整数（泊松回归要求）
    y = cast(pd.Series, pd.to_numeric(y, errors='coerce')).fillna(0)
    y = y.astype(int)
    if (y < 0).any():
        raise ValueError("泊松回归要求目标变量必须是非负整数，但发现负值")

    # 编码分类变量为数值
    label_encoders: Dict[str, LabelEncoder] = {}
    X_encoded = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    X = X_encoded

    # 保存原始数据用于计算非标准化系数
    X_original = X.to_numpy()
    y_original = y.to_numpy()

    # 计算原始数据的均值和标准差（用于系数转换）
    X_mean = np.mean(X_original, axis=0)
    X_std = np.std(X_original, axis=0, ddof=0)

    # 标准化特征（如果要求）
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.to_numpy()
    X_scaled = np.asarray(X_scaled, dtype=float)

    # 添加常数项（截距）
    X_with_const = sm.add_constant(X_scaled)

    # 拟合泊松回归模型
    model = sm.GLM(y_original, X_with_const, family=sm.families.Poisson())
    result = model.fit(maxiter=max_iter, method='bfgs')

    # 获取系数和统计量
    params = result.params
    bse = result.bse  # 标准误
    pvalues = result.pvalues
    conf_int = result.conf_int()  # 95%置信区间

    # 提取截距和系数
    intercept = float(params[0])
    coefficients = {feat: float(params[i + 1]) for i, feat in enumerate(features)}

    # 标准误
    intercept_se = float(bse[0])
    se_values = {feat: float(bse[i + 1]) for i, feat in enumerate(features)}

    # p值
    intercept_p = float(pvalues[0])
    p_values = {feat: float(pvalues[i + 1]) for i, feat in enumerate(features)}

    # 置信区间
    # conf_int 可能是 DataFrame 或 numpy 数组，需要统一处理
    if hasattr(conf_int, 'iloc'):
        # 如果是 DataFrame
        intercept_ci = [float(conf_int.iloc[0, 0]), float(conf_int.iloc[0, 1])]
        ci_values = {
            feat: [float(conf_int.iloc[i + 1, 0]), float(conf_int.iloc[i + 1, 1])]
            for i, feat in enumerate(features)
        }
    else:
        # 如果是 numpy 数组
        conf_int_array = np.asarray(conf_int)
        intercept_ci = [float(conf_int_array[0, 0]), float(conf_int_array[0, 1])]
        ci_values = {
            feat: [float(conf_int_array[i + 1, 0]), float(conf_int_array[i + 1, 1])]
            for i, feat in enumerate(features)
        }

    # 将标准化系数转换回原始尺度（非标准化系数，与SPSS对齐）
    # 对于泊松回归，系数转换：B_unstandardized = B_standardized / std_x
    coefficients_unstandardized = {}
    if standardize:
        for i, feat in enumerate(features):
            if X_std[i] > 0:
                coef_unstandardized = coefficients[feat] / X_std[i]
            else:
                coef_unstandardized = 0.0
            coefficients_unstandardized[feat] = float(coef_unstandardized)

        # 截距转换：intercept_unstandardized = intercept_standardized - sum(coef_unstandardized * mean_x)
        intercept_unstandardized = float(intercept - np.sum([coefficients_unstandardized[feat] * X_mean[i]
                                                           for i, feat in enumerate(features)]))
    else:
        coefficients_unstandardized = coefficients.copy()
        intercept_unstandardized = intercept

    # Z统计量（泊松回归使用Z统计量而不是t统计量，与SPSS一致）
    z_values = {feat: float(coefficients_unstandardized[feat] / se_values[feat]) if se_values[feat] > 0 else 0.0
               for feat in features}
    intercept_z = intercept_unstandardized / intercept_se if intercept_se > 0 else 0.0

    # 计算发生率比（IRR = exp(coefficient)）
    irr_values = {feat: float(np.exp(coef)) for feat, coef in coefficients_unstandardized.items()}
    intercept_irr = float(np.exp(intercept_unstandardized))

    # 预测值
    y_pred = result.fittedvalues

    # 计算模型拟合优度指标
    # 对数似然
    llf = float(result.llf)  # 对数似然
    ll_null = float(result.llnull)  # 零模型的对数似然

    # 伪R²（类似逻辑回归）
    # McFadden's R²
    mcfadden_r2 = 1 - (llf / ll_null) if ll_null != 0 else 0.0

    # Cox & Snell R²
    n = len(y_original)
    cox_snell_r2 = 1 - np.exp(2 * (ll_null - llf) / n)

    # Nagelkerke R²
    max_r2 = 1 - np.exp(2 * ll_null / n)
    nagelkerke_r2 = cox_snell_r2 / max_r2 if max_r2 > 0 else 0.0

    # AIC和BIC
    aic = float(result.aic)
    # 使用基于对数似然的 BIC（避免 FutureWarning）
    try:
        bic = float(result.bic_llf) if hasattr(result, 'bic_llf') else float(result.bic)
    except (AttributeError, ValueError):
        bic = float(result.bic)

    # 残差统计
    residuals = y_original - y_pred
    deviance = float(result.deviance)  # 偏差
    pearson_chi2 = float(result.pearson_chi2)  # Pearson卡方统计量

    # 计算VIF（方差膨胀因子）
    vif_values = {}
    if len(features) > 1:
        for i, feat in enumerate(features):
            X_other = np.delete(X_scaled, i, axis=1)
            y_feat = X_scaled[:, i]
            model_vif = LinearRegression()
            model_vif.fit(X_other, y_feat)
            r2_vif = r2_score(y_feat, model_vif.predict(X_other))
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            vif_values[feat] = float(vif)
    else:
        vif_values[features[0]] = 1.0

    # 计算IRR的95%置信区间（与SPSS一致）
    intercept_irr_ci = [
        float(np.exp(intercept_ci[0])),
        float(np.exp(intercept_ci[1]))
    ]
    irr_ci_values = {
        feat: [
            float(np.exp(ci_values[feat][0])),
            float(np.exp(ci_values[feat][1]))
        ]
        for feat in features
    }

    # 稳健性检验：异方差稳健标准误(HC0)
    robustness: Dict[str, Any] = {
        "method": "异方差稳健标准误(HC0)",
        "description": "为对泊松回归估计结果的稳健性进行检验，本文采用异方差稳健标准误(HC0)对模型重新估计。",
        "robust_se": {},
        "robust_p_values": {},
        "intercept_robust_se": None,
        "intercept_robust_p": None,
        "conclusion_consistent": True,
        "interpretation": "",
        "interpretation_detail": [],
    }
    try:
        robust_result = result.get_robustcov_results(cov_type="HC0")
        bse_r = robust_result.bse
        pvals_r = robust_result.pvalues
        robustness["intercept_robust_se"] = float(bse_r[0])
        robustness["intercept_robust_p"] = float(pvals_r[0])
        for i, feat in enumerate(features):
            if i + 1 < len(bse_r):
                robustness["robust_se"][feat] = float(bse_r[i + 1])
                robustness["robust_p_values"][feat] = float(pvals_r[i + 1])
        main_sig = {k: (v is not None and v < 0.05) for k, v in p_values.items()}
        robust_sig = {k: (robustness["robust_p_values"].get(k) or 1.0) < 0.05 for k in robustness["robust_p_values"]}
        robustness["conclusion_consistent"] = all(
            main_sig.get(k) == robust_sig.get(k) for k in set(main_sig) | set(robust_sig)
        )
        if robustness["conclusion_consistent"]:
            robustness["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC0)重新估计后，各变量的系数及显著性均与主回归结果一致，未发现明显变化，说明本文的实证结果是稳健的。"
            robustness["interpretation_detail"] = [
                "为考察泊松回归估计结果是否对标准误的设定敏感、是否具有稳健性，本文进行稳健性检验。计数数据常出现过度离散（方差大于均值），此时常规泊松标准误可能偏小，导致 Z 检验与 p 值过于乐观。本文采用异方差稳健标准误（HC0）对同一模型与样本重新估计，在不对方差结构做具体假定的前提下给出一致的标准误估计，使 Z 检验与 p 值在违反等离散假设时仍可靠。",
                "在稳健性检验中，本文基于稳健协方差矩阵重新计算各系数（含截距）的标准误与 p 值，并与主回归结果逐变量比较其在 0.05 显著性水平上是否一致。",
                "检验结果表明：在采用稳健标准误后，各变量的系数及显著性均没有明显变化；主回归中在 0.05 水平上显著的自变量及其发生率比(IRR)解释在稳健标准误下仍然显著，主回归中不显著的变量在稳健标准误下仍不显著。核心解释变量系数的符号与统计显著性均与主结果一致，表明泊松回归的结论不依赖于等离散假设，具备较好的稳健性。",
                "上述结果说明，泊松回归的估计结果在 0.05 的显著性水平上具有稳健性，并不随着标准误估计方法的变化而显著变化。因此，本文的实证结论具有较好的可信度与稳健性，主回归结果可作为全文结论与政策建议的主要依据，具有较好的内部效度。",
                "需要说明的是，本文的稳健性检验仅针对标准误估计方法这一维度。若存在明显过度离散，可进一步考虑负二项回归或报告负二项回归的稳健性检验，以更全面地增强结论的可信度。",
            ]
        else:
            robustness["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC0)重新估计后，部分变量的显著性结论与主结果存在差异，建议以稳健标准误下的估计与检验结果为准。"
            robustness["interpretation_detail"] = [
                "为考察泊松回归估计结果是否对标准误的设定敏感，本文进行稳健性检验。本文采用异方差稳健标准误（HC0）对同一模型与样本重新估计；当存在过度离散或异方差时，常规泊松标准误可能偏小，HC0 可给出一致的标准误估计。在稳健性检验中，本文逐变量比较主回归与稳健标准误下在 0.05 显著性水平上的结论是否一致。",
                "检验结果表明，存在至少一个变量在\"主回归结果\"与\"稳健标准误结果\"下的显著性结论不一致。这说明该变量在稳健标准误下的不确定性与主结果有所不同，其标准误或 Z 检验对等离散假设的违背较为敏感。",
                "在报告与解读时，应以稳健标准误下的系数、发生率比(IRR)及 p 值为准进行结论撰写。若过度离散明显，可考虑改用负二项回归并报告其稳健性检验，以充分验证实证结论的可信度与稳健性；并在正文中说明\"经稳健标准误检验\"，以增强结论的严谨性。",
                "稳健性检验的结果有助于识别对标准误设定敏感的变量，使研究结论更加审慎、更符合计数数据模型的计量规范。",
            ]
    except Exception as e:
        robustness["conclusion_consistent"] = False
        robustness["interpretation"] = f"稳健性检验计算失败: {str(e)}"
        robustness["interpretation_detail"] = []
    model_tests_poisson: Dict[str, Any] = {"robustness": robustness}
    # 异质性检验（似然比，分组系数稳定性）
    exog_poisson = np.asarray(X_with_const, dtype=float)
    endog_poisson = np.asarray(y_original, dtype=float)
    model_tests_poisson["heterogeneity"] = _compute_heterogeneity_lr_glm(
        exog_poisson, endog_poisson, llf, len(features) + 1, "poisson"
    )

    return {
        "model_type": "poisson",
        "coefficients": coefficients_unstandardized,
        "intercept": intercept_unstandardized,
        "intercept_se": intercept_se,
        "intercept_z": float(intercept_z),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "incidence_rate_ratios": irr_values,  # 发生率比
        "intercept_irr": intercept_irr,
        "intercept_irr_ci": intercept_irr_ci,
        "irr_ci_values": irr_ci_values,
        "p_values": p_values,
        "z_values": z_values,  # Z统计量
        "se_values": se_values,
        "mcfadden_r_squared": float(mcfadden_r2),
        "cox_snell_r_squared": float(cox_snell_r2),
        "nagelkerke_r_squared": float(nagelkerke_r2),
        "aic": aic,
        "bic": bic,
        "log_likelihood": llf,
        "deviance": deviance,
        "pearson_chi2": pearson_chi2,
        "vif_values": vif_values,
        "n_samples": n,
        "n_features": len(features),
        "y_pred": y_pred.tolist(),
        "model_tests": model_tests_poisson,
    }


def fit_negative_binomial(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    standardize: bool = True,
    max_iter: int = 100,
    **kwargs: Any,
) -> Dict[str, Any]:
    """拟合负二项回归模型。

    负二项回归用于建模计数数据（非负整数），特别适用于存在过度离散（overdispersion）的情况。
    当泊松回归的方差假设不满足时，负二项回归是更好的选择。
    使用statsmodels的NegativeBinomial回归实现。

    Args:
        df: 数据框
        target: 目标变量（因变量）列名，必须是计数数据（非负整数）
        features: 特征（自变量）列名列表
        standardize: 是否对特征进行标准化，默认 True
        max_iter: 最大迭代次数，默认 100

    Returns:
        Dict[str, Any]: 包含模型类型、系数、统计量等的字典
    """
    # 运行时动态检查 statsmodels 是否可用
    try:
        import statsmodels.api as sm
        # 设置使用基于对数似然的 BIC，避免 FutureWarning
        try:
            from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
            SET_USE_BIC_LLF(True)
        except (ImportError, AttributeError):
            # 如果设置失败，继续使用默认方法（会有警告但功能正常）
            pass
    except ImportError:
        raise ImportError("statsmodels 未安装，无法使用负二项回归。请运行: pip install statsmodels")

    X: pd.DataFrame = pd.DataFrame(df[features].copy())
    y: pd.Series = pd.Series(df[target].copy())

    # 处理缺失值：数值列用均值，分类列用众数
    for col in X.columns:
        col_series = cast(pd.Series, X[col])
        if pd.api.types.is_numeric_dtype(col_series):
            X[col] = col_series.fillna(col_series.mean())
        else:
            mode_val = col_series.mode()
            if len(mode_val) > 0:
                X[col] = col_series.fillna(mode_val.iloc[0])
            else:
                fallback = col_series.iloc[0] if len(col_series) > 0 else ""
                X[col] = col_series.fillna(fallback)

    # 处理目标变量的缺失值
    if pd.api.types.is_numeric_dtype(y):
        y = cast(pd.Series, y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]))
    else:
        mode_val = y.mode()
        if len(mode_val) > 0:
            y = cast(pd.Series, y.fillna(mode_val.iloc[0]))

    # 确保目标变量是非负整数（负二项回归要求）
    y = cast(pd.Series, pd.to_numeric(y, errors='coerce')).fillna(0)
    y = y.astype(int)
    if (y < 0).any():
        raise ValueError("负二项回归要求目标变量必须是非负整数，但发现负值")

    # 编码分类变量为数值
    label_encoders: Dict[str, LabelEncoder] = {}
    X_encoded = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    X = X_encoded

    # 保存原始数据用于计算非标准化系数
    X_original = X.to_numpy()
    y_original = y.to_numpy()

    # 计算原始数据的均值和标准差（用于系数转换）
    X_mean = np.mean(X_original, axis=0)
    X_std = np.std(X_original, axis=0, ddof=0)

    # 标准化特征（如果要求）
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.to_numpy()
    X_scaled = np.asarray(X_scaled, dtype=float)

    # 添加常数项（截距）
    X_with_const = sm.add_constant(X_scaled)

    # 拟合负二项回归模型
    # 使用 NegativeBinomial 族，默认使用 NB2 参数化（最常用）
    # 先拟合一个泊松回归来估计初始的离散参数 alpha
    # 这样可以避免警告，并且提供更好的初始值
    initial_alpha = 1.0  # 默认值
    try:
        # 先拟合泊松回归来估计过度离散程度
        poisson_model = sm.GLM(y_original, X_with_const, family=sm.families.Poisson())
        poisson_result = poisson_model.fit(maxiter=max_iter, method='bfgs')

        # 使用 Pearson 卡方统计量来估计 alpha
        # 对于负二项回归，alpha 可以通过 Pearson 卡方统计量估计
        # alpha ≈ (Pearson chi2 - deg_f) / deg_f，其中 deg_f = n - p
        pearson_chi2 = poisson_result.pearson_chi2
        n = len(y_original)
        p = len(features) + 1  # 特征数 + 截距
        deg_f = n - p

        if deg_f > 0 and pearson_chi2 > deg_f:
            # 估计 alpha：如果存在过度离散，alpha 应该大于 0
            # 使用更保守的估计方法
            estimated_alpha = (pearson_chi2 - deg_f) / deg_f
            # 限制 alpha 的范围在合理区间内（0.01 到 10）
            initial_alpha = max(0.01, min(10.0, estimated_alpha))
    except Exception:
        # 如果泊松回归拟合失败，使用默认值
        pass

    # 使用估计的 alpha 值创建 NegativeBinomial 族
    # 显式设置 alpha 参数以避免警告
    nb_family = sm.families.NegativeBinomial(alpha=initial_alpha)
    model = sm.GLM(y_original, X_with_const, family=nb_family)
    result = model.fit(maxiter=max_iter, method='bfgs')

    # 获取系数和统计量
    params = result.params
    bse = result.bse  # 标准误
    pvalues = result.pvalues
    conf_int = result.conf_int()  # 95%置信区间

    # 提取截距和系数
    intercept = float(params[0])
    coefficients = {feat: float(params[i + 1]) for i, feat in enumerate(features)}

    # 标准误
    intercept_se = float(bse[0])
    se_values = {feat: float(bse[i + 1]) for i, feat in enumerate(features)}

    # p值
    intercept_p = float(pvalues[0])
    p_values = {feat: float(pvalues[i + 1]) for i, feat in enumerate(features)}

    # 置信区间
    # conf_int 可能是 DataFrame 或 numpy 数组，需要统一处理
    if hasattr(conf_int, 'iloc'):
        # 如果是 DataFrame
        intercept_ci = [float(conf_int.iloc[0, 0]), float(conf_int.iloc[0, 1])]
        ci_values = {
            feat: [float(conf_int.iloc[i + 1, 0]), float(conf_int.iloc[i + 1, 1])]
            for i, feat in enumerate(features)
        }
    else:
        # 如果是 numpy 数组
        conf_int_array = np.asarray(conf_int)
        intercept_ci = [float(conf_int_array[0, 0]), float(conf_int_array[0, 1])]
        ci_values = {
            feat: [float(conf_int_array[i + 1, 0]), float(conf_int_array[i + 1, 1])]
            for i, feat in enumerate(features)
        }

    # 将标准化系数转换回原始尺度（非标准化系数，与SPSS对齐）
    # 对于负二项回归，系数转换：B_unstandardized = B_standardized / std_x
    coefficients_unstandardized = {}
    if standardize:
        for i, feat in enumerate(features):
            if X_std[i] > 0:
                coef_unstandardized = coefficients[feat] / X_std[i]
            else:
                coef_unstandardized = 0.0
            coefficients_unstandardized[feat] = float(coef_unstandardized)

        # 截距转换：intercept_unstandardized = intercept_standardized - sum(coef_unstandardized * mean_x)
        intercept_unstandardized = float(intercept - np.sum([coefficients_unstandardized[feat] * X_mean[i]
                                                           for i, feat in enumerate(features)]))
    else:
        coefficients_unstandardized = coefficients.copy()
        intercept_unstandardized = intercept

    # Z统计量（负二项回归使用Z统计量而不是t统计量，与SPSS一致）
    z_values = {feat: float(coefficients_unstandardized[feat] / se_values[feat]) if se_values[feat] > 0 else 0.0
               for feat in features}
    intercept_z = intercept_unstandardized / intercept_se if intercept_se > 0 else 0.0

    # 计算发生率比（IRR = exp(coefficient)）
    irr_values = {feat: float(np.exp(coef)) for feat, coef in coefficients_unstandardized.items()}
    intercept_irr = float(np.exp(intercept_unstandardized))

    # 预测值
    y_pred = result.fittedvalues

    # 计算模型拟合优度指标
    # 对数似然
    llf = float(result.llf)  # 对数似然
    ll_null = float(result.llnull)  # 零模型的对数似然

    # 伪R²（类似逻辑回归和泊松回归）
    # McFadden's R²
    mcfadden_r2 = 1 - (llf / ll_null) if ll_null != 0 else 0.0

    # Cox & Snell R²
    n = len(y_original)
    cox_snell_r2 = 1 - np.exp(2 * (ll_null - llf) / n)

    # Nagelkerke R²
    max_r2 = 1 - np.exp(2 * ll_null / n)
    nagelkerke_r2 = cox_snell_r2 / max_r2 if max_r2 > 0 else 0.0

    # AIC和BIC
    aic = float(result.aic)
    # 使用基于对数似然的 BIC（避免 FutureWarning）
    try:
        bic = float(result.bic_llf) if hasattr(result, 'bic_llf') else float(result.bic)
    except (AttributeError, ValueError):
        bic = float(result.bic)

    # 残差统计
    residuals = y_original - y_pred
    deviance = float(result.deviance)  # 偏差
    pearson_chi2 = float(result.pearson_chi2)  # Pearson卡方统计量

    # 获取离散参数（alpha，用于衡量过度离散程度）
    # 对于 NegativeBinomial，离散参数通常存储在模型的额外参数中
    # 如果模型收敛，可以通过残差方差与均值的比值来估计
    try:
        # 尝试从结果中获取离散参数
        if hasattr(result, 'scale'):
            dispersion_param = float(result.scale)
        else:
            # 计算过度离散参数：alpha = (variance - mean) / mean^2
            # 如果接近0，说明没有过度离散，泊松回归可能更合适
            mean_y = np.mean(y_original)
            var_y = np.var(y_original, ddof=0)
            if mean_y > 0:
                dispersion_param = float((var_y - mean_y) / (mean_y ** 2))
            else:
                dispersion_param = 0.0
    except (AttributeError, ValueError):
        dispersion_param = None

    # 计算VIF（方差膨胀因子）
    vif_values = {}
    if len(features) > 1:
        for i, feat in enumerate(features):
            X_other = np.delete(X_scaled, i, axis=1)
            y_feat = X_scaled[:, i]
            model_vif = LinearRegression()
            model_vif.fit(X_other, y_feat)
            r2_vif = r2_score(y_feat, model_vif.predict(X_other))
            vif = 1 / (1 - r2_vif) if r2_vif < 1 else np.inf
            vif_values[feat] = float(vif)
    else:
        vif_values[features[0]] = 1.0

    # 计算IRR的95%置信区间（与SPSS一致）
    intercept_irr_ci = [
        float(np.exp(intercept_ci[0])),
        float(np.exp(intercept_ci[1]))
    ]
    irr_ci_values = {
        feat: [
            float(np.exp(ci_values[feat][0])),
            float(np.exp(ci_values[feat][1]))
        ]
        for feat in features
    }

    # 稳健性检验：异方差稳健标准误(HC0)
    robustness_nb: Dict[str, Any] = {
        "method": "异方差稳健标准误(HC0)",
        "description": "为对负二项回归估计结果的稳健性进行检验，本文采用异方差稳健标准误(HC0)对模型重新估计。",
        "robust_se": {},
        "robust_p_values": {},
        "intercept_robust_se": None,
        "intercept_robust_p": None,
        "conclusion_consistent": True,
        "interpretation": "",
        "interpretation_detail": [],
    }
    try:
        robust_result_nb = result.get_robustcov_results(cov_type="HC0")
        bse_r = robust_result_nb.bse
        pvals_r = robust_result_nb.pvalues
        robustness_nb["intercept_robust_se"] = float(bse_r[0])
        robustness_nb["intercept_robust_p"] = float(pvals_r[0])
        for i, feat in enumerate(features):
            if i + 1 < len(bse_r):
                robustness_nb["robust_se"][feat] = float(bse_r[i + 1])
                robustness_nb["robust_p_values"][feat] = float(pvals_r[i + 1])
        main_sig = {k: (v is not None and v < 0.05) for k, v in p_values.items()}
        robust_sig = {k: (robustness_nb["robust_p_values"].get(k) or 1.0) < 0.05 for k in robustness_nb["robust_p_values"]}
        robustness_nb["conclusion_consistent"] = all(
            main_sig.get(k) == robust_sig.get(k) for k in set(main_sig) | set(robust_sig)
        )
        if robustness_nb["conclusion_consistent"]:
            robustness_nb["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC0)重新估计后，各变量的系数及显著性均与主回归结果一致，未发现明显变化，说明本文的实证结果是稳健的。"
            robustness_nb["interpretation_detail"] = [
                "为考察负二项回归估计结果是否对标准误的设定敏感、是否具有稳健性，本文进行稳健性检验。负二项回归虽已通过离散参数刻画过度离散，但残差方差结构仍可能偏离模型假定，常规标准误在某些情形下可能有偏。本文采用异方差稳健标准误（HC0）对同一模型与样本重新估计，在更宽松的方差假定下给出一致的标准误估计，使 Z 检验与 p 值更可靠。",
                "在稳健性检验中，本文基于稳健协方差矩阵重新计算各系数（含截距）的标准误与 p 值，并与主回归结果逐变量比较其在 0.05 显著性水平上是否一致。",
                "检验结果表明：在采用稳健标准误后，各变量的系数及显著性均没有明显变化；主回归中在 0.05 水平上显著的自变量及其发生率比(IRR)解释在稳健标准误下仍然显著，主回归中不显著的变量在稳健标准误下仍不显著。核心解释变量系数的符号与统计显著性均与主结果一致，表明负二项回归的结论具备较好的稳健性。",
                "上述结果说明，负二项回归的估计结果在 0.05 的显著性水平上具有稳健性，并不随着标准误估计方法的变化而显著变化。因此，本文的实证结论具有较好的可信度与稳健性，主回归结果可作为全文结论与政策建议的主要依据，具有较好的内部效度。",
                "需要说明的是，本文的稳健性检验仅针对标准误估计方法这一维度。若研究关注其他方面的稳健性，可在后续研究中进一步开展相应的稳健性分析，以更全面地增强结论的可信度。",
            ]
        else:
            robustness_nb["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC0)重新估计后，部分变量的显著性结论与主结果存在差异，建议以稳健标准误下的估计与检验结果为准。"
            robustness_nb["interpretation_detail"] = [
                "为考察负二项回归估计结果是否对标准误的设定敏感，本文进行稳健性检验。本文采用异方差稳健标准误（HC0）对同一模型与样本重新估计；当方差结构进一步偏离模型假定时，HC0 可给出一致的标准误估计。在稳健性检验中，本文逐变量比较主回归与稳健标准误下在 0.05 显著性水平上的结论是否一致。",
                "检验结果表明，存在至少一个变量在\"主回归结果\"与\"稳健标准误结果\"下的显著性结论不一致。这说明该变量在稳健标准误下的不确定性与主结果有所不同，其标准误或 Z 检验对方差假定的违背较为敏感。",
                "在报告与解读时，应以稳健标准误下的系数、发生率比(IRR)及 p 值为准进行结论撰写，并可在正文中说明\"经稳健标准误检验\"，以充分验证实证结论的可信度与稳健性。",
                "稳健性检验的结果有助于识别对标准误设定敏感的变量，使研究结论更加审慎、更符合计数数据模型的计量规范。",
            ]
    except Exception as e:
        robustness_nb["conclusion_consistent"] = False
        robustness_nb["interpretation"] = f"稳健性检验计算失败: {str(e)}"
        robustness_nb["interpretation_detail"] = []
    model_tests_nb: Dict[str, Any] = {"robustness": robustness_nb}
    # 异质性检验（似然比，分组系数稳定性）
    exog_nb = np.asarray(X_with_const, dtype=float)
    endog_nb = np.asarray(y_original, dtype=float)
    model_tests_nb["heterogeneity"] = _compute_heterogeneity_lr_glm(
        exog_nb, endog_nb, llf, len(features) + 1, "negative_binomial"
    )

    return {
        "model_type": "negative_binomial",
        "coefficients": coefficients_unstandardized,
        "intercept": intercept_unstandardized,
        "intercept_se": intercept_se,
        "intercept_z": float(intercept_z),
        "intercept_p": intercept_p,
        "intercept_ci": intercept_ci,
        "ci_values": ci_values,
        "incidence_rate_ratios": irr_values,  # 发生率比
        "intercept_irr": intercept_irr,
        "intercept_irr_ci": intercept_irr_ci,
        "irr_ci_values": irr_ci_values,
        "p_values": p_values,
        "z_values": z_values,  # Z统计量
        "se_values": se_values,
        "mcfadden_r_squared": float(mcfadden_r2),
        "cox_snell_r_squared": float(cox_snell_r2),
        "nagelkerke_r_squared": float(nagelkerke_r2),
        "aic": aic,
        "bic": bic,
        "log_likelihood": llf,
        "deviance": deviance,
        "pearson_chi2": pearson_chi2,
        "dispersion_param": dispersion_param,  # 离散参数（alpha）
        "vif_values": vif_values,
        "n_samples": n,
        "n_features": len(features),
        "y_pred": y_pred.tolist(),
        "model_tests": model_tests_nb,
    }
