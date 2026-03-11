"\"\"Helper functions for regression models: robustness checks, heterogeneity tests, etc.\"\""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None


def _rdd_p_value_passed(p: Any) -> bool:
    "\"\"Return True if p_value is None, NaN, or >= 0.05 (no rejection). Safe for type checker.\"\""
    if p is None:
        return True
    if isinstance(p, float) and np.isnan(p):
        return True
    return p >= 0.05


def _compute_robustness_ols(
    X_original: np.ndarray,
    y_original: np.ndarray,
    features: List[str],
    p_values_main: Dict[str, float],
) -> Dict[str, Any]:
    "\"\"使用异方差稳健标准误(HC1)对OLS模型进行稳健性检验。用于线性/岭/Lasso回归。\"\""
    out: Dict[str, Any] = {
        "method": "异方差稳健标准误(HC1)",
        "description": "为对主回归结果的稳健性进行检验，本文采用异方差稳健标准误(HC1)对模型重新估计。",
        "robust_se": {},
        "robust_t_values": {},
        "robust_p_values": {},
        "intercept_robust_se": None,
        "intercept_robust_t": None,
        "intercept_robust_p": None,
        "conclusion_consistent": True,
        "interpretation": "",
        "interpretation_detail": [],
    }
    if not STATSMODELS_AVAILABLE or X_original.size == 0 or len(y_original) == 0:
        out["interpretation"] = "稳健性检验需要 statsmodels，或数据为空，无法计算。"
        return out
    assert sm is not None
    try:
        X_const = sm.add_constant(X_original)
        ols_result = sm.OLS(y_original, X_const).fit()
        robust = ols_result.get_robustcov_results(cov_type="HC1")
        bse = robust.bse
        tvals = robust.tvalues
        pvals = robust.pvalues
        out["intercept_robust_se"] = float(bse[0])
        out["intercept_robust_t"] = float(tvals[0])
        out["intercept_robust_p"] = float(pvals[0])
        for i, feat in enumerate(features):
            idx = i + 1
            if idx < len(bse):
                out["robust_se"][feat] = float(bse[idx])
                out["robust_t_values"][feat] = float(tvals[idx])
                out["robust_p_values"][feat] = float(pvals[idx])
        # 比较显著性结论是否一致（以 0.05 为界）
        main_sig = {k: (v is not None and v < 0.05) for k, v in p_values_main.items()}
        robust_sig = {k: (out["robust_p_values"].get(k) or 1.0) < 0.05 for k in out["robust_p_values"]}
        consistent = all(main_sig.get(k) == robust_sig.get(k) for k in set(main_sig) | set(robust_sig))
        out["conclusion_consistent"] = consistent
        if consistent:
            out["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC1)重新估计后，各变量的系数及显著性均与主回归结果一致，未发现明显变化，说明本文的实证结果是稳健的。"
            out["interpretation_detail"] = [
                "为考察主回归估计结果是否对标准误的设定敏感、是否具有稳健性，本文进行稳健性检验。稳健性检验是实证研究中的重要环节，其目的在于判断当放松\"同方差\"等假设时，主回归所得的系数估计与显著性结论是否仍然成立；若存在异方差而仍使用常规标准误，则 t 检验与 p 值可能失真，影响结论的可信度。本文采用异方差稳健标准误（HC1，即 MacKinnon-White 修正）对同一模型与样本重新估计，在不对异方差形式做具体假定的前提下给出一致的标准误估计，使 t 检验与 p 值在异方差下仍可靠。",
                "在稳健性检验中，本文基于稳健协方差矩阵重新计算各回归系数（含截距）的标准误、t 统计量及 p 值，并与主回归结果逐变量比较其在 0.05 显著性水平上是否一致（即\"是否在 0.05 水平上显著\"的结论是否相同）。",
                "检验结果表明：在采用稳健标准误后，各变量的系数及显著性均没有明显变化；主回归中在 0.05 水平上显著的自变量在放松同方差假设后仍然显著，主回归中不显著的变量在稳健标准误下仍不显著。核心解释变量系数的符号、大小与统计显著性均与主结果一致，表明主回归的结论不依赖于同方差假设，具备较好的稳健性。",
                "上述结果说明，主回归的估计结果在 0.05 的显著性水平上具有稳健性，并不随着标准误估计方法（由常规 OLS 标准误改为异方差稳健标准误）的变化而显著变化。因此，本文的实证结论具有较好的可信度与稳健性，主回归结果可作为全文结论与政策建议的主要依据，具有较好的内部效度。",
                "需要说明的是，本文的稳健性检验仅针对\"异方差\"这一维度，采用 HC1 稳健标准误进行验证。若研究关注其他方面的稳健性（如不同样本期、不同子样本、不同变量度量等），可在后续研究中进一步开展相应的稳健性分析，以更全面地增强结论的可信度。",
            ]
        else:
            out["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC1)重新估计后，部分变量的显著性结论与主结果存在差异，建议以稳健标准误下的估计与检验结果为准。"
            out["interpretation_detail"] = [
                "为考察主回归估计结果是否对标准误的设定敏感，本文进行稳健性检验。本文采用异方差稳健标准误（HC1）对同一模型与样本重新估计；当残差存在异方差时，常规 OLS 标准误可能偏小或偏大，HC1 可在不设定具体异方差形式下得到一致的标准误估计。在稳健性检验中，本文逐变量比较主回归与稳健标准误下在 0.05 显著性水平上的结论是否一致。",
                "检验结果表明，存在至少一个变量在\"主回归结果\"与\"稳健标准误结果\"下的显著性结论不一致（即在 0.05 水平上，一方显著而另一方不显著，或反之）。这说明该变量在异方差修正后的不确定性与主结果有所不同，其标准误或 t 检验对同方差假设的违背较为敏感。",
                "在报告与解读时，应以稳健标准误下的系数估计、置信区间及 p 值为准进行结论撰写。若某变量在主回归中显著而在稳健标准误下不显著，不宜过度强调其显著效应；若某变量在主回归中不显著而在稳健标准误下显著，可说明在放松同方差假设后该变量具备统计显著性。建议进一步做残差同方差性检验（如 Breusch-Pagan）或残差图分析，以判断异方差的来源与程度，并在正文中说明\"经稳健标准误检验\"或报告稳健标准误下的结果，以增强结论的严谨性。",
                "稳健性检验的结果有助于识别对标准误设定敏感的变量，使研究结论更加审慎、更符合计量经济学规范，也可为后续的模型设定与诊断提供依据。",
            ]
    except Exception as e:
        out["conclusion_consistent"] = False
        out["interpretation"] = f"稳健性检验计算失败: {str(e)}"
        out["interpretation_detail"] = []
    return out


def _compute_robustness_logit(
    X_original: np.ndarray,
    y_original: np.ndarray,
    features: List[str],
    p_values_main: Dict[str, float],
) -> Dict[str, Any]:
    "\"\"使用异方差稳健标准误(HC1)对逻辑回归进行稳健性检验。\"\""
    out: Dict[str, Any] = {
        "method": "异方差稳健标准误(HC1)",
        "description": "为对逻辑回归估计结果的稳健性进行检验，本文采用异方差稳健标准误(HC1)对模型重新估计。",
        "robust_se": {},
        "robust_wald_values": {},
        "robust_p_values": {},
        "intercept_robust_se": None,
        "intercept_robust_wald": None,
        "intercept_robust_p": None,
        "conclusion_consistent": True,
        "interpretation": "",
        "interpretation_detail": [],
    }
    if not STATSMODELS_AVAILABLE or X_original.size == 0 or len(y_original) == 0:
        out["interpretation"] = "稳健性检验需要 statsmodels，或数据为空，无法计算。"
        return out
    assert sm is not None
    try:
        X_const = sm.add_constant(X_original)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制稳健性检验中 Logit 的完全分离、收敛等警告
            logit_result = sm.Logit(y_original, X_const).fit(disp=0)
        robust = logit_result.get_robustcov_results(cov_type="HC1")
        bse = robust.bse
        pvals = robust.pvalues
        params = robust.params
        # Wald = (coef/se)^2, 或直接用 pvalues
        out["intercept_robust_se"] = float(bse[0])
        out["intercept_robust_p"] = float(pvals[0])
        out["intercept_robust_wald"] = float((params[0] / bse[0]) ** 2) if bse[0] != 0 else None
        for i, feat in enumerate(features):
            idx = i + 1
            if idx < len(bse):
                out["robust_se"][feat] = float(bse[idx])
                out["robust_p_values"][feat] = float(pvals[idx])
                out["robust_wald_values"][feat] = float((params[idx] / bse[idx]) ** 2) if bse[idx] != 0 else None
        main_sig = {k: (v is not None and v < 0.05) for k, v in p_values_main.items()}
        robust_sig = {k: (out["robust_p_values"].get(k) or 1.0) < 0.05 for k in out["robust_p_values"]}
        consistent = all(main_sig.get(k) == robust_sig.get(k) for k in set(main_sig) | set(robust_sig))
        out["conclusion_consistent"] = consistent
        if consistent:
            out["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC1)重新估计后，各变量的系数及显著性均与主回归结果一致，未发现明显变化，说明本文的实证结果是稳健的。"
            out["interpretation_detail"] = [
                "为考察逻辑回归估计结果是否对标准误的设定敏感、是否具有稳健性，本文进行稳健性检验。二分类模型中因变量取值为 0/1，常规最大似然估计下的标准误在同方差（或类同方差）假设不成立时可能有偏，导致 Wald 检验与 p 值失真。本文采用异方差稳健标准误（HC1）对同一模型与样本重新估计，在异方差或类异方差情形下仍给出一致的标准误估计，使 Wald 检验与 p 值更可靠。",
                "在稳健性检验中，本文基于稳健协方差矩阵重新计算各系数（含截距）的标准误、Wald 统计量与 p 值，并与主回归结果逐变量比较其在 0.05 显著性水平上是否一致（即\"是否在 0.05 水平上显著\"的结论是否相同）。",
                "检验结果表明：在采用稳健标准误后，各变量的系数及显著性均没有明显变化；主回归中在 0.05 水平上显著的自变量及优势比(OR)解释在稳健标准误下仍然显著，主回归中不显著的变量在稳健标准误下仍不显著。核心解释变量系数的符号与统计显著性均与主结果一致，表明逻辑回归的结论不依赖于同方差假设，具备较好的稳健性。",
                "上述结果说明，逻辑回归的估计结果在 0.05 的显著性水平上具有稳健性，并不随着标准误估计方法的变化而显著变化。因此，本文的实证结论具有较好的可信度与稳健性，主回归结果可作为全文结论与政策建议的主要依据，具有较好的内部效度。",
                "需要说明的是，本文的稳健性检验仅针对\"异方差\"这一维度。若研究关注其他方面的稳健性（如不同样本、不同变量设定等），可在后续研究中进一步开展相应的稳健性分析，以更全面地增强结论的可信度。",
            ]
        else:
            out["interpretation"] = "在0.05的显著性水平上，采用异方差稳健标准误(HC1)重新估计后，部分变量的显著性结论与主结果存在差异，建议以稳健标准误下的估计与检验结果为准。"
            out["interpretation_detail"] = [
                "为考察逻辑回归估计结果是否对标准误的设定敏感，本文进行稳健性检验。本文采用异方差稳健标准误（HC1）对同一模型与样本重新估计；当常规标准误假设不成立时，HC1 可在不设定具体异方差形式下给出一致的标准误估计。在稳健性检验中，本文逐变量比较主回归与稳健标准误下在 0.05 显著性水平上的结论是否一致。",
                "检验结果表明，存在至少一个变量在\"主回归结果\"与\"稳健标准误结果\"下的显著性结论不一致。这说明该变量在稳健标准误下的不确定性与主结果有所不同，其标准误或 Wald 检验对假设的违背较为敏感。",
                "在报告与解读时，应以稳健标准误下的系数、优势比(OR)及 p 值为准进行结论撰写。若某变量在主回归中显著而在稳健标准误下不显著，不宜过度强调其显著效应；若某变量在主回归中不显著而在稳健标准误下显著，可说明在稳健标准误下该变量具备统计显著性。建议结合模型拟合优度与残差诊断，审视模型设定与变量选择，并在正文中说明\"经稳健标准误检验\"，以增强结论的严谨性。",
                "稳健性检验的结果有助于识别对标准误设定敏感的变量，使研究结论更加审慎、更符合计量规范，也可为后续的模型设定提供依据。",
            ]
    except Exception as e:
        out["conclusion_consistent"] = False
        out["interpretation"] = f"稳健性检验计算失败: {str(e)}"
        out["interpretation_detail"] = []
    return out


def _heterogeneity_interpretation_detail(
    test_type: str,
    statistic: float,
    p_value: float,
    no_heterogeneity: bool,
) -> List[str]:
    "\"\"生成异质性检验的详细解读段落（科研论文用语，可直接用于正文）。\"\""
    stat_label = "F统计量" if test_type == "chow" else "似然比统计量"
    test_label = "Chow检验" if test_type == "chow" else "似然比检验"
    p_str = "<0.001" if p_value < 0.001 else f"{p_value:.4f}"
    if no_heterogeneity:
        return [
            "为考察回归系数在不同子组间是否具有稳定性、是否存在结构性差异，本文对主回归结果进行异质性检验。"
            "异质性检验是实证研究中的重要环节，其目的在于判断所估计的效应是否在样本内具有一致性与可推广性；若系数在不同子组间存在显著差异，"
            "则全样本回归结果可能掩盖子组间的异质性，影响结论的准确性与政策含义。具体而言，本文以第一个自变量的中位数为界将样本划分为两个子组，"
            "在此基础上采用\" + test_label + \"检验两组回归系数是否相等：原假设H₀为两组系数向量相同（即不存在异质性），"
            "备择假设H₁为至少有一个系数在两组间存在显著差异。",
            "检验结果显示，\" + stat_label + \"为\" + f\"{statistic:.4f}\" + \"，对应的p值为\" + p_str + \"（p≥0.05），在0.05的显著性水平上无法拒绝原假设H₀。"
            "这表明，在按第一个自变量中位数划分的两个子组间，回归系数的估计结果未表现出显著的结构性差异，"
            "即两组系数相等的假设与样本数据相容，未发现显著的异质性证据。",
            "上述结果说明，本文所估计的自变量对因变量的影响在不同子组间具有较好的稳定性，主回归所得出的结论在样本内具有较好的可推广性。"
            "换言之，核心解释变量的系数大小、符号与统计显著性在不同子组间未发生显著改变，主回归结果能够较为一致地反映样本内各子组的共同特征。"
            "因此，本文的主要实证结论不依赖于上述分组方式，无需针对该分组进行单独的子组回归或引入分组与自变量的交互项；"
            "主回归结果可作为全文结论与政策建议的主要依据，具有较好的内部效度与解释力。",
            "需要说明的是，本文的异质性检验仅基于\"按第一个自变量中位数\"这一分组方式，其结论适用于该分组维度下的系数稳定性。"
            "若理论或政策关注其他分组维度（如行业、地区、时期、群体特征等），可在后续研究中进一步开展相应的子组分析或交互项检验，"
            "以更全面地考察效应在不同群体或情境下的异质性，从而增强研究的完整性与政策针对性。",
        ]
    else:
        return [
            "为考察回归系数在不同子组间是否具有稳定性，本文对主回归结果进行异质性检验。"
            "本文以第一个自变量的中位数为界将样本划分为两个子组，采用\" + test_label + \"检验两组回归系数是否相等："
            "原假设H₀为两组回归系数相等（无显著异质性），备择假设H₁为两组系数存在差异。"
            "检验结果显示，\" + stat_label + \"为\" + f\"{statistic:.4f}\" + \"，p值为\" + p_str + \"（p<0.05），在0.05的显著性水平上拒绝原假设H₀，"
            "表明两组回归系数存在显著差异，即存在显著的异质性特征。",
            "上述结果说明，在按第一个自变量中位数划分的子组间，回归系数存在显著的结构性差异，即自变量对因变量的影响在不同子组间并不一致。"
            "在此情况下，主回归所报告的全样本系数可视为各子组系数的某种加权平均或混合效应；若仅依据全样本结果进行解读，"
            "可能掩盖子组间的差异化表现，影响结论的准确性与政策含义。因此，有必要在报告与讨论中区分不同子组的效应，"
            "以更准确地反映自变量与因变量关系在不同群体或情境下的异质性。",
            '建议在正文或附录中报告按该分组方式得到的子组回归结果，或引入"分组虚拟变量×自变量"的交互项，以刻画并讨论效应在不同子组间的差异。'
            "同时，可从理论或制度背景出发，对异质性的可能来源进行讨论，例如：资源禀赋差异、制度环境差异、个体或群体特征差异等，"
            "从而增强实证分析的深度与说服力，并为政策制定提供更具针对性的依据。",
            "异质性检验的结果有助于揭示效应在不同子群体中的差异化表现，使研究结论更加细致、更符合实际情境中的异质性特征。"
            "该结果也可为后续研究提供分组分析或调节效应检验的方向与依据，具有重要的方法论意义与实证价值。",
        ]


def _compute_heterogeneity_chow_ols(
    X: np.ndarray,
    y: np.ndarray,
    pooled_ssr: float,
    k: int,
) -> Dict[str, Any]:
    "\"\"异质性检验（Chow检验）：检验线性回归系数在按第一列中位数分组后是否稳定。\"\""
    out: Dict[str, Any] = {
        "test_name": "Chow检验",
        "statistic": None,
        "p_value": None,
        "interpretation": "",
        "interpretation_detail": [],
        "split_by": "按第一个自变量的中位数将样本分为两组",
    }
    n = X.shape[0]
    if n < 2 * k + 1:
        out["interpretation"] = "样本量不足，无法进行异质性检验（需每组至少 k+1 个观测）"
        return out
    try:
        Z = np.column_stack([np.ones(n), X])
        med = np.median(X[:, 0])
        mask = X[:, 0] <= med
        n1, n2 = int(np.sum(mask)), int(np.sum(~mask))
        if n1 < k or n2 < k:
            out["interpretation"] = "分组后某一子组样本量不足，无法进行异质性检验"
            return out
        Z1, y1 = Z[mask], y[mask]
        Z2, y2 = Z[~mask], y[~mask]
        # 子组 OLS 残差平方和
        beta1 = np.linalg.lstsq(Z1, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(Z2, y2, rcond=None)[0]
        ssr1 = float(np.sum((y1 - Z1 @ beta1) ** 2))
        ssr2 = float(np.sum((y2 - Z2 @ beta2) ** 2))
        ssr_unrestricted = ssr1 + ssr2
        if ssr_unrestricted <= 0:
            out["interpretation"] = "子组残差平方和为零，无法计算Chow统计量"
            return out
        chow_num = (pooled_ssr - ssr_unrestricted) / k
        chow_den = ssr_unrestricted / (n - 2 * k)
        if chow_den <= 0:
            out["interpretation"] = "异质性检验分母为零，无法计算"
            return out
        F_stat = chow_num / chow_den
        F_stat = max(0.0, float(F_stat))
        p_val = float(1 - stats.f.cdf(F_stat, k, n - 2 * k))
        out["statistic"] = F_stat
        out["p_value"] = p_val
        if p_val >= 0.05:
            out["interpretation"] = '在0.05水平上未拒绝"两组系数相等"的原假设，未发现显著异质性，回归系数在子组间较为稳定。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("chow", F_stat, p_val, True)
        else:
            out["interpretation"] = '在0.05水平上拒绝"两组系数相等"的原假设，存在显著异质性，回归系数在子组间存在结构性差异，建议进行分组分析或引入交互项。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("chow", F_stat, p_val, False)
    except Exception as e:
        out["interpretation"] = f"异质性检验计算失败: {str(e)}"
    return out


def _compute_heterogeneity_chow_ridge(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    alpha: float,
    standardize: bool,
    scaler: Optional[Any],
    pooled_ssr: float,
    k: int,
) -> Dict[str, Any]:
    "\"\"异质性检验（Chow型检验）：岭回归按第一列中位数分组后系数稳定性。\"\""
    out: Dict[str, Any] = {
        "test_name": "Chow检验（岭回归）",
        "statistic": None,
        "p_value": None,
        "interpretation": "",
        "interpretation_detail": [],
        "split_by": "按第一个自变量的中位数将样本分为两组",
    }
    n = X.shape[0]
    if n < 2 * k + 1:
        out["interpretation"] = "样本量不足，无法进行异质性检验"
        return out
    try:
        med = np.median(X[:, 0])
        mask = X[:, 0] <= med
        X1, y1 = X[mask], y[mask]
        X2, y2 = X[~mask], y[~mask]
        n1, n2 = X1.shape[0], X2.shape[0]
        if n1 < k or n2 < k:
            out["interpretation"] = "分组后某一子组样本量不足，无法进行异质性检验"
            return out
        if standardize and scaler is not None:
            X1 = scaler.transform(X1)
            X2 = scaler.transform(X2)
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        m1 = Ridge(alpha=alpha, solver="auto")
        m2 = Ridge(alpha=alpha, solver="auto")
        m1.fit(X1, y1)
        m2.fit(X2, y2)
        ssr1 = float(np.sum((y1 - m1.predict(X1)) ** 2))
        ssr2 = float(np.sum((y2 - m2.predict(X2)) ** 2))
        ssr_unrestricted = ssr1 + ssr2
        if ssr_unrestricted <= 0:
            out["interpretation"] = "子组残差平方和为零，无法计算Chow统计量"
            return out
        chow_num = (pooled_ssr - ssr_unrestricted) / k
        chow_den = ssr_unrestricted / (n - 2 * k)
        if chow_den <= 0:
            return out
        F_stat = max(0.0, float(chow_num / chow_den))
        p_val = float(1 - stats.f.cdf(F_stat, k, n - 2 * k))
        out["statistic"] = F_stat
        out["p_value"] = p_val
        if p_val >= 0.05:
            out["interpretation"] = '在0.05水平上未拒绝"两组系数相等"的原假设，未发现显著异质性，回归系数在子组间较为稳定。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("chow", F_stat, p_val, True)
        else:
            out["interpretation"] = '在0.05水平上拒绝"两组系数相等"的原假设，存在显著异质性，回归系数在子组间存在结构性差异，建议进行分组分析或引入交互项。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("chow", F_stat, p_val, False)
    except Exception as e:
        out["interpretation"] = f"异质性检验计算失败: {str(e)}"
    return out


def _compute_heterogeneity_lr_logit(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    log_likelihood_pooled: float,
    k: int,
) -> Dict[str, Any]:
    "\"\"异质性检验（似然比）：逻辑回归按第一列中位数分组后系数是否一致。\"\""
    out: Dict[str, Any] = {
        "test_name": "似然比检验（分组系数稳定性）",
        "statistic": None,
        "p_value": None,
        "interpretation": "",
        "interpretation_detail": [],
        "split_by": "按第一个自变量的中位数将样本分为两组",
    }
    n = X.shape[0]
    if n < 2 * k + 1:
        out["interpretation"] = "样本量不足，无法进行异质性检验"
        return out
    try:
        med = np.median(X[:, 0])
        mask = X[:, 0] <= med
        X1, y1 = X[mask], y[mask]
        X2, y2 = X[~mask], y[~mask]
        n1, n2 = X1.shape[0], X2.shape[0]
        if n1 < k or n2 < k:
            out["interpretation"] = "分组后某一子组样本量不足，无法进行异质性检验"
            return out
        log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
        log_reg.fit(X1, y1)
        p1 = log_reg.predict_proba(X1)[:, 1]
        p1 = np.clip(p1, 1e-15, 1 - 1e-15)
        ll1 = float(np.sum(y1 * np.log(p1) + (1 - y1) * np.log(1 - p1)))
        log_reg.fit(X2, y2)
        p2 = log_reg.predict_proba(X2)[:, 1]
        p2 = np.clip(p2, 1e-15, 1 - 1e-15)
        ll2 = float(np.sum(y2 * np.log(p2) + (1 - y2) * np.log(1 - p2)))
        lr_stat = -2 * (log_likelihood_pooled - (ll1 + ll2))
        lr_stat = max(0.0, lr_stat)
        p_val = float(1 - stats.chi2.cdf(lr_stat, k))
        out["statistic"] = lr_stat
        out["p_value"] = p_val
        if p_val >= 0.05:
            out["interpretation"] = '在0.05水平上未拒绝"两组系数相等"的原假设，未发现显著异质性，回归系数在子组间较为稳定。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("lr", lr_stat, p_val, True)
        else:
            out["interpretation"] = '在0.05水平上拒绝"两组系数相等"的原假设，存在显著异质性，回归系数在子组间存在结构性差异，建议进行分组分析或引入交互项。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("lr", lr_stat, p_val, False)
    except Exception as e:
        out["interpretation"] = f"异质性检验计算失败: {str(e)}"
    return out


def _compute_heterogeneity_lr_glm(
    exog: np.ndarray,
    endog: np.ndarray,
    log_likelihood_pooled: float,
    k: int,
    family: str = "poisson",
    disp: Optional[float] = None,
) -> Dict[str, Any]:
    "\"\"异质性检验（似然比）：GLM（泊松/负二项）按第一列自变量中位数分组后系数是否一致。\"\""
    out: Dict[str, Any] = {
        "test_name": "似然比检验（分组系数稳定性）",
        "statistic": None,
        "p_value": None,
        "interpretation": "",
        "interpretation_detail": [],
        "split_by": "按第一个自变量的中位数将样本分为两组",
    }
    if not STATSMODELS_AVAILABLE or sm is None:
        out["interpretation"] = "异质性检验需要 statsmodels，无法计算。"
        return out
    n = exog.shape[0]
    if n < 2 * k + 1:
        out["interpretation"] = "样本量不足，无法进行异质性检验"
        return out
    try:
        med = np.median(exog[:, 1])
        mask = exog[:, 1] <= med
        exog1, endog1 = exog[mask], endog[mask]
        exog2, endog2 = exog[~mask], endog[~mask]
        n1, n2 = exog1.shape[0], exog2.shape[0]
        if n1 < k or n2 < k:
            out["interpretation"] = "分组后某一子组样本量不足，无法进行异质性检验"
            return out
        if family == "poisson":
            m1 = sm.GLM(endog1, exog1, family=sm.families.Poisson()).fit(disp=False)
            m2 = sm.GLM(endog2, exog2, family=sm.families.Poisson()).fit(disp=False)
        else:
            from statsmodels.discrete.discrete_model import NegativeBinomial
            m1 = NegativeBinomial(endog1, exog1, loglike_method="nb2").fit(disp=False)
            m2 = NegativeBinomial(endog2, exog2, loglike_method="nb2").fit(disp=False)
        ll1 = float(m1.llf)
        ll2 = float(m2.llf)
        lr_stat = -2 * (log_likelihood_pooled - (ll1 + ll2))
        lr_stat = max(0.0, lr_stat)
        p_val = float(1 - stats.chi2.cdf(lr_stat, k))
        out["statistic"] = lr_stat
        out["p_value"] = p_val
        if p_val >= 0.05:
            out["interpretation"] = '在0.05水平上未拒绝"两组系数相等"的原假设，未发现显著异质性，回归系数在子组间较为稳定。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("lr", lr_stat, p_val, True)
        else:
            out["interpretation"] = '在0.05水平上拒绝"两组系数相等"的原假设，存在显著异质性，回归系数在子组间存在结构性差异，建议进行分组分析或引入交互项。'
            out["interpretation_detail"] = _heterogeneity_interpretation_detail("lr", lr_stat, p_val, False)
    except Exception as e:
        out["interpretation"] = f"异质性检验计算失败: {str(e)}"
    return out
