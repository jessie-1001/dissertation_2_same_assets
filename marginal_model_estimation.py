# =============================================================================
# AUTOMATED MARGINAL GARCH MODEL SELECTION & DIAGNOSTICS  (rev 10)
# =============================================================================
"""
Fits univariate GARCH‑family models to SPX and DAX log‑returns,
selects the best spec via BIC, prints diagnostics, and exports PITs.

Fixes included
--------------
1. select_best now returns (res, spec, bic); run_stage captures bic.
2. EGARCH half‑life uses log(β) (no sqrt).
3. oos_mse uses forecast(..., reindex=False) to avoid mis‑alignment.
4. Exceptions in fit_once are logged.
"""

import os
import traceback
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_arch

DATA_FILE       = "spx_dax_daily_data.csv"
IN_SAMPLE_RATIO = 0.80
RESULTS_DIR     = "model_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

VOL_FAMILIES = {
    "GARCH":  dict(vol="GARCH",  o=0),
    "GJR":    dict(vol="GARCH",  o=1),
    "EGARCH": dict(vol="EGARCH")
}

DISTRIBUTIONS = ["t", "skewt", "ged"]
PQ_GRID       = [(1, 1)]
MEAN_SPEC     = {"Constant": dict(mean="Constant"),
                 "AR":       dict(mean="AR", lags=1)}

MIN_ALPHA = 1e-4          # ARCH effect threshold
MIN_LB_P  = 0.005         # Ljung‑Box p‑value cutoff


# ---------------------------------------------------------------- helpers ----
coef_sum = lambda params, pre: sum(v for k, v in params.items() if k.startswith(pre))


def calc_half_life(params, vol) -> float:
    """Return volatility half‑life (days); inf if persistence ≥ 1 or ≤ 0."""
    if vol == "GARCH":
        pers = coef_sum(params, "alpha[") + coef_sum(params, "beta[")
    elif vol == "GJR":
        pers = (coef_sum(params, "alpha[") + coef_sum(params, "beta[") +
                0.5 * coef_sum(params, "gamma["))
    elif vol == "EGARCH":
        beta = abs(coef_sum(params, "beta["))
        if beta >= 1 or beta <= 0:
            return np.inf
        return np.log(0.5) / np.log(beta)
    else:
        pers = np.nan

    if pers >= 1 or pers <= 0:
        return np.inf
    return np.log(0.5) / np.log(pers)


def fit_once(series, vol, dist, p, q, mean_kw):
    """Try one model; return result or None."""
    try:
        mdl = arch_model(series, p=p, q=q, dist=dist,
                         rescale=False, **mean_kw, **VOL_FAMILIES[vol])
        res = mdl.fit(disp="off")
        if res.convergence_flag != 0:
            print(f"⚠ {vol}-{dist} convergence flag = {res.convergence_flag}")
        return res
    except Exception as e:
        print(f"⚠ Fit failed for {vol}-{dist}: {e}")
        return None


def passes_basic(res, series) -> bool:
    """Quick filter: some ARCH plus Ljung‑Box on std‑resid."""
    a = abs(coef_sum(res.params, "alpha["))
    g = abs(coef_sum(res.params, "gamma["))
    if a < MIN_ALPHA and g >= MIN_ALPHA:
        return False
    lb_p = sm.stats.acorr_ljungbox(
        pd.Series(res.std_resid, index=series.index).dropna(),
        lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    return lb_p >= MIN_LB_P


def select_best(series, tag):
    """Return (result, spec_tuple, bic)."""
    candidates, shortlisted = [], []
    for mean_tag, mean_kw in MEAN_SPEC.items():
        for vol, dist, (p, q) in product(VOL_FAMILIES, DISTRIBUTIONS, PQ_GRID):
            res = fit_once(series, vol, dist, p, q, mean_kw)
            if res is None:
                continue
            spec = (mean_tag, vol, dist, p, q)
            candidates.append((res.bic, res, spec))
            if passes_basic(res, series):
                shortlisted.append((res.bic, res, spec))

    bic, res, spec = min(shortlisted or candidates, key=lambda x: x[0])
    mean_tag, vol, dist, p, q = spec
    if not shortlisted:
        print(f"⚠ best model for {tag} failed some diagnostics.")
    print(f"✔ best for {tag}: {mean_tag}+{vol}({p},{q})-{dist}, BIC={bic:,.2f}")
    return res, spec, bic


def diagnostics(res, series, tag, vol):
    params = res.params
    print(f"\n--- Parameters for {tag} ---")
    df_out = pd.DataFrame({
        "Coeff": params,
        "p‑value": res.pvalues.apply(lambda v: "<0.0001" if v < 1e-4
                                                     else f"{v:.4f}")
    })
    print(df_out.to_markdown(floatfmt=".4f"))

    dist_obj = getattr(res, "distribution", None) or res.model.distribution
    theta    = res.params[-dist_obj.num_params:]
    half     = calc_half_life(params, vol)
    if np.isfinite(half):
        print(f"Half‑life: {half:.1f} days")
    else:
        print("Half‑life: persistence ≥ 1")

    if dist_obj.name.lower() in {"t", "skewt", "skewstudent"}:
        df = float(theta[0])
        tail_prob = (dist_obj.cdf(-5, theta) +
                     1 - dist_obj.cdf(5, theta)) * 100
        label = "heavy" if df < 5 else "moderate" if df < 10 else "near‑normal"
        print(f"Tail df: {df:.2f} ({label}),  P(|r|>5σ): {tail_prob:.4f}% "
              f"(Normal≈{norm.sf(5)*200:.6f}%)")

    std = pd.Series(res.std_resid, index=series.index).dropna()
    for lag in (5, 10, 20):
        lb_p = sm.stats.acorr_ljungbox(std, lags=[lag],
                                       return_df=True)["lb_pvalue"].iloc[0]
        print(f"Ljung‑Box p lag {lag}: {lb_p:.4f}")
    arch_f, arch_p = het_arch(std)[:2]
    print(f"ARCH‑LM: F={arch_f:.3f}, p={arch_p:.4f}")


def pit_series(res, series, eps=1e-9):
    dist = getattr(res, "distribution", None) or res.model.distribution
    theta = res.params[-dist.num_params:]
    std   = pd.Series(res.std_resid, index=series.index).dropna()
    u     = dist.cdf(std.to_numpy(), theta)
    return pd.Series(u, index=std.index).clip(eps, 1 - eps)


def oos_mse(res, full_series, start_idx):
    """
    计算样本外波动率预测的均方误差
    
    参数:
    res - 拟合的GARCH模型结果
    full_series - 完整的时间序列数据
    start_idx - 样本外预测开始位置
    
    返回:
    MSE值
    """
    try:
        # 获取样本外数据长度
        oos_length = len(full_series) - start_idx
        if oos_length <= 0:
            print(f"⚠ 无效的样本外长度: {oos_length}")
            return np.nan
        
        # 确保模型有足够的拟合数据
        if start_idx > len(res.model.data):
            print(f"⚠ 开始位置 {start_idx} 超出模型数据范围 ({len(res.model.data)})")
            return np.nan
        
        # 获取预测对象（显式指定 horizon）
        forecast = res.forecast(horizon=oos_length, start=start_idx)
        
        # 检查预测结果是否为空
        if forecast.variance.empty:
            print("⚠ 预测结果为空")
            return np.nan
        
        # 获取波动率预测值
        var_fcst = forecast.variance.iloc[:, 0]
        
        # 获取实际波动率值（与预测值相同长度）
        realized = full_series.iloc[start_idx:start_idx+len(var_fcst)].values ** 2
        
        # 计算均方误差
        return np.mean((realized - var_fcst.values) ** 2)
    
    except Exception as e:
        print(f"⚠ OOS MSE计算失败: {str(e)}")
        return np.nan


def run_stage(frame, label):
    results = {}
    for col, short in [("SPX_Return", "SPX"),
                       ("DAX_Return", "DAX")]:
        res, spec, bic = select_best(frame[col], f"{short} {label}")
        _, vol, _, _, _ = spec
        diagnostics(res, frame[col], f"{short} {label}", vol)

        out = f"{RESULTS_DIR}/{short}_{label}.txt"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"{'+'.join(map(str, spec))} | BIC={bic:,.2f}\n\n"
                     f"{res.summary()}")
        results[short] = res
    return results


# ---------------------------------------------------------------- main -------
def main():
    print("=" * 80)
    print("MARGINAL GARCH MODELLING")

    data = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    split = int(len(data) * IN_SAMPLE_RATIO)

    # --------------------------------- in‑sample
    print(f"\n>>> Stage 1 — In‑sample ({split} obs) <<<")
    res_in = run_stage(data.iloc[:split], "IS")

    for short, col in [("SPX", "SPX_Return"), ("DAX", "DAX_Return")]:
        mse = oos_mse(res_in[short], data[col], split)
        print(f"OOS variance MSE for {short}: {mse:.6f}")

    pd.concat([
        pit_series(res_in["SPX"], data["SPX_Return"].iloc[:split]),
        pit_series(res_in["DAX"], data["DAX_Return"].iloc[:split])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1)\
      .to_csv("copula_input_data.csv")
    print("In‑sample PIT saved → copula_input_data.csv")

    # --------------------------------- full‑sample
    print(f"\n>>> Stage 2 — Full sample ({len(data)} obs) <<<")
    res_full = run_stage(data, "FS")

    pd.concat([
        pit_series(res_full["SPX"], data["SPX_Return"]),
        pit_series(res_full["DAX"], data["DAX_Return"])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1)\
      .to_csv("copula_input_data_full.csv")
    print("Full‑sample PIT saved → copula_input_data_full.csv")

    print("=" * 80)
    print("Finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()
