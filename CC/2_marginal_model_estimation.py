#!/usr/bin/env python3

# =============================================================================
# AUTOMATED MARGINAL GARCH MODEL SELECTION & DIAGNOSTICS (Leakage-Free)
# =============================================================================
"""
Fits univariate GARCH-family models to SPX and DAX log-returns,
selects the best spec via BIC, prints diagnostics, exports PITs,
and generates summary tables and diagnostic plots.
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
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Configuration ---
DATA_DIR        = "CC/data"
RESULTS_DIR     = "CC/model"
DATA_FILE       = f"{DATA_DIR}/spx_dax_daily_data.csv"
IN_SAMPLE_RATIO = 0.80  # 80% for training, 20% for testing
os.makedirs(RESULTS_DIR, exist_ok=True)

VOL_FAMILIES = {
    "GARCH":  dict(vol="GARCH",  o=0),
    "GJR":    dict(vol="GARCH",  o=1),
    "EGARCH": dict(vol="EGARCH"),
    "APARCH": dict(vol="APARCH")
}
DISTRIBUTIONS = ["t", "skewt", "ged"]
PQ_GRID       = [(1, 1)]  # 只使用(1,1)阶模型
MEAN_SPEC     = {"Constant": dict(mean="Constant"),
                 "AR":       dict(mean="AR", lags=1)}



# --- 2. Helper Functions ---
coef_sum = lambda params, pre: sum(v for k, v in params.items() if k.startswith(pre))


def fit_once(series, vol, dist, p, q, mean_kw):
    """
    最多重启 3 次；只有连续 3 次都未收敛才放弃该模型。
    这样 GED 等更难收敛的分布也能参与比较。
    """
    mdl = arch_model(series, p=p, q=q, dist=dist,
                     rescale=False, **mean_kw, **VOL_FAMILIES[vol])

    for _ in range(3):                           # 尝试 3 次
        res = mdl.fit(disp="off", update_freq=0, show_warning=False)
        if res.convergence_flag == 0:
            return res
    return None                 

MIN_ALPHA_KILL = 5e-4
MIN_ALPHA_WARN = 1e-3
MIN_LB_P  = 0.01  # 宽松的Ljung-Box检验

def passes_basic(res, series) -> bool:
    """
    返回 True ⇒ 该模型进入 BIC 评比池
    规则：
      1. 过滤 α≈0 γ>0 的 GJR / APARCH
      2. EGARCH beta 平稳 |β|<1
      3. APARCH power δ ∈ [1,3]
      4. Ljung‑Box p(10) ≥ MIN_LB_P
    """
    p        = res.params
    vol_name = res.model.volatility.__class__.__name__.upper()

    # ---------- 1. α≈0 + γ>0 ----------
    a = abs(coef_sum(p, "alpha["))
    g = abs(coef_sum(p, "gamma["))
    if g >= MIN_ALPHA_KILL and a < MIN_ALPHA_KILL:
        return False

    # ---------- 2. EGARCH 平稳 ----------
    if vol_name == "EGARCHVOLATILITY":
        if abs(coef_sum(p, "beta[")) >= 1:
            return False

    # ---------- 3. APARCH power ----------
    if vol_name == "APARCHVOLATILITY":
        delta = p.get("power", 2.0)
        if not (1.0 <= delta <= 3.0):
            return False

    # ---------- 4. Ljung‑Box ----------
    std = pd.Series(res.std_resid, index=series.index).dropna()
    lb_p = sm.stats.acorr_ljungbox(std, lags=[10],
                                   return_df=True)["lb_pvalue"].iloc[0]
    return lb_p >= MIN_LB_P


def select_best(series, tag):
    candidates, shortlisted = [], []
    for mean_tag, mean_kw in MEAN_SPEC.items():
        for vol, dist, (p, q) in product(VOL_FAMILIES, DISTRIBUTIONS, PQ_GRID):
            res = fit_once(series, vol, dist, p, q, mean_kw)
            if res is None: continue
            spec = (mean_tag, vol, dist, p, q)
            candidates.append((res.bic, res, spec))
            if passes_basic(res, series):
                shortlisted.append((res.bic, res, spec))
    
    if not shortlisted and candidates:
         print(f"⚠ No model for {tag} passed basic checks. Selecting from all candidates.")
    
    source = shortlisted or candidates
    if not source:
        raise RuntimeError(f"Could not fit any model for {tag}")

    bic, res, spec = min(source, key=lambda x: x[0])
    mean_tag, vol, dist, p, q = spec
    
    if not shortlisted:
        print(f"⚠ best model for {tag} failed some diagnostics.")
        
    spec_str = f"{mean_tag}+{vol}({p},{q})-{dist}"
    print(f"✔ best for {tag}: {spec_str}, BIC={bic:,.2f}")
    return res, spec, bic, spec_str

def pit_series(res, series, eps=1e-9):
    """
    Probability Integral Transform (PIT) for standardized residuals.
    This version passes parameters as a NumPy array to avoid all keyword naming issues.
    """
    dist_obj = res.model.distribution
    std = pd.Series(res.std_resid, index=series.index).dropna()

    # Get the distribution parameters as a pandas Series
    params = res.params[dist_obj.parameter_names()]

    # Convert the Series to a NumPy array and pass it directly. This is the robust method.
    u = dist_obj.cdf(std.to_numpy(), params.values)
    
    return pd.Series(u, index=std.index).clip(eps, 1 - eps)

def diagnostics(res, series, tag, vol):
    """
    打印参数表 + 关键诊断信息
    -------------------------------------------------
    • EGARCH  : 报告 β 与半衰期
    • GJR     : 杠杆显著性/α 提示
    • APARCH  : power(δ) 合理区间 + 近似持久性
    • GED     : ν 三段阈值 (<1.1 / 1.1‑1.5 / >1.5)
    • 持久性  : 
        - GJR     = α+β+0.5γ
        - APARCH  ≈ β  (常用近似)
        - 其他    = α+β
    """
    # ========== 1. 参数表 ==========
    params, pvals = res.params, res.pvalues
    print(f"\n--- Parameters for {tag} ---")
    print(pd.DataFrame({"Coeff": params, "p-value": pvals})
              .to_markdown(floatfmt=".4f"))

    # ========== 2. EGARCH 专用 ==========
    if vol == "EGARCH":
        beta_e = coef_sum(params, "beta[")
        print(f"Persistence β: {beta_e:.4f}")
        if abs(beta_e) >= 1:
            print("⚠ |β| ≥ 1 ⇒ non‑stationary EGARCH!")
        else:
            half = np.log(0.5) / np.log(abs(beta_e))
            print(f"Half‑life: {half:.1f} days")

    # ========== 3. 杠杆 / α 提示 ==========
    if vol == "GJR":
        if pvals.get("gamma[1]", 1) < 0.05:
            print("✓ Significant leverage effect (γ)")
        if pvals.get("alpha[1]", 0) > 0.10:
            print("⚠ alpha[1] not significant; consider EGARCH/APARCH.")
    elif vol == "GARCH":
        print("✗ No leverage effect captured (symmetric GARCH)")

    # ========== 4. APARCH 诊断 ==========
    if vol == "APARCH":
        delta = params.get("power", 2.0)
        print(f"Estimated δ (power): {delta:.3f}")
        if not (1.0 <= delta <= 3.0):
            print("⚠ δ out of [1,3] → unusual power parameter")
        beta_ap = coef_sum(params, "beta[")
        print(f"Approx. persistence (β): {beta_ap:.4f}")

    # ========== 5. GED ν 细分 ==========
    if res.model.distribution.name == "ged":
        nu = params["nu"]
        if nu < 1.1:
            print(f"⚠ GED ν={nu:.2f} → near‑Laplace (very heavy tails)")
        elif nu <= 1.5:
            print(f"✓ GED ν={nu:.2f} → typical heavy‑tailed finance")
        else:
            print(f"⚠ GED ν={nu:.2f} → approaching normality")

    # ========== 6. 持久性 ==========
    alpha = coef_sum(params, "alpha[")
    beta  = coef_sum(params, "beta[")
    gamma = coef_sum(params, "gamma[") if vol == "GJR" else 0

    if vol == "GJR":
        pers = alpha + beta + 0.5 * gamma
    elif vol == "APARCH":
        pers = beta       # 常见近似，严格公式与 δ 相关
    else:
        pers = alpha + beta

    print(f"Persistence estimate: {pers:.4f}")
    if pers > 0.98:
        print("!!! 高持久性警告：波动冲击可能持续超过 6 个月")

    # ========== 7. Ljung‑Box & ARCH‑LM ==========
    std = pd.Series(res.std_resid, index=series.index).dropna()
    for lag in (5, 10, 20):
        lb_p = sm.stats.acorr_ljungbox(std, lags=[lag],
                                       return_df=True)["lb_pvalue"].iloc[0]
        print(f"Ljung‑Box p lag{lag}: {lb_p:.4f}")
    arch_f, arch_p = het_arch(std)[:2]
    print(f"ARCH‑LM p: {arch_p:.4f}")




def oos_mse(res, full_series: pd.Series, split_idx: int) -> float:
    """
    1‑step 方差 MSE，列名再换也不怕：
    - 不设 start，arch 自认 len(train)-1
    - horizon = 测试集长度
    - 直接取预测表的【最后 test_len 行】【第一列】
    """
    test_len = len(full_series) - split_idx
    if test_len <= 0:
        return np.nan

    fcast = res.forecast(horizon=test_len, align="origin", reindex=True)

    # 直接拿最后 test_len 行、第一列
    var_fcst = (
        fcast.variance.iloc[-test_len:, 0]   # 无论 h.1 / h.001 / 0 / MultiIndex
        .astype(float)
        .dropna()
    )
    if var_fcst.empty:
        return np.nan

    actual = (full_series.iloc[split_idx : split_idx + len(var_fcst)]) ** 2
    return float(np.mean((actual.values - var_fcst.values) ** 2))



# --- 3. Visualization Functions ---
def plot_volatility(res, series, tag, short_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(series.index, series, color='grey', alpha=0.6, lw=1, label='Daily Returns')
    ax.plot(res.conditional_volatility.index, res.conditional_volatility, 'r-', lw=1.2, label='Conditional Volatility')
    ax.plot(res.conditional_volatility.index, -res.conditional_volatility, 'r-', lw=1.2)
    ax.set_title(f'Conditional Volatility vs Returns for {tag}', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    f_path = os.path.join(RESULTS_DIR, f"{short_name}_{tag.replace(' ', '_')}_volatility.png")
    plt.tight_layout()
    plt.savefig(f_path)
    plt.close()
    print(f"  > Volatility plot saved to {f_path}")

def plot_diagnostics(res, series, tag, short_name):
    std_resid = res.std_resid.dropna()
    pit_vals = pit_series(res, series)
    dist_obj = getattr(res, "distribution", res.model.distribution)
    theta = res.params[-dist_obj.num_params:]
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Diagnostic Plots for {tag}', fontsize=18)

    # 1. Standardized Residuals over Time
    ax[0, 0].plot(std_resid.index, std_resid)
    ax[0, 0].set_title('Standardized Residuals over Time')
    ax[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Histogram of Standardized Residuals
    sns.histplot(std_resid, ax=ax[0, 1], kde=True, stat="density", label="Empirical")
    ax[0, 1].set_title('Histogram of Standardized Residuals')
    ax[0, 1].legend()

    # 3. Manual Q-Q Plot
    sorted_resid = np.sort(std_resid)
    n = len(sorted_resid)
    p = (np.arange(n) + 0.5) / n  # Probability points
    
    # Handle different distributions
    if hasattr(dist_obj, 'ppf'):
        try:
            theoretical_quantiles = dist_obj.ppf(p, theta)
        except Exception as e:
            print(f"⚠ Using normal dist for QQ plot due to: {str(e)}")
            theoretical_quantiles = norm.ppf(p)
    else:
        print(f"⚠ Using normal dist for QQ plot (no ppf method)")
        theoretical_quantiles = norm.ppf(p)
    
    ax[1, 0].scatter(theoretical_quantiles, sorted_resid, alpha=0.6)
    min_val = min(theoretical_quantiles.min(), sorted_resid.min())
    max_val = max(theoretical_quantiles.max(), sorted_resid.max())
    ax[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    ax[1, 0].set_title('Q-Q Plot of Standardized Residuals')
    ax[1, 0].set_xlabel('Theoretical Quantiles')
    ax[1, 0].set_ylabel('Sample Quantiles')
    ax[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. PIT Histogram
    sns.histplot(pit_vals, bins=25, ax=ax[1, 1], stat="density")
    ax[1, 1].axhline(1.0, color='red', linestyle='--')
    ax[1, 1].set_title('Histogram of PIT Values')
    ax[1, 1].set_xlabel('PIT Value')
    ax[1, 1].set_ylabel('Density')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f_path = os.path.join(RESULTS_DIR, f"{short_name}_{tag.replace(' ', '_')}_diagnostics.png")
    plt.savefig(f_path)
    plt.close()
    print(f"  > Diagnostics plot saved to {f_path}")

# --- 4. Main Execution Logic ---
def run_stage(frame, label, full_data=None, split_idx=None):
    results = {}
    stage_summary = []
    
    for col, short in [("SPX_Return", "SPX"), ("DAX_Return", "DAX")]:
        series_in_sample = frame[col]
        tag = f"{short} {label}"
        
        res, spec, bic, spec_str = select_best(series_in_sample, tag)
        _, vol, _, _, _ = spec
        diagnostics(res, series_in_sample, tag, vol)

        plot_volatility(res, series_in_sample, tag, short)
        plot_diagnostics(res, series_in_sample, tag, short)

        mse = np.nan
        if label == "Train" and full_data is not None and split_idx is not None:
            mse = oos_mse(res, full_data[col], split_idx)
            if not np.isnan(mse):
                print(f"OOS variance MSE for {short}: {mse:.6f}")
            else:
                print(f"⚠ OOS MSE calculation failed for {short}")

        out_path = os.path.join(RESULTS_DIR, f"{short}_{label}.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"{spec_str} | BIC={bic:,.2f}\n\n{res.summary()}")
        
        results[short] = res
        stage_summary.append({
            "Index": short,
            "Sample": label,
            "Best Model": spec_str,
            "BIC": f"{bic:,.2f}",
            "OOS MSE": f"{mse:.6f}" if not np.isnan(mse) else "N/A"
        })
        
    return results, stage_summary
 

def main():
    print("=" * 80)
    print("MARGINAL GARCH MODELLING (Leakage-Free Protocol)")

    try:
        data = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("Please ensure the data file is in the same directory.")
        return

    split = int(len(data) * IN_SAMPLE_RATIO)
    train_data = data.iloc[:split]  # Training set (80%)
    all_summaries = []

    # ====================== STAGE 1: TRAINING SET MODELING ======================
    print(f"\n>>> STAGE 1 — Training Set Modeling ({len(train_data)} obs) <<<")
    res_train, summary_train = run_stage(train_data, "Train", full_data=data, split_idx=split)
    all_summaries.extend(summary_train)
    
    # Generate PIT for copula modeling (training set only)
    copula_input_data_path = f"{RESULTS_DIR}/copula_input_data.csv"
    pd.concat([
        pit_series(res_train["SPX"], train_data["SPX_Return"]),
        pit_series(res_train["DAX"], train_data["DAX_Return"])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1).to_csv(copula_input_data_path)
    print("Training set PIT saved → copula_input_data.csv")

    # ====================== STAGE 2: FULL SAMPLE MODELING ======================
    print(f"\n>>> STAGE 2 — Full Sample Modeling ({len(data)} obs) <<<")
    print("NOTE: Full sample models are for descriptive purposes only")
    
    res_full, summary_full = run_stage(data, "Full")
    all_summaries.extend(summary_full)

    # Generate PIT for copula modeling (full sample)
    copula_full_data_path = f"{RESULTS_DIR}/copula_input_data_full.csv"
    pd.concat([
        pit_series(res_full["SPX"], data["SPX_Return"]),
        pit_series(res_full["DAX"], data["DAX_Return"])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1).to_csv(copula_full_data_path)
    print("Full sample PIT saved → copula_input_data_full.csv")

    # ====================== FINAL SUMMARY ======================
    print("\n" + "=" * 80)
    print(">>> FINAL MODEL SUMMARY <<<")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_markdown(index=False))
    
    print("\n" + "=" * 80)
    print(">>> METHODOLOGICAL NOTES <<<")
    print("1. Training set models (80% of data) are used for statistical inference")
    print("2. OOS MSE is calculated on the test set (20% of data)")
    print("3. Full sample models are for descriptive purposes only")
    print("=" * 80)
    
    print("\nProcess finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {exc}")
        traceback.print_exc()