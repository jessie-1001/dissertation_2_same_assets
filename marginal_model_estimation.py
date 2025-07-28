# =============================================================================
# AUTOMATED MARGINAL GARCH MODEL SELECTION & DIAGNOSTICS (rev 13 - Fixed QQPlot)
# =============================================================================
"""
Fits univariate GARCH‑family models to SPX and DAX log‑returns,
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

MIN_ALPHA = 1e-4
MIN_LB_P  = 0.005

# --- 2. Helper Functions ---
coef_sum = lambda params, pre: sum(v for k, v in params.items() if k.startswith(pre))

def calc_half_life(params, vol) -> float:
    if vol == "GARCH":
        pers = coef_sum(params, "alpha[") + coef_sum(params, "beta[")
    elif vol == "GJR":
        pers = (coef_sum(params, "alpha[") + coef_sum(params, "beta[") +
                0.5 * coef_sum(params, "gamma["))
    elif vol == "EGARCH":
        beta = abs(coef_sum(params, "beta["))
        if beta >= 1 or beta <= 0: return np.inf
        return np.log(0.5) / np.log(beta)
    else:
        pers = np.nan
    if pers >= 1 or pers <= 0: return np.inf
    return np.log(0.5) / np.log(pers)

def fit_once(series, vol, dist, p, q, mean_kw):
    try:
        mdl = arch_model(series, p=p, q=q, dist=dist,
                         rescale=False, **mean_kw, **VOL_FAMILIES[vol])
        res = mdl.fit(disp="off")
        if res.convergence_flag != 0:
            print(f"⚠ {vol}-{dist} convergence flag = {res.convergence_flag}")
        return res
    except Exception:
        return None

def passes_basic(res, series) -> bool:
    a = abs(coef_sum(res.params, "alpha["))
    g = abs(coef_sum(res.params, "gamma["))
    if a < MIN_ALPHA and g >= MIN_ALPHA: return False
    lb_p = sm.stats.acorr_ljungbox(
        pd.Series(res.std_resid, index=series.index).dropna(),
        lags=[10], return_df=True)["lb_pvalue"].iloc[0]
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

def diagnostics(res, series, tag, vol):
    params = res.params
    print(f"\n--- Parameters for {tag} ---")
    df_out = pd.DataFrame({
        "Coeff": params,
        "p-value": res.pvalues.apply(lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}")
    })
    print(df_out.to_markdown(floatfmt=".4f"))
    half = calc_half_life(params, vol)
    if np.isfinite(half):
        print(f"Half-life: {half:.1f} days")
    else:
        print("Half-life: persistence >= 1")
    std = pd.Series(res.std_resid, index=series.index).dropna()
    for lag in (5, 10, 20):
        lb_p = sm.stats.acorr_ljungbox(std, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]
        print(f"Ljung-Box p lag {lag}: {lb_p:.4f}")
    arch_f, arch_p = het_arch(std)[:2]
    print(f"ARCH-LM: F={arch_f:.3f}, p={arch_p:.4f}")

def pit_series(res, series, eps=1e-9):
    dist_obj = getattr(res, "distribution", res.model.distribution)
    theta = res.params[-dist_obj.num_params:]
    std = pd.Series(res.std_resid, index=series.index).dropna()
    u = dist_obj.cdf(std.to_numpy(), theta)
    return pd.Series(u, index=std.index).clip(eps, 1 - eps)

def oos_mse(res, full_series, start_idx):
    try:
        oos_length = len(full_series) - start_idx
        if oos_length <= 0: return np.nan
        forecast = res.forecast(horizon=oos_length)
        if forecast.variance.empty: return np.nan
        var_fcst = forecast.variance.iloc[0]
        end_idx = start_idx + len(var_fcst)
        realized = full_series.iloc[start_idx:end_idx].values ** 2
        if len(realized) != len(var_fcst): return np.nan
        return np.mean((realized - var_fcst.values) ** 2)
    except Exception as e:
        print(f"⚠ OOS MSE calculation failed: {str(e)}")
        traceback.print_exc()
        return np.nan

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
    """Fixed QQ plot implementation for GED and other distributions"""
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

    # 3. Manual Q-Q Plot (FIXED for GED and other distributions)
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
        if full_data is not None and split_idx is not None:
            mse = oos_mse(res, full_data[col], split_idx)
            print(f"OOS variance MSE for {short}: {mse:.6f}" if not np.isnan(mse) else f"OOS variance MSE for {short}: calculation failed.")

        out_path = os.path.join(RESULTS_DIR, f"{short}_{label}.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"{spec_str} | BIC={bic:,.2f}\n\n{res.summary()}")
        
        results[short] = res
        stage_summary.append({
            "Index": short,
            "Sample": label,
            "Best Model": spec_str,
            "BIC": f"{bic:,.2f}",
            "OOS MSE": f"{mse:.4f}" if not np.isnan(mse) else "N/A"
        })
        
    return results, stage_summary

def main():
    print("=" * 80)
    print("MARGINAL GARCH MODELLING (with Visuals & Summary)")

    try:
        data = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("Please ensure the data file from the first script is in the same directory.")
        return

    split = int(len(data) * IN_SAMPLE_RATIO)
    all_summaries = []

    print(f"\n>>> Stage 1 — In-sample ({split} obs) <<<")
    res_in, summary_in = run_stage(data.iloc[:split], "IS", full_data=data, split_idx=split)
    all_summaries.extend(summary_in)
    
    pd.concat([
        pit_series(res_in["SPX"], data["SPX_Return"].iloc[:split]),
        pit_series(res_in["DAX"], data["DAX_Return"].iloc[:split])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1).to_csv("copula_input_data.csv")
    print("In-sample PIT saved → copula_input_data.csv\n")

    print(f">>> Stage 2 — Full sample ({len(data)} obs) <<<")
    res_full, summary_full = run_stage(data, "FS")
    all_summaries.extend(summary_full)

    pd.concat([
        pit_series(res_full["SPX"], data["SPX_Return"]),
        pit_series(res_full["DAX"], data["DAX_Return"])
    ], axis=1).set_axis(["u_spx", "u_dax"], axis=1).to_csv("copula_input_data_full.csv")
    print("Full-sample PIT saved → copula_input_data_full.csv")

    print("\n" + "=" * 80)
    print(">>> FINAL MODEL SUMMARY <<<")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_markdown(index=False))
    print("=" * 80)
    
    print("\nProcess finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred in main execution: {exc}")
        traceback.print_exc()