#!/usr/bin/env python3
# =============================================================================
# SCRIPT 04: BACKTESTING & PERFORMANCE ANALYSIS (Modified for Sub-Periods)
# =============================================================================
"""
This script serves as the final analysis layer for the GARCH-Copula pipeline.

It performs the following steps:
1.  Loads the simulation forecasts, actual market returns, and PIT data.
2.  Defines distinct market regimes (Crisis vs. Non-Crisis).
3.  Conducts rigorous backtesting for each copula model's VaR and ES forecasts
    ACROSS EACH REGIME, including Kupiec, Christoffersen, and ES tests.
4.  Calculates portfolio performance metrics for each regime.
5.  Generates a single, comprehensive summary report and diagnostic plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import matplotlib as mpl
from datetime import datetime
import re   

from config import Config

# ============================================================================ #
# 1. CONFIGURATION
# ============================================================================ #

# --- File Paths ---
os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)

# --- Input Files ---
FORECAST_RESULTS_CSV = f"{Config.SIMULATION_DIR}/garch_copula_all_results.csv"
ACTUAL_RETURNS_CSV = f"{Config.DATA_DIR}/spx_dax_daily_data.csv"
PIT_DATA_CSV = f"{Config.MODEL_DIR}/copula_input_data_full.csv"

print("All data files loaded successfully.")

# --- Parameters ---
ALPHA = 0.01  # For 99% VaR / ES
MODELS_TO_TEST = ("Gaussian", "StudentT", "Clayton", "Gumbel")

# --- Plotting & Style ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 120
})


# ============================================================================ #
# 2. STATISTICAL TEST FUNCTIONS (No changes needed here)
# ============================================================================ #

def kupiec_pof(hits: pd.Series, alpha=ALPHA):
    """
    Performs Kupiec's Proportion-of-Failures (POF) test for VaR backtesting.
    Checks if the observed number of VaR breaches is consistent with the
    expected number at a given confidence level.
    """
    n = len(hits)
    n1 = hits.sum()
    if n == 0:  # Safety check for empty data
        return n1, np.nan, np.nan, np.nan

    p = alpha
    pi_hat = n1 / n if n > 0 else 0

    # Log-likelihood calculation with edge case handling (0 or 100% breaches)
    if n1 in (0, n):
        log_likelihood_ratio = -2 * n * np.log((1 - p) if n1 == 0 else p)
    else:
        log_likelihood_ratio = 2 * (n1 * np.log(pi_hat / p) +
                                      (n - n1) * np.log((1 - pi_hat) / (1 - p)))

    p_value = 1 - stats.chi2.cdf(log_likelihood_ratio, 1)
    return n1, log_likelihood_ratio, p_value, pi_hat


def christoffersen(hits: pd.Series):
    """
    Performs Christoffersen's independence test.
    Checks if VaR breaches are independent of each other (i.e., not clustered).
    Uses a small sample smoothing adjustment (+0.5) to prevent division by zero.
    """
    if len(hits) < 15 or hits.sum() < 2: # Test requires some breaches to be meaningful
        return np.nan, np.nan

    trans = pd.DataFrame({"prev": hits.shift(1), "curr": hits}).dropna()
    n00 = len(trans[(trans.prev == 0) & (trans.curr == 0)]) + 0.5
    n01 = len(trans[(trans.prev == 0) & (trans.curr == 1)]) + 0.5
    n10 = len(trans[(trans.prev == 1) & (trans.curr == 0)]) + 0.5
    n11 = len(trans[(trans.prev == 1) & (trans.curr == 1)]) + 0.5

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi == 0 or pi ==1: return np.nan, np.nan
    
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)

    logL0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    
    # Handle cases where pi0 or pi1 could be 0 or 1
    if pi0 == 0 or pi0 == 1 or pi1 == 0 or pi1 == 1:
        logL1 = logL0
    else:
        logL1 = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    
    stat = 2 * (logL1 - logL0)
    pval = 1 - stats.chi2.cdf(stat, 1)
    return stat, pval


def es_backtest(realized_returns, es_forecast, var_forecast):
    """
    Simplified Expected Shortfall backtest (based on Acerbi–Székely).
    Checks if the average realized loss during VaR breaches is consistent with
    the forecasted Expected Shortfall. The ES should not underestimate the loss.
    """
    breaches = realized_returns <= var_forecast
    k = breaches.sum()
    if k < 5:  # Test requires a minimum number of breaches to be reliable
        return np.nan, np.nan, "Too-few"

    # Difference between forecasted ES and realized returns for breaches. Should be >= 0.
    diff = (es_forecast[breaches] - realized_returns[breaches]).values
    
    # One-sample t-test on the differences
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(k))
    p_value = 1 - stats.t.cdf(t_stat, df=k - 1) # One-sided test
    
    verdict = "Pass" if p_value > 0.05 else ("Marginal" if p_value > 0.01 else "Reject")
    return t_stat, p_value, verdict


# ============================================================================ #
# 3. MAIN BACKTESTING AND VISUALIZATION FUNCTIONS (No changes needed here)
# ============================================================================ #

def backtest_all_models(forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame, alpha=ALPHA):
    """
    Orchestrates the backtesting process for all specified copula models
    on a GIVEN DATAFRAME (can be a slice of the full data).
    """
    # Align actual returns with forecasts (T day's forecast for T+1's return)
    actuals_shifted = actuals_df.shift(-1)
    results = []

    for model in MODELS_TO_TEST:
        weight_cols = [f"{model}_wSPX", f"{model}_wDAX"]
        var_col, es_col = f"{model}_VaR", f"{model}_ES"

        if not set(weight_cols + [var_col, es_col]).issubset(forecasts_df.columns):
            print(f"[Warning] Skipping {model}: Required columns not found in forecast file.")
            continue

        # Combine forecasts and actuals, dropping any non-overlapping periods
        data = pd.concat([forecasts_df[weight_cols + [var_col, es_col]], actuals_shifted[["SPX_Return", "DAX_Return"]]], axis=1).dropna()
        if data.empty or len(data) < 15:
            # print(f"[Info] Skipping {model}: Not enough data for backtest in this period.")
            continue

        # --- KEY STEP: Calculate portfolio return using model's dynamic weights ---
        portfolio_return = data["SPX_Return"] * data[weight_cols[0]] + data["DAX_Return"] * data[weight_cols[1]]
        benchmark_return = 0.5 * data["SPX_Return"] + 0.5 * data["DAX_Return"]

        # Run tests
        hits = (portfolio_return <= data[var_col]).astype(int)
        n1, kup_stat, kup_p, brate = kupiec_pof(hits, alpha)
        chr_stat, chr_p = christoffersen(hits)
        t_es, p_es, v_es = es_backtest(portfolio_return, data[es_col], data[var_col])

        # Performance metrics
        ann_mu = portfolio_return.mean() * 252
        ann_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = ann_mu / ann_vol if ann_vol > 0 else np.nan
        bench_sharpe = benchmark_return.mean() * 252 / (benchmark_return.std() * np.sqrt(252))

        results.append({
            "Model": model,
            "Observations": len(portfolio_return),
            "Breaches": n1,
            "Expected": round(len(portfolio_return) * alpha, 1),
            "Breach Rate": f"{brate:.4f}",
            "Kupiec p-val": f"{kup_p:.4f}",
            "Christoffersen p-val": f"{chr_p:.4f}" if not np.isnan(chr_p) else "NA",
            "ES Test Verdict": v_es,
            "Annualized Return %": f"{ann_mu:.2f}",
            "Annualized Vol %": f"{ann_vol:.2f}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Performance vs 50/50": "Better" if sharpe > bench_sharpe else "Worse"
        })
    return pd.DataFrame(results)


def plot_dependence_structure(pit_df, period=None, tag="full_sample"):
    """
    Visualizes the joint distribution of PITs to analyze dependence.
    """
    df = pit_df.sort_index().loc[period[0]:period[1]] if period else pit_df.sort_index()
    if df.empty or not {"u_spx", "u_dax"}.issubset(df.columns):
        print(f"[Warning] Dependence plot for '{tag}' skipped: Data is empty or columns are missing.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"Joint PIT Dependence Structure – {tag.replace('_', ' ').title()} (n={len(df)})", fontsize=16)

    # Panel 1: Hexbin plot of the full distribution
    hb = ax1.hexbin(df.u_spx, df.u_dax, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax1, label="Observation Count")
    ax1.set(title="Full Distribution", xlabel="PIT (SPX)", ylabel="PIT (DAX)", aspect="equal")

    # Panel 2: Scatter plot of the lower 10% tail
    q = 0.10
    tail_mask = (df.u_spx < q) & (df.u_dax < q)
    ax2.scatter(df.u_spx[tail_mask], df.u_dax[tail_mask], s=10, alpha=0.3, c="red")
    ax2.set(xlim=(0, q), ylim=(0, q), title=f"Lower Tail (<{int(q*100)}%)",
            xlabel="PIT (SPX)", ylabel="PIT (DAX)", aspect="equal")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{Config.ANALYSIS_DIR}/dependence_{tag}.png", dpi=150)
    plt.close()

def multi_level_var_test(forecast_df: pd.DataFrame,
                         actual_returns: pd.DataFrame,
                         levels=(0.99, 0.995, 0.999),
                         models=MODELS_TO_TEST):
    """
    • 在多置信水平 (默认 99%、99.5%、99.9%) 下做 VaR 失超率测试  
    • 需要 forecast_df 中含有每个模型的 1×25 000 模拟 P/L 列  (形如 'Gaussian_SimPL')
    • 结果 CSV:  <analysis_dir>/multi_level_var.csv
      图表 PNG:  <analysis_dir>/var_curve_comparison.png
    """
    # ---------- 工具：把字符串 / list / ndarray 统一成 float ndarray ----------
    def _str_to_ndarray(x) -> np.ndarray:
        """
        支持：
          • '[0.01  0.02 …]'   • '[0.01,0.02,…]'  • 带换行 / 科学计数
          • list / tuple / ndarray / pandas-array
        解析失败或空串时返回空 ndarray
        """
        if isinstance(x, str):
            # 去掉方括号 & 换行，逗号→空格，连续空格压缩
            clean = re.sub(r'[\[\]\n]', ' ', x).replace(',', ' ').strip()
            if not clean:
                return np.array([], dtype=float)
            arr = np.fromstring(clean, sep=' ', dtype=float)
            return arr
        else:
            return np.asarray(x, dtype=float)  # list / ndarray / 等

    # ---------- 计算实际组合收益 ----------
    act = actual_returns.copy()
    act["Portfolio_Return"] = 0.5*act["SPX_Return"] + 0.5*act["DAX_Return"]

    rows = []

    for model in models:
        col_sim = f"{model}_SimPL"
        if col_sim not in forecast_df.columns:
            print(f"[Warning] {col_sim} missing – skip {model}")
            continue

        # 对齐日期索引
        merged = forecast_df[[col_sim]].join(
            act["Portfolio_Return"], how="inner"
        ).dropna()
        if merged.empty:
            continue

        # 解析字符串 → ndarray；保留索引便于后面对齐
        sim_series = merged[col_sim].map(_str_to_ndarray)

        for alpha in levels:
            # 对每日 25 000 跑分布分位；只保留成功解析的那几天
            VaR_list = []
            realized_list = []

            for dt, pl in sim_series.items():
                if pl.size:  # 非空
                    VaR_list.append(np.percentile(pl, 100*(1-alpha)))
                    realized_list.append(merged.at[dt, "Portfolio_Return"])

            if not VaR_list:        # 该 alpha 没可用数据
                continue

            VaR_arr  = np.asarray(VaR_list)
            real_arr = np.asarray(realized_list)

            breaches = (real_arr < VaR_arr).mean()

            rows.append({
                "Model"      : model,
                "ConfLevel"  : alpha,
                "BreachRate" : breaches,
                "Expected"   : 1-alpha,
                "RelError"   : breaches / (1-alpha)
            })

    # ---------- 输出结果 ----------
    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("[Info] No multi-level data generated."); return

    out_csv = f"{Config.ANALYSIS_DIR}/multi_level_var.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"• Multi-level VaR table saved → {out_csv}")

    # ---------- 绘图 ----------
    plt.figure(figsize=(8,5))
    for mdl, grp in result_df.groupby("Model"):
        plt.plot(grp["Expected"], grp["RelError"], "o-", label=mdl)

    plt.axhline(1, ls="--", color="grey")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Expected Breach Probability")
    plt.ylabel("Relative Error (actual / expected)")
    plt.title("Multi-level VaR Backtest")
    plt.legend(); plt.tight_layout()

    out_png = f"{Config.ANALYSIS_DIR}/var_curve_comparison.png"
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"• Multi-level VaR plot  saved → {out_png}")


# ============================================================================ #
# 4. MAIN EXECUTION BLOCK
# ============================================================================ #
if __name__ == "__main__":
    print("=" * 80)
    print(">>> SCRIPT 04: BACKTESTING & PERFORMANCE ANALYSIS <<<")
    
    # --- 4.1 Load Data ---
    print("\n--- 1. Loading Data ---")
    try:
        forecasts_df = pd.read_csv(FORECAST_RESULTS_CSV, index_col="Date", parse_dates=True)
        actuals_df = pd.read_csv(ACTUAL_RETURNS_CSV, index_col="Date", parse_dates=True)
        pit_full_df = pd.read_csv(PIT_DATA_CSV, index_col="Date", parse_dates=True)
        print("All data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"[Error] Could not find a required data file: {e}")
        print("Please ensure scripts 1, 2, and 3 have been run successfully.")
        exit()
    
    # === DEBUG: 看看 forecasts_df 里到底有哪些列 =========
    print(">> First 40 columns in forecasts_df:")
    print(forecasts_df.columns.tolist()[:40])
    print("="*60)


    # Clean timezone info for consistency
    for df in (forecasts_df, actuals_df, pit_full_df):
        if df.index.tz:
            df.index = df.index.tz_localize(None)

    # <<< NEW: DEFINE MARKET REGIMES HERE >>>
    # --------------------------------------------------------------------------
    print("\n--- 2. Defining Market Regimes ---")
    
    # Define crisis periods
    crisis_periods = {
        "CovidCrash": ("2020-02-19", "2020-04-10"),
        "BankingCrisis23": ("2023-03-08", "2023-03-31")
    }
    
    # Create a boolean mask for all crisis days
    crisis_mask = pd.Series(False, index=forecasts_df.index)
    for name, (start, end) in crisis_periods.items():
        crisis_mask.loc[start:end] = True
        print(f"Regime '{name}' defined: {start} to {end}")

    # Define all regimes for analysis
    all_regimes = {
        "Full Sample": forecasts_df.index,
        **{name: forecasts_df.loc[start:end].index for name, (start, end) in crisis_periods.items()},
        "Non-Crisis": forecasts_df.index[~crisis_mask]
    }
    print(f"Regime 'Non-Crisis' defined: All days excluding crisis periods.")
    # --------------------------------------------------------------------------
    # <<< END NEW >>>

    # <<< NEW: LOOP THROUGH REGIMES FOR BACKTESTING >>>
    # --------------------------------------------------------------------------
    print("\n--- 3. Running Backtests for Each Regime ---")
    
    all_results = []
    
    for regime_name, regime_index in all_regimes.items():
        print(f"\n... Backtesting for: {regime_name} ({len(regime_index)} obs) ...")
        
        # Slice the data for the current regime
        forecasts_slice = forecasts_df.loc[regime_index]
        actuals_slice = actuals_df.loc[regime_index]
        
        # Run the backtest on the sliced data
        regime_summary_df = backtest_all_models(forecasts_slice, actuals_slice, alpha=ALPHA)
        
        # Add the regime name to the results
        if not regime_summary_df.empty:
            regime_summary_df.insert(0, "Market Regime", regime_name)
            all_results.append(regime_summary_df)

    # Combine all results into a single comprehensive table
    final_summary_df = pd.concat(all_results, ignore_index=True)
    
    print("\n--- 4. Comprehensive Backtesting Summary ---")
    print(final_summary_df.to_markdown(index=False))
    # --------------------------------------------------------------------------
    # <<< END NEW >>>
    
    summary_filename = f"{Config.ANALYSIS_DIR}/backtesting_summary_{datetime.now():%Y%m%d_%H%M}.csv"
    final_summary_df.to_csv(summary_filename, index=False)
    print(f"\nSaved COMPREHENSIVE backtesting summary to: {summary_filename}")

    # --- 4.3 Generate Diagnostic Plots (This part remains largely the same) ---
    print("\n--- 5. Generating Diagnostic Plots ---")
    
    # Dependence Plots (No changes needed, uses the original crisis_periods dict)
    plot_dependence_structure(pit_full_df, tag="full_sample")
    for name, (start, end) in crisis_periods.items():
        plot_dependence_structure(pit_full_df, period=(start, end), tag=name)

    # VaR Breaches Plot (using the best performing model, e.g., Student-t)
    model_to_plot = "StudentT"
    if f"{model_to_plot}_VaR" in forecasts_df.columns:
        w_cols = [f"{model_to_plot}_wSPX", f"{model_to_plot}_wDAX"]
        var_col = f"{model_to_plot}_VaR"
        
        tmp = pd.concat([forecasts_df[[var_col] + w_cols], actuals_df.shift(-1)], axis=1).dropna()
        ret = tmp["SPX_Return"] * tmp[w_cols[0]] + tmp["DAX_Return"] * tmp[w_cols[1]]
        var = tmp[var_col]
        breaches = ret <= var
        
        plt.figure(figsize=(14, 6))
        plt.plot(ret, label="Portfolio Return", lw=0.8, alpha=0.8)
        plt.plot(var, 'r--', label=f"VaR (alpha={ALPHA})", lw=1)
        plt.scatter(ret.index[breaches], ret[breaches], c="red", s=25, zorder=5, label=f"Breaches: {breaches.sum()}")
        plt.title(f"{model_to_plot} Copula: Portfolio Return vs. 99% VaR Forecast (Full Sample)")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{Config.ANALYSIS_DIR}/var_breaches_plot.png", dpi=150)
        plt.close()

    # Weight Distribution Plots
    plt.figure(figsize=(12, 10))
    for i, model in enumerate(MODELS_TO_TEST, 1):
        w_col = f"{model}_wSPX"
        if w_col in forecasts_df.columns:
            plt.subplot(2, 2, i)
            sns.histplot(forecasts_df[w_col], bins=30, kde=True)
            plt.title(f"{model} Copula: Weight on SPX")
            plt.axvline(0.5, c='r', ls='--', lw=1)
            plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{Config.ANALYSIS_DIR}/weight_distributions.png", dpi=150)
    plt.close()

    print("\nDiagnostic plots saved to:", Config.ANALYSIS_DIR)
    print("=" * 80)
    print(">>> SCRIPT 04 FINISHED <<<")
    print("=" * 80)

    # ===== 新增函数 =====
    print("\n--- 6. Advanced Tail Risk Analysis ---")
    multi_level_var_test(forecasts_df, actuals_df)
