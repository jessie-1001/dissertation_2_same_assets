#!/usr/bin/env python3
# =============================================================================
# SCRIPT 04 (REVISED): BACKTESTING & PERFORMANCE ANALYSIS
# =============================================================================
"""
This script serves as the final analysis layer for the GARCH-Copula pipeline.

REVISIONS:
- Switched from reading a CSV to a Parquet file for massive performance gains
  and to correctly load simulation data as numpy arrays.
- Removed the `_str_to_ndarray` parsing function as it is now obsolete.

It performs the following steps:
1.  Loads the simulation forecasts (from Parquet), actual market returns, and PIT data.
2.  Defines distinct market regimes (Crisis vs. Non-Crisis).
3.  Conducts rigorous backtesting for each copula model's VaR and ES forecasts
    ACROSS EACH REGIME, including Kupiec, Christoffersen, and ES tests.
4.  Calculates portfolio performance metrics for each regime.
5.  Generates a comprehensive summary report and diagnostic plots, including an
    advanced multi-level VaR backtest using the full simulation data.
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

from config import Config

# ============================================================================ #
# 1. CONFIGURATION
# ============================================================================ #

# --- File Paths ---
os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)

# --- Input Files (REVISED to use Parquet) ---
FORECAST_RESULTS_PQ = f"{Config.SIMULATION_DIR}/garch_copula_all_results.parquet"
ACTUAL_RETURNS_CSV = f"{Config.DATA_DIR}/spx_dax_daily_data.csv"
PIT_DATA_CSV = f"{Config.MODEL_DIR}/copula_input_data_full.csv"

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
# 2. STATISTICAL TEST FUNCTIONS
# ============================================================================ #

def kupiec_pof(hits: pd.Series, alpha=ALPHA):
    """
    Performs Kupiec's Proportion-of-Failures (POF) test for VaR backtesting.
    """
    n = len(hits)
    n1 = hits.sum()
    if n == 0:
        return n1, np.nan, np.nan, np.nan

    p = alpha
    pi_hat = n1 / n if n > 0 else 0

    if n1 in (0, n) or p in (0, 1) or pi_hat in (0, 1):
        log_likelihood_ratio = 0
    else:
        log_likelihood_ratio = 2 * (n1 * np.log(pi_hat / p) +
                                      (n - n1) * np.log((1 - pi_hat) / (1 - p)))

    p_value = 1 - stats.chi2.cdf(log_likelihood_ratio, 1)
    return n1, log_likelihood_ratio, p_value, pi_hat


def christoffersen(hits: pd.Series):
    """
    Performs Christoffersen's independence test.
    Checks if VaR breaches are independent (not clustered).
    """
    if len(hits) < 15 or hits.sum() < 2:
        return np.nan, np.nan

    trans = pd.DataFrame({"prev": hits.shift(1), "curr": hits}).dropna()
    n00 = ((trans.prev == 0) & (trans.curr == 0)).sum()
    n01 = ((trans.prev == 0) & (trans.curr == 1)).sum()
    n10 = ((trans.prev == 1) & (trans.curr == 0)).sum()
    n11 = ((trans.prev == 1) & (trans.curr == 1)).sum()
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
    
    logL0 = (n00 + n10) * np.log(1 - pi_all) + (n01 + n11) * np.log(pi_all) if pi_all not in (0, 1) else 0
    logL1_comp1 = n00 * np.log(1 - pi0) + n01 * np.log(pi0) if pi0 not in (0, 1) else 0
    logL1_comp2 = n10 * np.log(1 - pi1) + n11 * np.log(pi1) if pi1 not in (0, 1) else 0
    logL1 = logL1_comp1 + logL1_comp2
    
    stat = 2 * (logL1 - logL0) if (logL1 - logL0) > 0 else 0
    pval = 1 - stats.chi2.cdf(stat, 1)
    return stat, pval


def es_backtest(realized_returns, es_forecast, var_forecast):
    """
    Simplified Expected Shortfall backtest (based on Acerbi–Székely).
    """
    breaches = realized_returns <= var_forecast
    k = breaches.sum()
    if k < 5:
        return np.nan, np.nan, "Too-few"

    diff = (es_forecast[breaches] - realized_returns[breaches]).values
    
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(k)) if (diff.std(ddof=1) > 0 and k > 1) else 0
    p_value = 1 - stats.t.cdf(t_stat, df=k - 1)
    
    verdict = "Pass" if p_value > 0.05 else ("Marginal" if p_value > 0.01 else "Reject")
    return t_stat, p_value, verdict


# ============================================================================ #
# 3. MAIN BACKTESTING AND VISUALIZATION FUNCTIONS
# ============================================================================ #

def backtest_all_models(forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame, alpha=ALPHA):
    """Orchestrates the backtesting process for all copula models on a given dataset."""
    actuals_shifted = actuals_df.shift(-1)
    results = []

    for model in MODELS_TO_TEST:
        weight_cols = [f"{model}_wSPX", f"{model}_wDAX"]
        var_col, es_col = f"{model}_VaR_990", f"{model}_ES_990"

        if not set(weight_cols + [var_col, es_col]).issubset(forecasts_df.columns):
            continue

        data = pd.concat([forecasts_df[weight_cols + [var_col, es_col]], actuals_shifted[["SPX_Return", "DAX_Return"]]], axis=1).dropna()
        if data.empty or len(data) < 15:
            continue

        portfolio_return = data["SPX_Return"] * data[weight_cols[0]] + data["DAX_Return"] * data[weight_cols[1]]
        benchmark_return = 0.5 * data["SPX_Return"] + 0.5 * data["DAX_Return"]

        hits = (portfolio_return <= data[var_col]).astype(int)
        n1, _, kup_p, brate = kupiec_pof(hits, alpha)
        _, chr_p = christoffersen(hits)
        _, _, v_es = es_backtest(portfolio_return, data[es_col], data[var_col])

        ann_mu = portfolio_return.mean() * 252
        ann_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = ann_mu / ann_vol if ann_vol > 0 else np.nan
        bench_sharpe = benchmark_return.mean() * 252 / (benchmark_return.std() * np.sqrt(252)) if benchmark_return.std() > 0 else np.nan

        results.append({
            "Model": model, "Observations": len(portfolio_return), "Breaches": n1,
            "Expected": round(len(portfolio_return) * alpha, 1), "Breach Rate": f"{brate:.4f}",
            "Kupiec p-val": f"{kup_p:.4f}",
            "Christoffersen p-val": f"{chr_p:.4f}" if not np.isnan(chr_p) else "NA",
            "ES Test Verdict": v_es, "Annualized Return %": f"{ann_mu:.2f}",
            "Annualized Vol %": f"{ann_vol:.2f}", "Sharpe Ratio": f"{sharpe:.3f}",
            "Performance vs 50/50": "Better" if sharpe > bench_sharpe else "Worse"
        })
    return pd.DataFrame(results)


def plot_dependence_structure(pit_df, period=None, tag="full_sample"):
    """Visualizes the joint distribution of PITs to analyze dependence."""
    df = pit_df.sort_index().loc[period[0]:period[1]] if period else pit_df.sort_index()
    if df.empty or not {"u_spx", "u_dax"}.issubset(df.columns):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"Joint PIT Dependence Structure – {tag.replace('_', ' ').title()} (n={len(df)})", fontsize=16)

    hb = ax1.hexbin(df.u_spx, df.u_dax, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax1, label="Observation Count")
    ax1.set(title="Full Distribution", xlabel="PIT (SPX)", ylabel="PIT (DAX)", aspect="equal")

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
                         levels={'VaR_990': 0.01, 'VaR_995': 0.005, 'VaR_999': 0.001},
                         models=MODELS_TO_TEST):
    """Performs VaR breach rate tests at multiple levels using pre-calculated VaR columns."""
    act = actual_returns.copy()
    act["Portfolio_Return"] = 0.5 * act["SPX_Return"] + 0.5 * act["DAX_Return"]
    rows = []

    for model in models:
        for var_suffix, alpha in levels.items():
            var_col = f"{model}_{var_suffix}"
            if var_col not in forecast_df.columns:
                continue

            # Join to align forecast with actual return for that day
            merged = forecast_df[[var_col]].join(act["Portfolio_Return"], how="inner").dropna()
            if merged.empty:
                continue

            breaches = (merged["Portfolio_Return"] < merged[var_col]).mean()
            conf_level = 1 - alpha
            
            rows.append({
                "Model": model, "ConfLevel": conf_level, "BreachRate": breaches,
                "Expected": alpha, "RelError": breaches / alpha if alpha > 0 else 0
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("[Info] No multi-level data generated."); return

    out_csv = f"{Config.ANALYSIS_DIR}/multi_level_var.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"• Multi-level VaR table saved → {out_csv}")

    # Plotting logic remains the same
    plt.figure(figsize=(8, 5))
    for mdl, grp in result_df.groupby("Model"):
        plt.plot(grp["Expected"], grp["RelError"], "o-", label=mdl)
    plt.axhline(1, ls="--", color="grey")
    plt.xscale("log"); plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Expected Breach Probability (α)")
    plt.ylabel("Relative Error (Actual / Expected)")
    plt.title("Multi-Level VaR Backtest")
    plt.legend(); plt.tight_layout()
    out_png = f"{Config.ANALYSIS_DIR}/var_curve_comparison.png"
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"• Multi-level VaR plot saved → {out_png}")


# ============================================================================ #
# 4. MAIN EXECUTION BLOCK
# ============================================================================ #
if __name__ == "__main__":
    print("=" * 80)
    print(">>> SCRIPT 04 (REVISED): BACKTESTING & PERFORMANCE ANALYSIS <<<")
    
    print("\n--- 1. Loading Data ---")
    try:
        # REVISED: Load from Parquet file
        forecasts_df = pd.read_parquet(FORECAST_RESULTS_PQ)
        actuals_df = pd.read_csv(ACTUAL_RETURNS_CSV, index_col="Date", parse_dates=True)
        pit_full_df = pd.read_csv(PIT_DATA_CSV, index_col="Date", parse_dates=True)
        print("All data files loaded successfully (Forecasts from Parquet).")
    except FileNotFoundError as e:
        print(f"[Error] Could not find a required data file: {e}")
        print("Please ensure scripts 1, 2, and 3 have been run successfully.")
        exit()

    for df in (forecasts_df, actuals_df, pit_full_df):
        if df.index.tz: df.index = df.index.tz_localize(None)

    print("\n--- 2. Defining Market Regimes ---")
    crisis_periods = {
        "CovidCrash": ("2020-02-19", "2020-04-10"),
        "BankingCrisis23": ("2023-03-08", "2023-03-31")
    }
    crisis_mask = pd.Series(False, index=forecasts_df.index)
    for name, (start, end) in crisis_periods.items():
        crisis_mask.loc[start:end] = True
        print(f"Regime '{name}' defined: {start} to {end}")

    all_regimes = {
        "Full Sample": forecasts_df.index,
        **{name: forecasts_df.loc[start:end].index for name, (start, end) in crisis_periods.items()},
        "Non-Crisis": forecasts_df.index[~crisis_mask]
    }
    print(f"Regime 'Non-Crisis' defined: All days excluding crisis periods.")
    
    print("\n--- 3. Running Backtests for Each Regime ---")
    all_results = []
    for regime_name, regime_index in all_regimes.items():
        print(f"\n... Backtesting for: {regime_name} ({len(regime_index)} obs) ...")
        forecasts_slice = forecasts_df.loc[forecasts_df.index.isin(regime_index)]
        actuals_slice = actuals_df.loc[actuals_df.index.isin(regime_index)]
        
        regime_summary_df = backtest_all_models(forecasts_slice, actuals_slice, alpha=ALPHA)
        
        if not regime_summary_df.empty:
            regime_summary_df.insert(0, "Market Regime", regime_name)
            all_results.append(regime_summary_df)

    final_summary_df = pd.concat(all_results, ignore_index=True)
    
    print("\n--- 4. Comprehensive Backtesting Summary ---")
    print(final_summary_df.to_markdown(index=False))
    
    summary_filename = f"{Config.ANALYSIS_DIR}/backtesting_summary_{datetime.now():%Y%m%d_%H%M}.csv"
    final_summary_df.to_csv(summary_filename, index=False)
    print(f"\nSaved COMPREHENSIVE backtesting summary to: {summary_filename}")

    print("\n--- 5. Generating Diagnostic Plots ---")
    plot_dependence_structure(pit_full_df, tag="full_sample")
    for name, (start, end) in crisis_periods.items():
        plot_dependence_structure(pit_full_df, period=(start, end), tag=name)

    model_to_plot = "StudentT"
    if f"{model_to_plot}_VaR" in forecasts_df.columns:
        w_cols = [f"{model_to_plot}_wSPX", f"{model_to_plot}_wDAX"]
        var_col = f"{model_to_plot}_VaR_990"
        
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
    
    print("\n--- 6. Advanced Tail Risk Analysis ---")
    multi_level_var_test(forecasts_df, actuals_df)

    print("=" * 80)
    print(">>> SCRIPT 04 FINISHED <<<")
    print("=" * 80)