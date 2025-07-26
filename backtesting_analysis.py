# =============================================================================
# SCRIPT 04 : BACKTESTING AND ANALYSIS (REV 8.1 - COMPREHENSIVE REVISION)
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import matplotlib as mpl

# ----- 0 | Plotting and Global Parameters -----------------------------------
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 110
})

# ----- 1 | Core Statistical Test Functions ----------------------------------
def kupiec_pof(hits: pd.Series, alpha=0.01):
    n = len(hits)
    n1 = hits.sum()
    p = alpha
    if n == 0:
        return n1, np.nan, np.nan, np.nan
    pi_hat = n1 / n
    if n1 in (0, n):
        ll = -2 * n * np.log((1 - p) if n1 == 0 else p)
    else:
        ll = 2 * (n1 * np.log(pi_hat / p) + (n - n1) * np.log((1 - pi_hat) / (1 - p)))
    p_val = 1 - stats.chi2.cdf(ll, 1)
    return n1, ll, p_val, pi_hat

def christoffersen(hits: pd.Series):
    """Enhanced Christoffersen test with small sample adjustment"""
    if len(hits) < 15:  # 提高最小样本量要求
        return np.nan, np.nan
    
    try:
        trans = pd.DataFrame({"prev": hits.shift(1), "curr": hits}).dropna()
        transitions = trans.groupby(["prev", "curr"]).size()
        
        # 确保有足够的转换数据
        if len(transitions) < 3:
            return np.nan, np.nan
            
        n00 = transitions.get((0, 0), 1e-6)
        n01 = transitions.get((0, 1), 1e-6)
        n10 = transitions.get((1, 0), 1e-6)
        n11 = transitions.get((1, 1), 1e-6)
        
        total = n00 + n01 + n10 + n11
        
        # 添加先验平滑
        alpha = 0.01
        n00 = (n00 + alpha) / (1 + 4*alpha)
        n01 = (n01 + alpha) / (1 + 4*alpha)
        n10 = (n10 + alpha) / (1 + 4*alpha)
        n11 = (n11 + alpha) / (1 + 4*alpha)
        total = n00 + n01 + n10 + n11
        
        pi = (n01 + n11) / total
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        
        # 避免极端值
        pi0 = max(0.001, min(0.999, pi0))
        pi1 = max(0.001, min(0.999, pi1))
        
        ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        ll1 = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
        
        stat = 2 * (ll1 - ll0)  # 注意符号调整
        pval = 1 - stats.chi2.cdf(stat, 1)
        return stat, pval
    except Exception as e:
        print(f"Christoffersen test failed: {e}")
        return np.nan, np.nan

def es_backtest(real, es, var):
    breaches = real <= var
    n = breaches.sum()
    if n < 5:
        return np.nan, np.nan, "Too few breaches"
    exc = real[breaches] - es[breaches]
    if exc.std(ddof=1) < 1e-8:
        return np.nan, np.nan, "Zero variance"
    z = exc.mean() / (exc.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    verdict = "Pass" if p > 0.05 else ("Marginal" if p > 0.01 else "Reject")
    return z, p, verdict

# ----- 2 | Main Backtesting Function ----------------------------------------
def backtest_models(frc: pd.DataFrame, act: pd.DataFrame,
                    models=("Gaussian", "StudentT", "Clayton", "Gumbel"),
                    alpha=0.01):
    
    # 移除全局基准计算
    # act['Portfolio_Return'] = 0.5 * act['SPX_Return'] + 0.5 * act['DAX_Return']
    
    rows = []
    shift_act = act.shift(-1)  # Align: T-day forecast for T+1 realization
    
    for m in models:
        ws, wd = f"{m}_wSPX", f"{m}_wDAX"
        var_c, es_c = f"{m}_VaR", f"{m}_ES"
        
        if not all(c in frc.columns for c in (ws, wd, var_c, es_c)):
            print(f"[Skipped] {m} - missing columns")
            continue

        w = frc[[ws, wd]].dropna()
        act_r = shift_act.loc[w.index]
        
        if act_r.empty:
            print(f"[Skipped] {m} - no matching returns")
            continue
            
        # 正确应用动态权重
        port_r = pd.Series(
            (act_r["SPX_Return"] * w[ws] + act_r["DAX_Return"] * w[wd]).values,
            index=act_r.index
        )
        
        # 计算基准收益（等权重）
        bench_r = 0.5 * act_r["SPX_Return"] + 0.5 * act_r["DAX_Return"]
        
        merged = pd.DataFrame({
            "real": port_r,
            "bench": bench_r,
            "var": frc.loc[w.index, var_c],
            "es": frc.loc[w.index, es_c]
        }).dropna()
        
        if merged.empty:
            print(f"[Skipped] {m} - no valid data")
            continue


        hits = (merged["real"] <= merged["var"]).astype(int)
        n1, _, kup_p, ratio = kupiec_pof(hits, alpha)
        _, christ_p = christoffersen(hits)
        z_es, p_es, res_es = es_backtest(
            merged["real"], merged["es"], merged["var"]
        )
        
        # 新增：计算风险调整收益
        mean_return = merged["real"].mean() * 252
        vol = merged["real"].std() * np.sqrt(252)
        sharpe = mean_return / vol if vol > 0 else 0
        
        # 新增：与基准比较
        bench_sharpe = (merged["bench"].mean() * 252) / (merged["bench"].std() * np.sqrt(252))
        
        rows.append({
            "Model": m,
            "Days": len(merged),
            "Breaches": n1,
            "Expected": round(len(merged)*alpha, 1),
            "Breach Rate": f"{ratio:.4f}",
            "Kupiec p-val": f"{kup_p:.4f}",
            "Christ p-val": f"{christ_p:.4f}" if not np.isnan(christ_p) else "NA",
            "ES Result": res_es,
            "Ann. Return (%)": f"{mean_return:.2f}",
            "Ann. Vol (%)": f"{vol:.2f}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "vs Benchmark": "Better" if sharpe > bench_sharpe else "Worse"
        })
        
    return pd.DataFrame(rows)

# ----- 3 | Dependence Structure Visualization -------------------------------
def plot_dependence(data, date_range=None, title="Full Sample", file_suffix="full"):
    df = data.sort_index()
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df.loc[start:end]
    
    if df.empty or not all(c in df for c in ("u_spx", "u_dax")):
        print(f"[Warning] {title} - missing data/columns")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Dependence Structure - {title} (n={len(df)})")
    
    # Joint PIT hexbin
    hb = ax1.hexbin(
        df["u_spx"], df["u_dax"],
        gridsize=35, cmap="viridis", mincnt=1
    )
    fig.colorbar(hb, ax=ax1, label="Density")
    ax1.set(xlabel="SPX PIT", ylabel="DAX PIT", title="Joint Distribution")
    ax1.set_aspect("equal", "box")
    
    # Tail dependence scatter
    ax2.scatter(df["u_spx"], df["u_dax"], s=8, alpha=0.25, c="red")
    ax2.set(
        xlim=(0, 0.1), ylim=(0, 0.1),
        xlabel="SPX PIT", ylabel="DAX PIT",
        title="Tail Dependence (0-10% quantile)"
    )
    ax2.set_aspect("equal", "box")
    
    plt.tight_layout()
    plt.savefig(f"dependence_{file_suffix}.png", dpi=300)
    plt.close()
    print(f"Saved: dependence_{file_suffix}.png")

# ----- 4 | Main Execution --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 86)
    print(">>> SCRIPT 04 : BACKTESTING AND ANALYSIS (REV 8.1) <<<")

    # 4-1 Load data
    frc_main = pd.read_csv("garch_copula_all_results.csv", index_col="Date", parse_dates=True)
    act = pd.read_csv("spx_dax_daily_data.csv", index_col="Date", parse_dates=True)
    pit_full = pd.read_csv("copula_input_data_full.csv", index_col="Date", parse_dates=True)
    
    # 时区处理
    for df in (frc_main, act, pit_full):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    # 4-2 Run backtests
    print("\n--- VaR/ES Backtest Summary (99% confidence) ---")
    summary = backtest_models(frc_main, act)
    print(summary.to_markdown(index=False))
    
    # 保存唯一文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"backtesting_summary_{timestamp}.csv"
    summary.to_csv(summary_filename, index=False)
    print(f"Saved: {summary_filename}")

    # 4-3 Plot dependence structures
    crisis_periods = {
        "Covid_Crash": ("2020-02-19", "2020-04-07"),
        "Banking_Crisis_2023": ("2023-03-08", "2023-03-24")
    }
    
    plot_dependence(pit_full, None, "Full Out-of-Sample", "full_oos")
    for name, period in crisis_periods.items():
        plot_dependence(pit_full, period, name, name.lower())

    # 4-4 Prepare diagnostic data for StudentT model
    print("\n--- Preparing Diagnostic Data ---")
    model = "StudentT"
    ws, wd = f"{model}_wSPX", f"{model}_wDAX"
    var_col = f"{model}_VaR"
    
    if all(col in frc_main.columns for col in (ws, wd, var_col)):
        w = frc_main[[ws, wd]].dropna()
        act_shifted = act.shift(-1).loc[w.index]
        
        port_returns = act_shifted["SPX_Return"] * w[ws] + act_shifted["DAX_Return"] * w[wd]
        bench_returns = 0.5 * act_shifted['SPX_Return'] + 0.5 * act_shifted['DAX_Return']
        var_series = frc_main.loc[w.index, var_col]
        
        diag_data = pd.DataFrame({
            "returns": port_returns,
            "bench": bench_returns,
            "var": var_series
        }).dropna()
        
        # Create VaR breach plot
        breaches = diag_data["returns"] <= diag_data["var"]
        plt.figure(figsize=(14, 6))
        plt.plot(diag_data.index, diag_data["returns"], lw=0.8, label="Portfolio Returns")
        plt.plot(diag_data.index, diag_data["var"], '--', label="99% VaR")
        plt.scatter(diag_data.index[breaches], diag_data["returns"][breaches],
                    c="red", s=22, label=f"Breaches ({breaches.sum()})")
        plt.title(f"{model} Copula - VaR Backtest")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("var_breaches.png", dpi=300)
        plt.close()
        print("Saved: var_breaches.png")

        # Model diagnostics
        print("\n--- Generating Model Diagnostics ---")
        plt.figure(figsize=(14, 10))
        
        # Subplot 1: Returns vs VaR
        plt.subplot(2, 1, 1)
        plt.plot(diag_data.index, diag_data["returns"], label="Portfolio Returns")
        plt.plot(diag_data.index, diag_data["var"], 'r--', label="99% VaR")
        plt.scatter(diag_data.index[breaches], diag_data["returns"][breaches],
                    c="red", s=22, label=f"Breaches ({breaches.sum()})")
        plt.title(f"{model} Model: Returns vs VaR")
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Cumulative performance
        plt.subplot(2, 1, 2)
        cum_returns = (1 + diag_data["returns"]/100).cumprod()
        cum_bench = (1 + diag_data["bench"]/100).cumprod()
        plt.plot(cum_returns, label="Managed Portfolio")
        plt.plot(cum_bench, label="50/50 Benchmark")
        plt.title("Cumulative Performance")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("model_diagnostics.png", dpi=300)
        plt.close()
        print("Saved: model_diagnostics.png")
    else:
        print(f"Skipping diagnostics for {model} - required columns missing")

    # 4-5 Plot weight distributions
    print("\n--- Plotting Weight Distributions ---")
    plt.figure(figsize=(12, 8))
    models = ["Gaussian", "StudentT", "Clayton", "Gumbel"]
    
    for i, model in enumerate(models, 1):
        w_col = f"{model}_wSPX"
        if w_col in frc_main.columns:
            plt.subplot(2, 2, i)
            sns.histplot(frc_main[w_col], bins=30, kde=True)
            plt.title(f"{model} SPX Weight Distribution")
            plt.xlabel("SPX Weight")
            plt.grid(True)

    plt.tight_layout()
    plt.savefig("weight_distributions.png", dpi=300)
    plt.close()
    print("Saved: weight_distributions.png")
    
    # 4-6 Model comparison analysis
    print("\n--- Model Comparison Analysis ---")
    model_pairs = [("StudentT", "Gaussian"), ("StudentT", "Clayton"), ("StudentT", "Gumbel")]
    
    for model1, model2 in model_pairs:
        w_col1 = f"{model1}_wSPX"
        w_col2 = f"{model2}_wSPX"
        
        if w_col1 in frc_main.columns and w_col2 in frc_main.columns:
            common_dates = frc_main.dropna(subset=[w_col1, w_col2]).index
            weight_diff = frc_main.loc[common_dates, w_col1] - frc_main.loc[common_dates, w_col2]
            avg_diff = weight_diff.mean()
            print(f"Average SPX weight difference ({model1} vs {model2}): {avg_diff:.4f}")
            
            plt.figure(figsize=(10, 6))
            weight_diff.plot(title=f"{model1} vs {model2} SPX Weight Difference")
            plt.axhline(0, color='r', linestyle='--')
            plt.ylabel("Weight Difference")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"weight_diff_{model1}_vs_{model2}.png", dpi=300)
            plt.close()
            print(f"Saved: weight_diff_{model1}_vs_{model2}.png")

    print("\n" + "=" * 86)
    print(">>> SCRIPT 04 EXECUTION COMPLETE <<<")
    print("=" * 86 + "\n")