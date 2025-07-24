# =============================================================================
# SCRIPT 04: BACKTESTING GARCH-COPULA (改进版)
# =============================================================================
import pandas as pd
import numpy as np
from scipy.stats import chi2, norm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------- 1. VaR和ES回测统计 -----------------
def kupiec_pof_test(hits, alpha=0.01):
    """Kupiec比例检验（POF）"""
    n = len(hits)
    n1 = hits.sum()
    if n == 0: return 0, np.nan, 1.0
    p = alpha
    pi_hat = n1 / n
    if n1 == 0:
        log_lr = -2 * (n * np.log(1 - p))
    elif np.isclose(pi_hat, 1.0):
        log_lr = -2 * (n * np.log(p))
    else:
        log_lr = 2 * (n1 * np.log(pi_hat / p) + (n - n1) * np.log((1 - pi_hat) / (1 - p)))
    p_value = 1 - chi2.cdf(log_lr, df=1)
    return n1, log_lr, p_value

def christoffersen_test(hits):
    """Christoffersen独立性检验"""
    n00 = n01 = n10 = n11 = 0
    prev = hits.iloc[0]
    for cur in hits.iloc[1:]:
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        elif prev == 1 and cur == 1: n11 += 1
        prev = cur
    n0 = n00 + n01
    n1 = n10 + n11
    pi0 = n01 / n0 if n0 else 0
    pi1 = n11 / n1 if n1 else 0
    pi = (n01 + n11) / (n0 + n1)
    L0 = ((1 - pi)**(n00 + n10)) * (pi**(n01 + n11))
    L1 = (1 - pi0)**n00 * pi0**n01 * (1 - pi1)**n10 * pi1**n11
    if L0 == 0 or L1 == 0: return np.nan, np.nan
    stat = -2 * np.log(L0 / L1) if L0 > 0 and L1 > 0 else np.nan
    pval = 1 - chi2.cdf(stat, df=1) if not np.isnan(stat) else np.nan
    return stat, pval

def bayer_scheule_test(realized, es_series, var_series, alpha=0.01):
    """Bayer-Scheule ES回测方法"""
    # 筛选突破点
    breaches = realized <= var_series
    n_breaches = breaches.sum()
    
    if n_breaches < 10:
        return np.nan, np.nan, "Insufficient Data"
    
    # 计算实际尾部损失
    realized_losses = -realized[breaches]
    predicted_es = -es_series[breaches]
    
    # 计算偏差
    bias = realized_losses.mean() - predicted_es.mean()
    
    # 标准化偏差
    std_dev = realized_losses.std() / np.sqrt(n_breaches)
    z_score = bias / std_dev
    
    # 显著性检验
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    result = "Pass" if p_value > 0.05 else "Reject"
    
    return z_score, p_value, result

# ----------------- 2. 回测主程序 -----------------
def backtest_var_es(forecast_df, actual_df, models, alpha=0.01):
    """对各Copula模型的VaR/ES做回测评估"""
    results = []
    for model in models:
        var_col = f"{model}_VaR_99"
        es_col = f"{model}_ES_99"
        # 以SPX和DAX的组合真实收益为实际收益
        w_spx = forecast_df[f"{model}_Weight_SPX"]
        w_dax = forecast_df[f"{model}_Weight_DAX"]
        r_spx = actual_df.loc[forecast_df.index, "SPX_Return"]
        r_dax = actual_df.loc[forecast_df.index, "DAX_Return"]
        realized = w_spx * r_spx + w_dax * r_dax

        var_series = forecast_df[var_col]
        es_series = forecast_df[es_col]
        hits = (realized <= var_series).astype(int)

        n_breaches, kupiec_stat, kupiec_pval = kupiec_pof_test(hits, alpha=alpha)
        christ_stat, christ_pval = christoffersen_test(hits)
        expected = len(hits) * alpha

        # Basel zone
        if n_breaches <= expected + 4 and n_breaches >= expected - 4:
            basel = "Green Zone"
        elif n_breaches <= expected + 8:
            basel = "Yellow Zone"
        else:
            basel = "Red Zone"

        # ES回测 - 使用改进方法
        es_z, es_pval, es_result = bayer_scheule_test(
            realized, es_series, var_series, alpha=alpha)
        
        # 尾部损失比率
        breach_losses = realized[hits == 1]
        tail_ratio = breach_losses.mean() / es_series[hits == 1].mean() if hits.sum() > 0 else np.nan
        max_dev = breach_losses.min() if hits.sum() > 0 else np.nan

        results.append({
            "Model": model,
            "VaR Breaches": f"{n_breaches} (Exp: {expected:.1f})",
            "Kupiec p-val": kupiec_pval,
            "Christoffersen p-val": christ_pval,
            "Basel Zone": basel,
            "ES Z-stat": es_z,
            "ES p-val": es_pval,
            "ES Result": es_result,
            "Tail Ratio": tail_ratio,
            "Max Deviation": max_dev
        })
    return pd.DataFrame(results)

# ----------------- 3. 压力测试函数 -----------------
def stress_test(forecast_df, actual_df, crisis_periods):
    """对危机期间进行专项压力测试"""
    results = []
    for name, period in crisis_periods.items():
        crisis_data = forecast_df.loc[period[0]:period[1]]
        if crisis_data.empty:
            continue
            
        # 计算实际组合收益
        w_spx = crisis_data['StudentT_Weight_SPX']
        w_dax = crisis_data['StudentT_Weight_DAX']
        r_spx = actual_df.loc[crisis_data.index, "SPX_Return"]
        r_dax = actual_df.loc[crisis_data.index, "DAX_Return"]
        realized = w_spx * r_spx + w_dax * r_dax
        
        # 计算VaR突破
        var_series = crisis_data['StudentT_VaR_99']
        breaches = realized <= var_series
        n_breaches = breaches.sum()
        expected = len(crisis_data) * 0.01
        
        # 计算尾部损失
        breach_losses = realized[breaches]
        avg_breach = breach_losses.mean() if not breach_losses.empty else 0
        max_breach = breach_losses.min() if not breach_losses.empty else 0
        
        results.append({
            "Crisis Period": name,
            "Start": period[0],
            "End": period[1],
            "Days": len(crisis_data),
            "Breaches": n_breaches,
            "Expected Breaches": expected,
            "Breach Rate (%)": n_breaches/len(crisis_data)*100,
            "Avg Breach Size": avg_breach,
            "Max Breach Size": max_breach
        })
    
    return pd.DataFrame(results)

# ----------------- 4. 依赖结构可视化 -----------------
def plot_dependence_structure(data, period_name="Full Sample", file_suffix=""):
    """可视化Copula依赖结构"""
    plt.figure(figsize=(14, 10))
    
    # 选择子样本
    if period_name != "Full Sample":
        data = data.loc[period_name[0]:period_name[1]]
    
    # 转换到Copula空间
    u_spx = data['u_spx']
    u_dax = data['u_dax']
    
    # 左尾依赖（损失协同）
    plt.subplot(2, 2, 1)
    plt.hexbin(u_spx, u_dax, gridsize=50, cmap='Blues', mincnt=1)
    plt.title(f"Joint Distribution ({period_name})")
    plt.xlabel("SPX PIT")
    plt.ylabel("DAX PIT")
    
    # 左尾聚焦
    plt.subplot(2, 2, 2)
    tail_data = data[(data['u_spx'] < 0.1) | (data['u_dax'] < 0.1)]
    if not tail_data.empty:
        plt.hexbin(tail_data['u_spx'], tail_data['u_dax'], gridsize=30, cmap='Reds', mincnt=1)
    plt.title("Left Tail Dependence")
    plt.xlabel("SPX PIT")
    plt.ylabel("DAX PIT")
    
    # Kendall tau滚动窗口 - 修复计算方式
    plt.subplot(2, 1, 2)
    window = 252  # 1年窗口
    
    # 正确计算滚动Kendall tau
    rolling_tau = u_spx.rolling(window).apply(
        lambda x: u_dax.loc[x.index].corr(x, method='kendall'),
        raw=False
    )
    
    rolling_tau.plot()
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title(f"Rolling Kendall Tau ({window}-day window)")
    plt.ylabel("Dependence")
    
    plt.tight_layout()
    plt.savefig(f"dependence_structure_{file_suffix}.png")
    plt.close()
    print(f"Saved dependence plot: dependence_structure_{file_suffix}.png")

# ----------------- 5. 主执行块 -----------------
if __name__ == "__main__":
    # 1. 加载数据
    forecast_df = pd.read_csv("garch_copula_forecasts_improved.csv", index_col=0, parse_dates=True)
    actual_df = pd.read_csv("spx_dax_daily_data.csv", index_col=0, parse_dates=True)
    models = ["Gaussian", "StudentT", "Gumbel", "Clayton"]
    
    # 2. 运行回测
    summary = backtest_var_es(forecast_df, actual_df, models)
    print("\n" + "="*80)
    print(">>> SCRIPT 04: GARCH-COPULA BACKTESTING SUMMARY (改进版) <<<\n")
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print("\nDone. Summary table above saved to results.")
    print("="*80 + "\n")
    summary.to_csv("garch_copula_backtest_summary_improved.csv", index=False)
    
    # 3. 压力测试
    crisis_periods = {
        "COVID-19": ("2020-02-19", "2020-04-07"),
        "Russia-Ukraine": ("2022-02-16", "2022-03-07"),
        "Inflation Surge": ("2022-06-01", "2022-09-30")
    }
    stress_results = stress_test(forecast_df, actual_df, crisis_periods)
    print("\nStress Test Results:")
    print(stress_results.to_markdown(index=False))
    stress_results.to_csv("stress_test_results.csv", index=False)
    
    # 4. 可视化依赖结构
    copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True)
    # 全样本
    plot_dependence_structure(copula_input_data, "Full Sample", "full")
    # COVID期间
    plot_dependence_structure(copula_input_data, ("2020-02-01", "2020-04-30"), "covid")