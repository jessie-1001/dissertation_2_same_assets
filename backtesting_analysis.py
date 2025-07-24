import pandas as pd
import numpy as np
from scipy.stats import chi2
import warnings

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
    if L0 == 0 or L1 == 0: return np.nan
    stat = -2 * np.log(L0 / L1) if L0 > 0 and L1 > 0 else np.nan
    pval = 1 - chi2.cdf(stat, df=1) if not np.isnan(stat) else np.nan
    return stat, pval

def es_z_test(realized, es_series, alpha=0.01):
    """ES回测的Z统计量"""
    mask = realized <= es_series
    es_breaches = realized[mask]
    es_vals = es_series[mask]
    if len(es_breaches) == 0:
        return np.nan, "Reject"
    z = (es_breaches.mean() - es_vals.mean()) / (es_vals.std() / np.sqrt(len(es_vals)))
    # 简单通过/拒绝判断
    return z, "Pass" if abs(z) < 1.96 else "Reject"

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

        # ES回测
        es_z, es_result = es_z_test(realized, es_series, alpha=alpha)

        # Tail ratio
        if hits.sum() > 0:
            tail_ratio = (realized[hits == 1].mean()) / (es_series[hits == 1].mean())
            max_dev = (realized[hits == 1] - var_series[hits == 1]).min()
        else:
            tail_ratio = np.nan
            max_dev = np.nan

        results.append({
            "Model": model,
            "VaR Breaches": f"{n_breaches} (Exp: {expected:.1f})",
            "Kupiec p-val": kupiec_pval,
            "Christoffersen p-val": christ_pval,
            "Basel Zone": basel,
            "ES Z-stat": es_z,
            "ES Pass/Fail": es_result,
            "Tail Ratio": tail_ratio,
            "Max Deviation": max_dev
        })
    return pd.DataFrame(results)

# ----------------- 3. 加载数据并运行回测 -----------------

if __name__ == "__main__":
    # 1. 加载你的数据
    forecast_df = pd.read_csv("garch_copula_forecasts.csv", index_col=0, parse_dates=True)
    actual_df = pd.read_csv("spx_dax_daily_data.csv", index_col=0, parse_dates=True)
    models = ["Gaussian", "StudentT", "Gumbel", "Clayton"]

    # 2. 运行回测
    summary = backtest_var_es(forecast_df, actual_df, models)
    print("\n" + "="*80)
    print(">>> SCRIPT 04: GARCH-COPULA BACKTESTING SUMMARY <<<\n")
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print("\nDone. Summary table above saved to results.")
    print("="*80 + "\n")

    # 3. 保存结果
    summary.to_csv("garch_copula_backtest_summary.csv", index=False)

print(forecast_df.head(10))
print(actual_df.loc[forecast_df.index].head(10))
for m in ["Gaussian", "StudentT", "Gumbel", "Clayton"]:
    print(f"{m} VaR", forecast_df[f"{m}_VaR_99"].describe())
    print(f"{m} ES", forecast_df[f"{m}_ES_99"].describe())
    print(f"{m} Weight SPX", forecast_df[f"{m}_Weight_SPX"].describe())
    print(f"{m} Weight DAX", forecast_df[f"{m}_Weight_DAX"].describe())


