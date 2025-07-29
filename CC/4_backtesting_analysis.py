#!/usr/bin/env python3
# =============================================================================
# SCRIPT 04 : BACKTESTING ‑‑ REV 8.2  (sync‑up with 01‑03 pipeline)
# =============================================================================
"""
‑ 读取 01‑03 阶段生成的:
    • garch_copula_all_results.csv   ← 组合权重 / VaR / ES
    • spx_dax_daily_data.csv         ← 实际次日收益 (单位: %)
    • copula_input_data_full.csv     ← PIT (用来画相关结构)
‑ 逐模型 (Gaussian / StudentT / Clayton / Gumbel) 做:
    • Kupiec POF, Christoffersen 独立性, ES‑backtest
    • Sharpe 与 50‑50 基准比较
‑ 生成:
    • backtesting_summary_*.csv      ← 主要表
    • dependence_*.png, var_breaches.png, weight_distributions.png, …
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import matplotlib as mpl
from datetime import datetime

# ========== 0 | 画图全局 ==========
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 110
})

ALPHA      = 0.01        # 99% VaR / ES
MODELS     = ("Gaussian", "StudentT", "Clayton", "Gumbel")
RESULT_CSV = "garch_copula_all_results.csv"
RET_CSV    = "spx_dax_daily_data.csv"
PIT_CSV    = "copula_input_data_full.csv"

# =============================================================================
# 1 | 统计检验函数
# =============================================================================
def kupiec_pof(hits: pd.Series, alpha=ALPHA):
    """Proportion‑of‑Failures test"""
    n  = len(hits)
    n1 = hits.sum()
    if n == 0:                             # safety
        return n1, np.nan, np.nan, np.nan
    p  = alpha
    pi_hat = n1 / n
    if n1 in (0, n):                       # 极端情形的 log=‑inf 处理
        ll = -2 * n * np.log((1 - p) if n1 == 0 else p)
    else:
        ll = 2 * (n1 * np.log(pi_hat / p) +
                   (n - n1) * np.log((1 - pi_hat) / (1 - p)))
    p_val = 1 - stats.chi2.cdf(ll, 1)
    return n1, ll, p_val, pi_hat


def christoffersen(hits: pd.Series):
    """Christoffersen independence test（小样本平滑）"""
    if len(hits) < 15:
        return np.nan, np.nan
    trans = pd.DataFrame({"prev": hits.shift(1), "curr": hits}).dropna()
    n00 = len(trans[(trans.prev == 0) & (trans.curr == 0)]) + 0.5
    n01 = len(trans[(trans.prev == 0) & (trans.curr == 1)]) + 0.5
    n10 = len(trans[(trans.prev == 1) & (trans.curr == 0)]) + 0.5
    n11 = len(trans[(trans.prev == 1) & (trans.curr == 1)]) + 0.5

    pi  = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)

    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00*np.log(1-pi0) + n01*np.log(pi0) + n10*np.log(1-pi1) + n11*np.log(pi1)
    stat = 2 * (ll1 - ll0)
    pval = 1 - stats.chi2.cdf(stat, 1)
    return stat, pval


def es_backtest(real, es, var):
    """(Acerbi–Székely 简化版) — ES 是否足够保守"""
    breaches = real <= var
    k = breaches.sum()
    if k < 5:
        return np.nan, np.nan, "Too‑few"
    diff = (es[breaches] - real[breaches]).values    # 应 ≥0
    t    = diff.mean() / (diff.std(ddof=1) / np.sqrt(k))
    pval = 2 * (1 - stats.t.cdf(abs(t), df=k-1))
    verdict = "Pass" if pval > 0.05 else ("Marginal" if pval > 0.01 else "Reject")
    return t, pval, verdict

# =============================================================================
# 2 | Backtest 主函数
# =============================================================================
def backtest_models(frc: pd.DataFrame, act: pd.DataFrame, alpha=ALPHA):
    act1 = act.shift(-1)                               # T 日预测 → T+1 收益
    rows = []
    for m in MODELS:
        wcols = [f"{m}_wSPX", f"{m}_wDAX"]
        vcol, ecol = f"{m}_VaR", f"{m}_ES"
        if not set(wcols + [vcol, ecol]).issubset(frc.columns):
            print(f"[Skip] {m}: 缺列")
            continue

        tmp = pd.concat([frc[wcols + [vcol, ecol]], act1[["SPX_Return", "DAX_Return"]]], axis=1).dropna()
        if tmp.empty:
            print(f"[Skip] {m}: 无重叠区间")
            continue

        pret = tmp["SPX_Return"] * tmp[wcols[0]] + tmp["DAX_Return"] * tmp[wcols[1]]
        bench = 0.5*tmp["SPX_Return"] + 0.5*tmp["DAX_Return"]

        hits = (pret <= tmp[vcol]).astype(int)
        n1, kup_stat, kup_p, brate = kupiec_pof(hits, alpha)
        chr_stat, chr_p            = christoffersen(hits)
        t_es, p_es, v_es           = es_backtest(pret, tmp[ecol], tmp[vcol])

        ann_mu  = pret.mean()*252
        ann_vol = pret.std()*np.sqrt(252)
        sharpe  = ann_mu/ann_vol if ann_vol>0 else np.nan
        bench_sharpe = bench.mean()*252 / (bench.std()*np.sqrt(252))

        rows.append({
            "Model": m,
            "Obs": len(pret),
            "Breaches": n1,
            "Expected": round(len(pret)*alpha,1),
            "BreachRate": f"{brate:.4f}",
            "Kupiec_p":   f"{kup_p:.4f}",
            "Christ_p":   f"{chr_p:.4f}" if not np.isnan(chr_p) else "NA",
            "ES_test":    v_es,
            "AnnRet%":    f"{ann_mu:.2f}",
            "AnnVol%":    f"{ann_vol:.2f}",
            "Sharpe":     f"{sharpe:.2f}",
            "vs50/50":    "Better" if sharpe>bench_sharpe else "Worse"
        })
    return pd.DataFrame(rows)

# =============================================================================
# 3 | 辅助可视化
# =============================================================================
def plot_dependence(pit, period=None, tag="full"):
    df = pit.sort_index().loc[period[0]:period[1]] if period else pit.sort_index()
    if df.empty or not {"u_spx","u_dax"}.issubset(df.columns):
        print(f"[Warn] Dependence plot {tag}: data 空")
        return

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    fig.suptitle(f"Joint PIT – {tag} (n={len(df)})")

    # panel 1 ─ 全样本 hexbin
    hb=ax1.hexbin(df.u_spx, df.u_dax, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax1, label="count")
    ax1.set(title="All", xlabel="u_spx", ylabel="u_dax", aspect="equal")

    # panel 2 ─ 下 10% tail
    q = 0.10
    mask = (df.u_spx < q) & (df.u_dax < q)        # ★ 修正：统一掩码
    ax2.scatter(df.u_spx[mask], df.u_dax[mask],
                s=8, alpha=.25, c="red")
    ax2.set(xlim=(0,q), ylim=(0,q),
            title=f"Lower‑tail (<{int(q*100)}%)",
            xlabel="u_spx", ylabel="u_dax", aspect="equal")

    plt.tight_layout()
    plt.savefig(f"dependence_{tag}.png", dpi=300)
    plt.close()


# =============================================================================
# 4 | MAIN
# =============================================================================
if __name__=="__main__":
    print("="*88, "\n>>> SCRIPT 04  –  BACKTEST & ANALYSIS  (REV 8.2) <<<")
    frc  = pd.read_csv(RESULT_CSV, index_col="Date", parse_dates=True)
    act  = pd.read_csv(RET_CSV,    index_col="Date", parse_dates=True)
    pitf = pd.read_csv(PIT_CSV,    index_col="Date", parse_dates=True)

    for df in (frc,act,pitf):
        if df.index.tz: df.index = df.index.tz_localize(None)

    # --- 4.1 Backtest table ---
    print("\n--- VaR / ES Backtest (99%) ---")
    summ = backtest_models(frc, act)
    print(summ.to_markdown(index=False))
    fn = f"backtesting_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    summ.to_csv(fn, index=False);  print("Saved:", fn)

    # --- 4.2 Dependence plots ---
    plot_dependence(pitf, tag="oos_full")
    crisis = {"CovidCrash":("2020-02-19","2020-04-07"),
              "BankCrisis23":("2023-03-08","2023-03-24")}
    for k,(s,e) in crisis.items():
        plot_dependence(pitf,(s,e),k)

    # --- 4.3 Student‑t 诊断图 ---
    m="StudentT"
    cols=[f"{m}_wSPX",f"{m}_wDAX",f"{m}_VaR"]
    if set(cols).issubset(frc.columns):
        tmp = frc[cols].dropna()
        shift = act.shift(-1).loc[tmp.index]
        ret = shift.SPX_Return*tmp[cols[0]] + shift.DAX_Return*tmp[cols[1]]
        var = tmp[cols[2]]
        breaches = ret<=var
        plt.figure(figsize=(14,6))
        plt.plot(ret,label="Return"); plt.plot(var,'r--',label="VaR99")
        plt.scatter(ret.index[breaches],ret[breaches],c="red",s=22,label=f"Breaches {breaches.sum()}")
        plt.title("Student‑t Portfolio Return vs VaR"); plt.legend()
        plt.tight_layout(); plt.savefig("var_breaches.png",dpi=300); plt.close()

    # --- 4.4 权重分布 ---
    plt.figure(figsize=(10,8))
    for i,m in enumerate(MODELS,1):
        wc=f"{m}_wSPX"
        if wc in frc.columns:
            plt.subplot(2,2,i); sns.histplot(frc[wc],bins=30,kde=True)
            plt.title(f"{m}  w_SPX"); plt.axvline(0.5,c='r',ls='--')
    plt.tight_layout(); plt.savefig("weight_distributions.png",dpi=300); plt.close()
    print("\nPlots saved: dependence_*.png  var_breaches.png  weight_distributions.png")
    print("="*88, "\n>>> SCRIPT 04  FINISHED <<<\n")
