# =============================================================================
# SCRIPT 04 : BACKTESTING AND ANALYSIS  (REV 8.1 – 全面修正)
# =============================================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import warnings, matplotlib as mpl

# ----- 0 | 画图与全局参数 ----------------------------------------------------
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "figure.dpi": 110})

# ----- 1 | 基础统计检验函数 --------------------------------------------------
def kupiec_pof(hits: pd.Series, alpha=0.01):
    n = len(hits); n1 = hits.sum(); p  = alpha
    if n == 0: return n1, np.nan, np.nan, np.nan
    pi_hat = n1 / n
    if n1 in (0, n): ll = -2 * n * np.log((1-p) if n1==0 else p)
    else:
        ll = 2 * (n1*np.log(pi_hat/p) + (n-n1)*np.log((1-pi_hat)/(1-p)))
    p_val = 1 - stats.chi2.cdf(ll, 1)
    return n1, ll, p_val, pi_hat

def christoffersen(hits: pd.Series):
    if len(hits) < 2: return np.nan, np.nan
    trans = pd.DataFrame({"p": hits.shift(1), "c": hits}).dropna()
    cts = trans.groupby(["p", "c"]).size()
    n00,n01 = cts.get((0,0),0), cts.get((0,1),0)
    n10,n11 = cts.get((1,0),0), cts.get((1,1),0)
    pi  = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11)>0 else 0.0
    pi0 = n01/(n00+n01) if (n00+n01)>0 else 0
    pi1 = n11/(n10+n11) if (n10+n11)>0 else 0
    if any(x in (0,1) for x in (pi,pi0,pi1)): return np.nan, np.nan
    ll0 = (n00+n10)*np.log(1-pi) + (n01+n11)*np.log(pi)
    ll1 = n00*np.log(1-pi0)+n01*np.log(pi0)+n10*np.log(1-pi1)+n11*np.log(pi1)
    stat = -2*(ll0-ll1); pval = 1 - stats.chi2.cdf(stat, 1)
    return stat, pval

def es_backtest(real, es, var):
    breaches = real <= var; n = breaches.sum()
    if n < 5: return np.nan, np.nan, "Too few breaches"
    exc = real[breaches] - es[breaches]
    if exc.std(ddof=1) < 1e-8: return np.nan, np.nan, "Zero‑var"
    z   = exc.mean() / (exc.std(ddof=1)/np.sqrt(n))
    p   = 2*(1-stats.norm.cdf(abs(z)))
    verdict = "Pass" if p>0.05 else ("Marginal" if p>0.01 else "Reject")
    return z, p, verdict

# ----- 2 | 主回测函数 --------------------------------------------------------
def backtest_models(frc: pd.DataFrame, act: pd.DataFrame,
                    models=("Gaussian","StudentT","Clayton","Gumbel"),
                    alpha=0.01):

    rows=[]; shift_act = act.shift(-1)         # **对齐：T 日预测对 T+1 实现**
    for m in models:
        ws, wd = f"{m}_Weight_SPX", f"{m}_Weight_DAX"
        var_c, es_c = f"{m}_VaR_99", f"{m}_ES_99"
        if not all(c in frc.columns for c in (ws,wd,var_c,es_c)):
            print(f"[跳过] {m} 缺列"); continue

        w      = frc[[ws,wd]].dropna()
        act_r  = shift_act.loc[w.index, ["SPX_Return","DAX_Return"]]
        if act_r.empty: print(f"[跳过] {m} 无匹配实际收益"); continue
        port_r = (act_r * w.values).sum(axis=1)

        MERGED = pd.DataFrame({"real": port_r,
                               "var" : frc.loc[w.index, var_c],
                               "es"  : frc.loc[w.index, es_c]}).dropna()
        if MERGED.empty: print(f"[跳过] {m} 无有效数据"); continue

        hits   = (MERGED["real"] <= MERGED["var"]).astype(int)
        n1, _, kup_p, ratio = kupiec_pof(hits, alpha)
        _, christ_p         = christoffersen(hits)
        z_es, p_es, res_es  = es_backtest(MERGED["real"],
                                          MERGED["es"],
                                          MERGED["var"])
        rows.append(dict(Model=m, Days=len(MERGED),
                         Breaches=f"{n1} / {len(MERGED)*alpha:.1f}",
                         Ratio=f"{ratio:.2f}",
                         Kupiec=f"{kup_p:.4f}",
                         Christ=f"{christ_p:.4f}",
                         ES_Z=f"{z_es:.2f}" if np.isfinite(z_es) else "NA",
                         ES_Res=res_es))
    return pd.DataFrame(rows)

# ----- 3 | 依赖结构可视化 ----------------------------------------------------
def plot_dep(data, rng=None, name="Full", suffix="full"):
    df = data.sort_index()
    if rng: df = df.loc[pd.to_datetime(rng[0]):pd.to_datetime(rng[1])]
    if df.empty or not all(c in df for c in ("u_spx","u_dax")):
        print(f"[WARN] {name} 无数据/列"); return
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    fig.suptitle(f"Dependence – {name} ({len(df)})")
    hb=ax1.hexbin(df["u_spx"],df["u_dax"],gridsize=35,cmap="viridis",mincnt=1)
    fig.colorbar(hb,ax=ax1,label="Density")
    ax1.set(xlabel="SPX PIT",ylabel="DAX PIT",title="Joint PIT")
    ax1.set_aspect("equal","box")

    ax2.scatter(df["u_spx"],df["u_dax"],s=8,alpha=.25,c="red")
    ax2.set(xlim=(0,.1),ylim=(0,.1),
            xlabel="SPX PIT",ylabel="DAX PIT",
            title="Lower‑tail zoom")
    ax2.set_aspect("equal","box")
    plt.tight_layout()
    plt.savefig(f"dependence_{suffix}.png",dpi=300); plt.close()
    print(f"-- 保存: dependence_{suffix}.png")

# ----- 4 | 主程序 -----------------------------------------------------------
if __name__ == "__main__":
    print("\n"+"="*86)
    print(">>> SCRIPT 04 : BACKTESTING AND ANALYSIS  (REV 8.1) <<<")

    # 4‑1 读数据 -------------------------------------------------------------
    frc_main = pd.read_csv("garch_copula_main_results.csv", index_col="Date",
                           parse_dates=True)
    frc_rb   = pd.read_csv("garch_copula_robustness.csv", index_col="Date",
                           parse_dates=True)
    frc      = pd.concat([frc_main, frc_rb], axis=1)
    act      = pd.read_csv("spx_dax_daily_data.csv", index_col="Date",
                           parse_dates=True)
    pit_full = pd.read_csv("copula_input_data_full.csv", index_col="Date",
                           parse_dates=True)
    for d in (frc,act,pit_full): d.index = d.index.tz_localize(None)

    # 4‑2 回测摘要 -----------------------------------------------------------
    print("\n--- VaR / ES Backtest Summary (99%) ---")
    summ = backtest_models(frc, act, models=("Gaussian","StudentT","Clayton","Gumbel"))
    print(summ.to_markdown(index=False))
    summ.to_csv("backtesting_summary_final.csv",index=False)
    print("已保存 backtesting_summary_final.csv")

    # 4‑3 依赖结构图 ---------------------------------------------------------
    crisis = {"Covid crash":("2020-02-19","2020-04-07"),
              "Banking23": ("2023-03-08","2023-03-24")}
    plot_dep(pit_full, None, "Full OOS", "full_oos")
    for n,r in crisis.items(): plot_dep(pit_full,r,n,n.lower().replace(" ","_"))

    # 4‑4 单一模型 VaR 违约图 -------------------------------------------------
    mdl="StudentT"; ws,wd=f"{mdl}_Weight_SPX",f"{mdl}_Weight_DAX"
    if all(c in frc for c in (ws,wd,f"{mdl}_VaR_99")):
        w      = frc[[ws,wd]].dropna(); act_s = act.shift(-1).loc[w.index]
        port_r = (act_s[["SPX_Return","DAX_Return"]]*w.values).sum(axis=1)
        var_s  = frc.loc[w.index, f"{mdl}_VaR_99"]
        merged = pd.DataFrame({"real":port_r, "var":var_s}).dropna()
        br     = merged["real"]<=merged["var"]
        plt.figure(figsize=(14,6))
        plt.plot(merged.index, merged["real"],lw=.8,label="Realized")
        plt.plot(merged.index, merged["var"],'--',label="99% VaR")
        plt.scatter(merged.index[br], merged["real"][br],c="red",s=22,
                    label=f"Breaches ({br.sum()})")
        plt.title(f"{mdl}‑Copula VaR Backtest"); plt.legend(); plt.tight_layout()
        plt.savefig("var_breach_plot.png",dpi=300); plt.close()
        print("已保存 var_breach_plot.png")

    print("\n"+"="*86)
    print(">>> SCRIPT 04 FINISHED <<<")
    print("="*86+"\n")
