# =============================================================================
# SCRIPT 02 : AUTOMATED MARGINAL MODEL SELECTION & DIAGNOSTICS  (REV 7 compact)
# =============================================================================
import os, traceback
from itertools import product

import numpy as np, pandas as pd, statsmodels.api as sm
from arch import arch_model
from scipy.stats import t, norm
from statsmodels.stats.diagnostic import het_arch

# ----------------------------- CONFIG -------------------------------------- #
DATA_FILE       = "spx_dax_daily_data.csv"
IN_SAMPLE_RATIO = 0.80
RESULTS_DIR     = "model_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

VOL_FAMILIES = {"GARCH": dict(vol="GARCH", o=0),
                "GJR"  : dict(vol="GARCH", o=1),
                "EGARCH": dict(vol="EGARCH")}

DISTRIBUTIONS = ["t", "skewt", "ged"]
PQ_GRID       = [(1, 1)]                        # 仅 (1,1) 规避冗余 α2

MEAN_SPEC = {"Constant": dict(mean="Constant"),
             "AR"      : dict(mean="AR", lags=1)}

MIN_ALPHA, MIN_LB_P = 1e-4, 0.005
# --------------------------------------------------------------------------- #

# --------------------------- helpers --------------------------------------- #
coef_sum = lambda p, pre: sum(v for k, v in p.items() if k.startswith(pre))

def leverage_report(p) -> str:
    a, g = abs(coef_sum(p, "alpha[")), abs(coef_sum(p, "gamma["))
    if a < 0.05:
        return "Leverage effect present but ARCH component negligible"
    if g < 0.05:
        return "Leverage effect: not significant"
    return f"Leverage effect: − shocks raise volatility by {g/a*100:.1f}%"

def _tail_df(res) -> float:
    for k, v in res.params.items():
        if k.lower().startswith(("nu", "eta")):
            return float(v)
    return np.inf
# --------------------------------------------------------------------------- #

def _fit_once(series, vol, dist, p, q, mean_kw):
    try:
        mdl = arch_model(series, p=p, q=q, dist=dist,
                         rescale=False, **mean_kw, **VOL_FAMILIES[vol])
        return mdl.fit(disp="off")
    except Exception:
        return None

def _passes_basic(res, series):
    a = abs(coef_sum(res.params, "alpha["))
    g = abs(coef_sum(res.params, "gamma["))
    if a < MIN_ALPHA and g >= MIN_ALPHA:
        return False
    lb = sm.stats.acorr_ljungbox(
        pd.Series(res.std_resid, index=series.index).dropna(),
        lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    return lb >= MIN_LB_P

def select_best(series, tag):
    good, all_ = [], []
    for mtag, mkw in MEAN_SPEC.items():
        for vol, dist, (p, q) in product(VOL_FAMILIES, DISTRIBUTIONS, PQ_GRID):
            res = _fit_once(series, vol, dist, p, q, mkw)
            if res is None:          # failed fit
                continue
            all_.append((res.bic, res, (mtag, vol, dist, p, q)))
            if _passes_basic(res, series):
                good.append((res.bic, res, (mtag, vol, dist, p, q)))

    src = good or all_
    bic, res, spec = min(src, key=lambda x: x[0])
    mtag, vol, dist, p, q = spec
    if not good:
        print(f"⚠ WARNING: picked {mtag}-{vol}-{dist} (no model met checks).")
    print(f"\n✔ Best marginal for {tag}: {mtag}+{vol}({p},{q})-{dist} | BIC={bic:,.2f}")
    return res, spec, bic
# --------------------------------------------------------------------------- #

def diagnostics(res, series, tag):
    p = res.params
    print(f"\n--- Parameter Estimates for {tag} ---")
    print(pd.DataFrame({
        "Coefficient": p,
        "P‑value": res.pvalues.map(lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}")
    }).to_markdown(floatfmt=".4f"))

    dist_obj = getattr(res, "distribution", None) or res.model.distribution
    name = dist_obj.name.lower()
    df  = _tail_df(res)
    print(f"Tail thickness (df): {df:.2f} → "
          f"{'heavy' if df<5 else 'moderate' if df<10 else 'near‑normal'}")
    print(leverage_report(p))
    if "beta[1]" in p:
        half = np.log(0.5) / np.log(p["beta[1]"])
        print(f"Volatility half‑life: {half:.1f} days")

    if name in {"t", "skewt", "skewstudent"}:
        theta = res.params[-dist_obj.num_params:]
        prob = (dist_obj.cdf(-5, theta) + 1 - dist_obj.cdf(5, theta)) * 100
        print(f"P(|r|>5σ): {prob:.4f}% (Normal≈{norm.sf(5)*200:.6f}%)")
    else:
        print("P(|r|>5σ): n/a for GED")

    std = pd.Series(res.std_resid, index=series.index).dropna()
    for lag in (5, 10, 20):
        lb = sm.stats.acorr_ljungbox(std, lags=[lag],
                                     return_df=True)["lb_pvalue"].iloc[0]
        print(f"L‑Box StdRes (lag {lag}): p={lb:.4f}")
    lm_f, lm_p = het_arch(std)[:2]
    print(f"ARCH‑LM: F={lm_f:.3f}, p={lm_p:.4f}")
# --------------------------------------------------------------------------- #

def calc_pit(res, series):
    std = pd.Series(res.std_resid, index=series.index).dropna()
    dist_obj = getattr(res, "distribution", None) or res.model.distribution
    theta    = res.params[-dist_obj.num_params:]
    u = dist_obj.cdf(std.to_numpy(), theta)
    return pd.Series(u, index=std.index).clip(1e-6, 1-1e-6)

def oos_mse(res, series, start_idx):
    var = res.forecast(start=start_idx).variance['h.1']
    truth = series.iloc[start_idx:]**2
    truth, var = truth.align(var, join='inner')
    return ((truth - var)**2).mean()
# --------------------------------------------------------------------------- #

def run_stage(frame, label, full_series=None, split_idx=None):
    results = {}
    for col, short in [("SPX_Return", "SPX"), ("DAX_Return", "DAX")]:
        res, spec, bic = select_best(frame[col], f"{short} ({label})")
        diagnostics(res, frame[col], f"{short} ({label})")

        with open(f"{RESULTS_DIR}/{short}_{label}.txt", "w", encoding="utf-8") as f:
            f.write(f"{'+'.join(map(str, spec))} | BIC={bic:,.2f}\n\n{res.summary()}")

        # Optional OOS evaluation (only when full_series passed)
        if full_series is not None and split_idx is not None:
            mse = oos_mse(res, full_series[col], split_idx)
            print(f"OOS Volatility MSE for {short}: {mse:.6f}")

        results[short] = res
    return results
# --------------------------------------------------------------------------- #

def main():
    print("\n" + "="*80)
    print(">>> SCRIPT 02: AUTOMATED MARGINAL MODEL ESTIMATION <<<")

    data  = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
    split = int(len(data) * IN_SAMPLE_RATIO)

    print(f"\n>>> Stage 1 | In‑Sample ({split} obs) <<<")
    res_in = run_stage(data.iloc[:split], "IS")

    pd.concat([calc_pit(res_in["SPX"], data["SPX_Return"].iloc[:split]),
               calc_pit(res_in["DAX"], data["DAX_Return"].iloc[:split])],
              axis=1, join="inner").set_axis(["u_spx", "u_dax"], axis=1)\
              .to_csv("copula_input_data.csv")
    print("✔ In‑sample PIT saved → copula_input_data.csv")

    print(f"\n>>> Stage 2 | Full Sample ({len(data)} obs) <<<")
    res_full = run_stage(data, "FS", full_series=data, split_idx=split)

    pd.concat([calc_pit(res_full["SPX"], data["SPX_Return"]),
               calc_pit(res_full["DAX"], data["DAX_Return"])],
              axis=1, join="inner").set_axis(["u_spx", "u_dax"], axis=1)\
              .to_csv("copula_input_data_full.csv")
    print("✔ Full‑sample PIT saved → copula_input_data_full.csv")

    print("\n" + "="*80)
    print("SCRIPT 02 finished successfully.")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
