#!/usr/bin/env python3
# =============================================================================
# SCRIPT 03 : GARCH‑COPULA ROLLING FORECAST – REV 8
# =============================================================================
"""
Rolling 1‑day VaR / ES forecast using:
    • GED‑marginal GARCH(1,1) (SPX AR(1)+GJR | DAX Const+GARCH)
    • Four copulas (Gaussian / Student‑t / survival‑Clayton / survival‑Gumbel)
    • Dynamic min‑var portfolio weights (risk‑aversion & hard floor/cap)

Rev 8 fix:
    1. drop NaNs before τ / ρ estimation → ρ_G could be NaN
    2. guard against empty ‘port’ array (percentile crash)
"""
# --------------------------------------------------------------------------- #
# 0. Imports & helpers
# --------------------------------------------------------------------------- #
import os, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, t, kendalltau, gennorm
from scipy.linalg import cholesky
from scipy.optimize import minimize
from arch import arch_model
from joblib import Parallel, delayed
from tqdm import tqdm
from math import gamma

# === global knobs =========================================================
ALPHA_SCALE  = 1.00        # GARCH α 放大倍数（>1 更灵敏）
BETA_CAP     = 0.70        # GARCH β 上限（越小→记忆越短）
REFIT_FREQ   = 63          # GARCH 重新估计频率（≈1 个季度）

TAIL_ADJ     = 1.30        # Copula θ 放大（>1 加强尾依赖）
USE_SURV     = True        # True→用 survival‑Clayton / survival‑Gumbel（上尾）
WEIGHT_FLOOR = 0.25        # 组合最小 w_SPX
WEIGHT_CAP   = 0.75        # 组合最大 w_SPX
# ==========================================================================

_last_idx, _cache_spx, _cache_dax = -999, None, None    # forecast cache
# --------------------------------------------------------------------------- #

warnings.filterwarnings("ignore", category=UserWarning)

def _safe_chol(corr):
    try:
        return cholesky(corr, lower=True)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        corr = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        d    = np.diag(1/np.sqrt(np.diag(corr)))
        return cholesky(d @ corr @ d, lower=True)

# ---------- GED ppf helper --------------------------------------------------
def _ged_ppf(u, nu):
    beta = np.sqrt((2**(2/nu))*gamma(1/nu)/gamma(3/nu))
    return gennorm.ppf(u, nu, scale=beta)

def _ppf(u, dist, df):
    return _ged_ppf(u, df) if dist=="ged" else t.ppf(u, df=df)


# --------------------------------------------------------------------------- #
# 1. Copula samplers
# --------------------------------------------------------------------------- #
def gauss_copula(n, corr):
    z = np.random.randn(n, 2) @ _safe_chol(corr).T
    return norm.cdf(z)

def t_copula(n, corr, df):
    g = np.random.chisquare(df, n)
    z = np.random.randn(n, 2) @ _safe_chol(corr).T
    return t.cdf(z * np.sqrt(df / g)[:, None], df=df)

def clayton_copula(n, theta):
    if theta <= 0:
        return np.random.rand(n, 2)
    g  = np.random.gamma(1 / theta, 1, n)
    e1, e2 = np.random.exponential(size=(2, n))
    u = np.exp(-np.log1p(e1 / g) / theta)
    v = np.exp(-np.log1p(e2 / g) / theta)
    return np.column_stack((u, v)).clip(1e-6, 1 - 1e-6)

def gumbel_copula(n, theta):
    if theta <= 1:
        return np.random.rand(n, 2)
    beta = 1 / theta
    s    = np.random.gamma(1 / beta, 1, n)
    e    = np.random.exponential(size=(n, 2))
    u = np.exp(-e[:, 0] / (s ** beta))
    v = np.exp(-e[:, 1] / (s ** beta))
    return np.column_stack((u, v)).clip(1e-6, 1 - 1e-6)

def surv_clayton(n, θ):   # upper‑tail dependence
    u, v = clayton_copula(n, θ).T
    return 1 - u, 1 - v

def surv_gumbel(n, θ):
    u, v = gumbel_copula(n, θ).T
    return 1 - u, 1 - v

# --------------------------------------------------------------------------- #
# 2. Copula parameter estimation
# --------------------------------------------------------------------------- #
def fit_t_copula(u_df: pd.DataFrame):
    """MLE for bivariate Student‑t copula (ρ, ν)."""
    u_df = u_df.dropna()
    tau, _ = kendalltau(u_df.iloc[:, 0], u_df.iloc[:, 1])
    rho0   = np.sin(np.pi * tau / 2)

    u_arr  = u_df.to_numpy()                    # <<< 转成 ndarray 仅一次
    bounds = [(-0.99, .99), (2.1, 40)]

    def nll(p):
        ρ, ν = p
        if abs(ρ) >= .99:
            return 1e6
        L   = _safe_chol(np.array([[1, ρ], [ρ, 1]]))
        z   = t.ppf(u_arr.clip(1e-6, 1 - 1e-6), df=ν)   # Nx2 ndarray
        q   = np.linalg.solve(L, z.T).T
        ll  = t.logpdf(q, df=ν).sum(axis=1) - 2*np.log(np.diag(L)).sum()
        return -ll.sum()

    ρ_hat, ν_hat = minimize(nll, [rho0, 10],
                            bounds=bounds, method="L-BFGS-B").x
    return {"corr_matrix": np.array([[1, ρ_hat], [ρ_hat, 1]]), "df": ν_hat}

def _theta_from_tau(fam, u_df):
    tau, _ = kendalltau(u_df.iloc[:, 0], u_df.iloc[:, 1])
    if fam == "Clayton":
        return max(0.1, 2 * tau / (1 - tau + 1e-9))
    if fam == "Gumbel":
        return max(1.01, 1 / (1 - tau + 1e-9))

# --------------------------------------------------------------------------- #
# 2. Copula parameter estimation (✱ re‑write)                                #
# --------------------------------------------------------------------------- #
def estimate_copulas(u_raw: pd.DataFrame):
    """
    输出 dict:  Gaussian / StudentT / Clayton / Gumbel
    Clayton‧Gumbel 的 θ 乘以全局 TAIL_ADJ，可通过 USE_SURV
    决定是否改用 survival‑copula（上尾依赖）。
    """
    u = u_raw.dropna()
    if u.empty:
        raise ValueError("PIT 序列为空，无法估计 copula 参数。")

    z   = norm.ppf(u.clip(1e-6, 1-1e-6).values)
    rhoG = np.corrcoef(z.T)[0, 1] if np.isfinite(z).all() else 0.0
    gaussian = {"corr_matrix": np.array([[1, rhoG], [rhoG, 1]]), "rho": rhoG}

    # ---- Student‑t ---------------------------------------------------------
    t_par = fit_t_copula(u)
    t_par["df"] = min(30, t_par["df"])                      # 上限 30

    # ---- Clayton / Gumbel --------------------------------------------------
    θc = _theta_from_tau("Clayton", u) * TAIL_ADJ
    θg = _theta_from_tau("Gumbel",  u) * TAIL_ADJ
    ρc = np.sin(np.pi * (θc / (θc + 2)) / 2)                # 近似 ρ
    ρg = np.sin(np.pi * (1 - 1 / θg) / 2)

    print("\n--- Adjusted Copula Parameters ---")
    print(f" Gaussian   ρ = {rhoG:.3f}")
    print(f" Student‑t  ρ = {t_par['corr_matrix'][0,1]:.3f}  ν = {t_par['df']:.1f}")
    print(f" Clayton    θ = {θc:.2f}  ρ ≈ {ρc:.3f}")
    print(f" Gumbel     θ = {θg:.2f}  ρ ≈ {ρg:.3f}")

    return {
        "Gaussian": gaussian,
        "StudentT": t_par,
        "Clayton" : {"theta": θc, "rho": ρc},
        "Gumbel"  : {"theta": θg, "rho": ρg},
    }

# --------------------------------------------------------------------------- #
# 3. Marginal GARCH fit (✱ re‑write)                                         #
# --------------------------------------------------------------------------- #
def garch_fit(series, asset):
    if asset == "SPX":
        mdl = arch_model(series, mean="AR", lags=1,
                         vol="GARCH", p=1, o=1, q=1, dist="ged")
    else:
        mdl = arch_model(series, mean="Constant",
                         vol="GARCH", p=1, q=1, dist="ged")

    res = mdl.fit(disp="off")

    # —— 软约束：调 α、截 β ——
    if "alpha[1]" in res.params:
        res.params["alpha[1]"] *= ALPHA_SCALE
    if "beta[1]" in res.params:
        res.params["beta[1]"]  = min(res.params["beta[1]"], BETA_CAP)
    return res


# --------------------------------------------------------------------------- #
# 4. Single‑day simulation block (✱ re‑write)                                #
# --------------------------------------------------------------------------- #
def one_day(date, full_df, cop, sims=40_000):
    global _last_idx, _cache_spx, _cache_dax
    idx = full_df.index.get_loc(date)
    if idx == 0:
        return None

    # —— 每 REFIT_FREQ 天重新估计一次 marginals ——
    if (idx - _last_idx) >= REFIT_FREQ or _cache_spx is None:
        hist          = full_df.iloc[:idx]
        _cache_spx    = garch_fit(hist["SPX_Return"], "SPX")
        _cache_dax    = garch_fit(hist["DAX_Return"], "DAX")
        _last_idx     = idx
    fit_s, fit_d = _cache_spx, _cache_dax

    σs = np.sqrt(fit_s.forecast(reindex=False).variance.iloc[0, 0])
    σd = np.sqrt(fit_d.forecast(reindex=False).variance.iloc[0, 0])
    μs, μd = fit_s.params.get("mu", 0), fit_d.params.get("mu", 0)
    νs, νd = fit_s.params.get("nu", 1.5), fit_d.params.get("nu", 1.5)

    samplers = {
        "Gaussian": lambda n: gauss_copula(n, cop["Gaussian"]["corr_matrix"]),
        "StudentT": lambda n: t_copula(n,
                        cop["StudentT"]["corr_matrix"], cop["StudentT"]["df"]),
        "Clayton" : (lambda n:
            np.column_stack(surv_clayton(n, cop["Clayton"]["theta"]))
            if USE_SURV else clayton_copula(n, cop["Clayton"]["theta"])),
        "Gumbel"  : (lambda n:
            np.column_stack(surv_gumbel(n,  cop["Gumbel"]["theta"]))
            if USE_SURV else gumbel_copula(n,  cop["Gumbel"]["theta"]))
    }

    out = {}
    for name, gen in samplers.items():
        rho = {
            "Gaussian": cop["Gaussian"]["rho"],
            "StudentT": cop["StudentT"]["corr_matrix"][0, 1],
            "Clayton" : cop["Clayton"]["rho"],
            "Gumbel"  : cop["Gumbel"]["rho"],
        }[name]

        w_s, w_d = min_var_w(σs, σd, rho)          # floor / cap 来自全局常量
        u   = gen(sims)
        port= w_s*(μs + σs*_ppf(u[:,0],"ged",νs)) + \
              w_d*(μd + σd*_ppf(u[:,1],"ged",νd))
        port = port[np.isfinite(port)]
        if port.size < 50:
            VaR = ES = np.nan
        else:
            VaR = np.percentile(port, 1)
            ES  = port[port <= VaR].mean()

        out.update({
            f"{name}_VaR"   : VaR,
            f"{name}_ES"    : ES,
            f"{name}_wSPX"  : w_s,
            f"{name}_wDAX"  : w_d,
            f"{name}_volSPX": σs,
            f"{name}_volDAX": σd
        })

    out["Date"] = date
    return out
# --------------------------------------------------------------------------- #
# 5 |  util for weights
# --------------------------------------------------------------------------- #
def min_var_w(vol1, vol2, rho,
              floor=WEIGHT_FLOOR,   # ← 用全局常量
              cap=WEIGHT_CAP):      # ← 用全局常量
    Σ = np.array([[vol1**2, rho*vol1*vol2],
                  [rho*vol1*vol2, vol2**2]])
    inv = np.linalg.inv(Σ)
    w   = inv @ np.ones(2) / (np.ones(2) @ inv @ np.ones(2))
    w1  = np.clip(w[0], floor, cap)
    return w1, 1 - w1
# --------------------------------------------------------------------------- #
if __name__=="__main__":
    print("="*80,"\n>>> SCRIPT 03 : GARCH‑COPULA ROLLING FORECAST <<<")
    u_df=pd.read_csv("copula_input_data.csv",index_col=0,parse_dates=True)
    full=pd.read_csv("spx_dax_daily_data.csv",index_col=0,parse_dates=True)
    cop=estimate_copulas(u_df)

    oos_start="2020-01-02"
    dates=full.loc[oos_start:].index
    print(f"\nRolling forecast for {len(dates)} trading days …")

    forecasts=Parallel(n_jobs=-1,backend="loky")(
        delayed(one_day)(d,full,cop) for d in tqdm(dates))
    res=pd.DataFrame([f for f in forecasts if f]).set_index("Date").sort_index()

    res.to_csv("garch_copula_all_results.csv",float_format="%.6f",encoding="utf-8")
    with open("copula_params.pkl","wb") as fh: pickle.dump(cop,fh)
    print("\nSaved → garch_copula_all_results.csv  &  copula_params.pkl")
    print("="*80)