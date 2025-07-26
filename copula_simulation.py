# =============================================================================
# SCRIPT 03 : GARCH‑COPULA ROLLING FORECAST – REV 7 (matches GED marginals)
# =============================================================================
import os, pickle, warnings
from itertools import product
import numpy as np, pandas as pd
from scipy.stats import norm, t, kendalltau, gennorm
from scipy.linalg import cholesky
from scipy.optimize import minimize
from arch import arch_model
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.stats.diagnostic import het_arch
from math import gamma   

warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------------------------------------------------------- #
# 0. Utility helpers
# --------------------------------------------------------------------------- #
def _safe_chol(corr):
    """Cholesky with PD‑repair on the fly."""
    try:
        return cholesky(corr, lower=True)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.clip(eigval, 1e-6, None)
        corr   = eigvec @ np.diag(eigval) @ eigvec.T
        D      = np.diag(1 / np.sqrt(np.diag(corr)))
        return cholesky(D @ corr @ D, lower=True)

# ‘generalised error’ helper – SciPy’s gennorm uses β param = ν
def _ged_ppf(u, nu):
    β = np.sqrt((2**(2/nu)) * gamma(1/nu) / gamma(3/nu))
    return gennorm.ppf(u, nu, scale=β)

def _ppf(u, dist_name, df):
    return _ged_ppf(u, df) if dist_name == 'ged' else t.ppf(u, df=df)

# --------------------------------------------------------------------------- #
# 1. Copula samplers
# --------------------------------------------------------------------------- #
def gauss_copula(n, corr):
    L = _safe_chol(corr)
    z = np.random.normal(size=(n, 2)) @ L.T
    return norm.cdf(z)

def t_copula(n, corr, df):
    L = _safe_chol(corr)
    g = np.random.chisquare(df, n)
    z = np.random.normal(size=(n, 2)) @ L.T
    return t.cdf(z * np.sqrt(df / g)[:, None], df=df)

def clayton_copula(n, theta):
    """Stable Clayton sampler (Marshall–Olkin, log‑space)."""
    if theta <= 0:
        return np.random.uniform(size=(n, 2))
    g   = np.random.gamma(1.0 / theta, 1.0, n)       # Γ(k=1/θ,1)
    e1  = np.random.exponential(size=n)
    e2  = np.random.exponential(size=n)
    logu = -np.log1p(e1 / g) / theta                 # log U
    logv = -np.log1p(e2 / g) / theta
    return np.column_stack((np.exp(logu), np.exp(logv))).clip(1e-6, 1-1e-6)

def gumbel_copula(n, theta):
    """Stable Gumbel sampler (Marshall–Olkin, log‑space)."""
    if theta <= 1.0:
        return np.random.uniform(size=(n, 2))
    beta = 1.0 / theta
    s    = np.random.gamma(1.0 / beta, 1.0, n)       # shared Γ
    e    = np.random.exponential(size=(n, 2))
    logu = -e[:, 0] / (s ** beta)
    logv = -e[:, 1] / (s ** beta)
    return np.column_stack((np.exp(logu), np.exp(logv))).clip(1e-6, 1-1e-6)

def surv_clayton(n, theta):
    u, v = clayton_copula(n, theta).T
    return 1.0 - u, 1.0 - v

def surv_gumbel(n, theta):
    u, v = gumbel_copula(n, theta).T
    return 1.0 - u, 1.0 - v

# --------------------------------------------------------------------------- #
# 2. Copula parameter fitting (MLE + Kendall’s τ transform)
# --------------------------------------------------------------------------- #
def fit_t_copula(u):
    tau, _ = kendalltau(u.iloc[:,0], u.iloc[:,1])
    rho0   = np.sin(np.pi * tau / 2)
    bounds = [(-0.99, .99), (2.1, 30)]
    def nll(p):
        ρ, ν = p
        if abs(ρ) >= .99: return 1e10
        z = t.ppf(u, df=ν)
        L = _safe_chol(np.array([[1, ρ],[ρ,1]]))
        q = np.linalg.solve(L, z.T).T
        logdet = 2*np.log(L.diagonal()).sum()
        ll = t.logpdf(q, df=ν).sum(axis=1) - logdet
        return -ll.sum()
    best = minimize(nll, [rho0, 8], bounds=bounds, method='L-BFGS-B')
    ρ, ν = best.x
    return {'corr_matrix': np.array([[1, ρ],[ρ,1]]), 'df': ν}

def fit_theta(family, u):
    if family == 'Clayton':
        tau, _ = kendalltau(*u.T)
        return max(0.1, 2*tau/(1-tau+1e-9))
    if family == 'Gumbel':
        tau, _ = kendalltau(*u.T)
        return max(1.01, 1/(1-tau+1e-9))

def estimate_copulas(u_df):
    # 增加尾部依赖的保守性调整因子（0.8-0.9范围）
    TAIL_ADJUST = 0.85
    
    # Gaussian
    ρG = np.corrcoef(norm.ppf(u_df.values).T)[0,1]
    gaussian = {'corr_matrix': np.array([[1, ρG],[ρG,1]]), 'rho': ρG}
    
    # t-Copula
    t_par = fit_t_copula(u_df)
    
    # 应用尾部调整
    t_par['corr_matrix'] = np.array([[1, ρG],[ρG,1]])  # 使用高斯相关度
    t_par['df'] = max(10, t_par['df'] * 1.2)  # 增加自由度（减少尾部厚度）
    
    # Clayton/Gumbel - 减少尾部依赖
    θc = fit_theta('Clayton', u_df.values) * TAIL_ADJUST
    θg = fit_theta('Gumbel', u_df.values) * TAIL_ADJUST
    
    ρc = np.sin(np.pi*(θc/(θc+2))/2)
    ρg = np.sin(np.pi*(1-1/θg)/2)
    
    print("\n--- Adjusted Copula Parameters ---")
    print(f" Student-t ρ={t_par['corr_matrix'][0,1]:.3f}  ν={t_par['df']:.1f}")
    print(f" Clayton   θ={θc:.2f}  ρ={ρc:.3f}")
    print(f" Gumbel    θ={θg:.2f}  ρ={ρg:.3f}")
    
    return {
        'Gaussian': gaussian,
        'StudentT': t_par,
        'Clayton': {'theta': θc, 'rho': ρc},
        'Gumbel': {'theta': θg, 'rho': ρg}
    }

# --------------------------------------------------------------------------- #
# 3. Marginal GARCH (GED) fit + 1‑day forecast
# --------------------------------------------------------------------------- #
def garch_fit(series, asset):
    # 减少GARCH参数持续性 (alpha + beta)
    if asset == 'SPX':
        mdl = arch_model(series, mean='AR', lags=1,
                         vol='GARCH', p=1, o=1, q=1, dist='ged')
        res = mdl.fit(disp='off', update_freq=0)
        
        # 调整参数：减少持续性，增加新息影响
        alpha = min(0.15, res.params.get('alpha[1]', 0.1))
        beta = min(0.80, res.params.get('beta[1]', 0.8))
        res.params['alpha[1]'] = alpha * 1.2  # 增加新息影响
        res.params['beta[1]'] = beta * 0.9    # 减少持续性
        
        return res
    else:
        # DAX保持相对稳定
        mdl = arch_model(series, mean='Constant',
                         vol='GARCH', p=1, q=1, dist='ged')
        return mdl.fit(disp='off')
    
# --------------------------------------------------------------------------- #
def min_var_w(vol1, vol2, rho, floor=0.3, cap=0.7, risk_aversion=1.2):
    """Constrained risk-adjusted weights"""
    # 计算协方差矩阵
    cov = rho * vol1 * vol2
    cov_matrix = np.array([
        [vol1**2, cov],
        [cov, vol2**2]
    ])
    
    # 计算最小方差权重
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(2)
    min_var_w = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
    
    # 应用风险偏好调整
    w1 = min_var_w[0] * risk_aversion
    w1 = np.clip(w1, floor, cap)
    return w1, 1 - w1

# --------------------------------------------------------------------------- #
def one_day(date, full_df, cop_par, sims=40000):
    idx = full_df.index.get_loc(date)
    if idx == 0: return None          # skip first day
    hist = full_df.iloc[:idx]         # window until t‑1
    # marginal fits
    fit_s = garch_fit(hist['SPX_Return'], 'SPX')
    fit_d = garch_fit(hist['DAX_Return'], 'DAX')
    σs = np.sqrt(fit_s.forecast(reindex=False).variance.iloc[0,0])
    σd = np.sqrt(fit_d.forecast(reindex=False).variance.iloc[0,0])
    μs = fit_s.params.get('mu',0)
    μd = fit_d.params.get('mu',0)
    νs = fit_s.params.get('nu',1.5)
    νd = fit_d.params.get('nu',1.5)

    results = {}
    samplers = {'Gaussian':lambda n: gauss_copula(n,cop_par['Gaussian']['corr_matrix']),
                'StudentT':lambda n: t_copula(n,cop_par['StudentT']['corr_matrix'],
                                              cop_par['StudentT']['df']),
                'Clayton' :lambda n: np.column_stack(surv_clayton(n, cop_par['Clayton']['theta'])),
                'Gumbel'  :lambda n: np.column_stack(surv_gumbel (n, cop_par['Gumbel']['theta']))}

    for name, gen in samplers.items():
        # 为每个copula模型使用其特定的相关性参数
        if name == 'Gaussian':
            rho = cop_par['Gaussian']['corr_matrix'][0,1]
        elif name == 'StudentT':
            rho = cop_par['StudentT']['corr_matrix'][0,1]
        elif name == 'Clayton':
            rho = cop_par['Clayton']['rho']
        elif name == 'Gumbel':
            rho = cop_par['Gumbel']['rho']
        
        # 使用特定copula的相关性计算权重
        w_s, w_d = min_var_w(σs, σd, rho)
        
        u = gen(sims)
        zs = _ppf(u[:,0], 'ged', νs)
        zd = _ppf(u[:,1], 'ged', νd)
        r_s = μs + σs*zs
        r_d = μd + σd*zd
        port = w_s*r_s + w_d*r_d
        port = port[np.isfinite(port)]
        VaR = np.percentile(port, 1)
        ES  = port[port<=VaR].mean()
        
        results[name] = {
            'VaR': VaR,
            'ES': ES,
            'wSPX': w_s,
            'wDAX': w_d,
            'volSPX': σs,
            'volDAX': σd
        }
    
    flat = {'Date': date}
    for k, v in results.items():
        for kk, vv in v.items(): 
            flat[f'{k}_{kk}'] = vv
    
    return flat

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("\n"+"="*80)
    print(">>> SCRIPT 03 : GARCH‑COPULA ROLLING FORECAST – REV 7 <<<")
    # -------- load data -------------
    u_df = pd.read_csv("copula_input_data.csv", index_col=0, parse_dates=True)
    full = pd.read_csv("spx_dax_daily_data.csv", index_col=0, parse_dates=True)
    # -------- copula parameters -----
    cop_par = estimate_copulas(u_df)
    # -------- rolling dates ---------
    oos_start = "2020-01-02"
    dates = full.loc[oos_start:].index
    print(f"\nRolling forecast {len(dates)} days …")
    forecasts = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(one_day)(d, full, cop_par) for d in tqdm(dates)
    )
    forecasts = [f for f in forecasts if f]  # drop None (first day)
    res = pd.DataFrame(forecasts).set_index('Date').sort_index()
    
    # -------- save ------------------
    # 修复：指定UTF-8编码保存CSV
    res.to_csv("garch_copula_all_results.csv", float_format="%.6f", encoding='utf-8')
    with open("copula_params.pkl", "wb") as f: 
        pickle.dump(cop_par, f)
    
    print("Output  : garch_copula_all_results.csv")
    print("Copulas : copula_params.pkl")
    print("="*80+"\n")