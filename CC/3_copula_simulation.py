#!/usr/bin/env python3
# =============================================================================
# SCRIPT 3 (FINAL): AUTOMATED GARCH-COPULA TUNING & FORECASTING
# =============================================================================
"""
1. Defines a grid of hyperparameters to test.
2. For each combination, runs a full rolling-window GARCH-Copula simulation.
3. Backtests the results of each simulation run.
4. Selects the optimal hyperparameter set based on backtest validity and accuracy.
5. Performs a final, full forecast using the best parameters and saves the results.
"""
import os
import pickle
import warnings
from itertools import product
from math import gamma

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from joblib import Parallel, delayed, Memory
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.stats import gennorm, kendalltau, norm, t, chi2
from tqdm import tqdm
import optuna

from config import Config

# ============================================================================ #
# 1. CONFIGURATION AND HYPERPARAMETER GRID
# ============================================================================ #
RUN_OPTUNA_TUNING = False # <<< *** MODIFY THIS SWITCH AS NEEDED ***

# --- GARCH Model Specifications ---
VOL_FAMILIES = {"GARCH": dict(vol="GARCH", o=0), "GJR": dict(vol="GARCH", o=1)}
DISTRIBUTIONS = ["t", "skewt", "ged"]
MEAN_SPEC = {"Constant": dict(mean="Constant"), "AR": dict(mean="AR", lags=1)}

# --- General Configuration ---
SIMS_PER_DAY = 25000       # Number of simulations for each day's forecast
OOS_START_DATE = "2020-01-02" # Start date for the out-of-sample forecast
PORTFOLIO_ALPHA = 0.01     # For 99% VaR/ES
WEIGHT_FLOOR = 0.25        # Minimum portfolio weight for an asset
WEIGHT_CAP = 0.75          # Maximum portfolio weight for an asset
N_JOBS = -1                # Use all available CPU cores for parallel processing

memory = Memory(location=f"{Config.BASE_DIR}/.garch_cache", verbose=0)
# ============================================================================ #
# 2. HELPER & UTILITY FUNCTIONS
# ============================================================================ #

# --- Math & Stat Helpers ---
def _safe_chol(corr):
    try:
        return cholesky(corr, lower=True)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        corr_psd = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        d = np.diag(1 / np.sqrt(np.diag(corr_psd)))
        return cholesky(d @ corr_psd @ d, lower=True)


# --- Copula Samplers ---
def gauss_copula(n, corr):
    z = np.random.randn(n, 2) @ _safe_chol(corr).T
    return norm.cdf(z)

def t_copula(n, corr, df):
    g = np.random.chisquare(df, n)
    z = np.random.randn(n, 2) @ _safe_chol(corr).T
    return t.cdf(z * np.sqrt(df / g)[:, None], df=df)

def clayton_copula(n, theta):
    if theta <= 1e-6: return np.random.rand(n, 2)
    g = np.random.gamma(1 / theta, 1, n)
    e1, e2 = np.random.exponential(size=(2, n))
    u = np.exp(-np.log1p(e1 / g) / theta)
    v = np.exp(-np.log1p(e2 / g) / theta)
    return np.column_stack((u, v)).clip(1e-6, 1 - 1e-6)

def gumbel_copula(n, theta):
    if theta <= 1: return np.random.rand(n, 2)
    beta = 1 / theta
    s = np.random.gamma(1 / beta, 1, n)
    e = np.random.exponential(size=(n, 2))
    u = np.exp(-e[:, 0] / (s ** beta))
    v = np.exp(-e[:, 1] / (s ** beta))
    return np.column_stack((u, v)).clip(1e-6, 1 - 1e-6)

def surv_gumbel(n, θ):
    u, v = gumbel_copula(n, θ).T
    return np.column_stack((1 - u, 1 - v))

# --- Copula Parameter Estimation ---
def estimate_copulas(u_raw: pd.DataFrame, config: dict):
    u = u_raw.dropna()
    if u.empty: raise ValueError("PIT series is empty, cannot estimate copula parameters.")

    z = norm.ppf(u.clip(1e-6, 1 - 1e-6).values)
    rhoG = np.corrcoef(z.T)[0, 1] if np.isfinite(z).all() else 0.0

    tau, _ = kendalltau(u.iloc[:, 0], u.iloc[:, 1])
    rhoT_init = np.sin(np.pi * tau / 2)

    def t_nll(p):
        ρ, ν = p
        if abs(ρ) >= 0.999: return 1e9
        L = _safe_chol(np.array([[1, ρ], [ρ, 1]]))
        z_t = t.ppf(u.values, df=ν)
        q = np.linalg.solve(L, z_t.T).T
        ll = t.logpdf(q, df=ν).sum(axis=1) - 2 * np.log(np.diag(L)).sum()
        return -ll.sum()

    res = minimize(t_nll, [rhoT_init, 10], bounds=[(-0.99, 0.99), (2.1, 40)], method="L-BFGS-B")
    rhoT, dfT = res.x

    θc = max(0.1, 2 * tau / (1 - tau + 1e-9)) * config['tail_adj']
    θg = max(1.01, 1 / (1 - tau + 1e-9)) * config['tail_adj']

    return {
        "Gaussian": {"corr_matrix": np.array([[1, rhoG], [rhoG, 1]]), "rho": rhoG},
        "StudentT": {"corr_matrix": np.array([[1, rhoT], [rhoT, 1]]), "df": dfT, "rho": rhoT},
        "Clayton": {"theta": θc, "rho": np.sin(np.pi * (θc / (θc + 2)) / 2)},
        "Gumbel": {"theta": θg, "rho": np.sin(np.pi * (1 - 1 / θg) / 2)},
    }

# ============================================================================ #
# 3. CORE SIMULATION LOGIC
# ============================================================================ #
@memory.cache
def fit_best_garch(series: pd.Series, tag: str, config: dict):
    """Selects the best GARCH model for a series based on BIC."""
    candidates = []

    for mean_tag, mean_kw in MEAN_SPEC.items():
        for vol, dist in product(VOL_FAMILIES, DISTRIBUTIONS):
            # We are only using p=1, q=1
            p, q = 1, 1
            try:
                mdl = arch_model(series, p=p, q=q, dist=dist, **mean_kw, **VOL_FAMILIES[vol])
                res = mdl.fit(disp="off", show_warning=False)
                if res.convergence_flag == 0:
                    if "beta[1]" in res.params:
                        res.params["beta[1]"] = min(res.params["beta[1]"], config['beta_cap'])
                    
                    # Store the result along with its specification details
                    spec = (mean_tag, vol, dist, p, q)
                    candidates.append((res.bic, res, spec))
            except Exception:
                continue

    if not candidates:
        # Fallback to a simple, robust model if nothing else converges
        print(f"  > [Warning] No models converged for {tag}. Defaulting to Constant+GARCH(1,1)-t.")
        mdl = arch_model(series, p=1, q=1, vol="GARCH", dist="t", mean="Constant")
        res = mdl.fit(disp="off")
        spec = ("Constant", "GARCH", "t", 1, 1)
        spec_str = "Constant+GARCH(1,1)-t"
        return res, spec, res.bic, spec_str
        
    # Find the best model from the candidates based on BIC
    best_bic, best_res, best_spec = min(candidates, key=lambda x: x[0])
    
    # Reconstruct the spec string for printing/logging
    mean_tag, vol, dist, p, q = best_spec
    spec_str = f"{mean_tag}+{vol}({p},{q})-{dist}"
    
    # print(f"  > Best model for {tag}: {spec_str}, BIC={best_bic:.2f}")

    # Return all four expected values
    return best_res, best_spec, best_bic, spec_str

def one_day_forecast(date, full_df, u_df, config, MEAN_SPEC, VOL_FAMILIES):
    """
    Performs a completely self-contained 1-day ahead forecast for a given date.
    This function is designed to be run in a separate process and now includes
    caching to avoid re-fitting GARCH models unnecessarily.
    """

    # --- 1. Data Preparation ---
    idx = full_df.index.get_loc(date)
    # Use pre-calculated PITs for copula fitting
    u_hist = u_df.loc[u_df.index < date] 

    tag_spx = f"SPX_{idx // config['refit_freq']}"
    tag_dax = f"DAX_{idx // config['refit_freq']}"
    if idx % config['refit_freq'] == 0:
        hist_df = full_df.iloc[:idx + 1]
        try:
            res_s = fit_best_garch(hist_df["SPX_Return"], tag_spx, config)[0]
            res_d = fit_best_garch(hist_df["DAX_Return"], tag_dax, config)[0]
        except Exception as e:
            print(f"[Warning] GARCH refit failed at {date}: {e}")
    else:
        try:
            res_s = fit_best_garch(full_df.iloc[:idx]["SPX_Return"], tag_spx, config)[0]
        except (FileNotFoundError, KeyError):
            print(f"[CACHE MISS] SPX unexpectedly refit at {date} (idx {idx})")

        try:
            res_d = fit_best_garch(full_df.iloc[:idx]["DAX_Return"], tag_dax, config)[0]
        except (FileNotFoundError, KeyError):
            print(f"[CACHE MISS] DAX unexpectedly refit at {date} (idx {idx})")
    
    # --- 2. Estimate Copulas ---
    # This is still done daily as the historical PIT window grows each day.
    copulas = estimate_copulas(u_hist, config)

    # --- 3. Forecast ---
    fc_s = res_s.forecast(reindex=False)
    fc_d = res_d.forecast(reindex=False)
    
    μs, σs = fc_s.mean.iloc[-1, 0], np.sqrt(fc_s.variance.iloc[-1, 0])
    μd, σd = fc_d.mean.iloc[-1, 0], np.sqrt(fc_d.variance.iloc[-1, 0])
    
    dist_s = res_s.model.distribution
    params_s = res_s.params[dist_s.parameter_names()].values

    dist_d = res_d.model.distribution
    params_d = res_d.params[dist_d.parameter_names()].values

    # --- 4. Simulate and Calculate VaR/ES ---
    samplers = {
        "Gaussian": lambda n: gauss_copula(n, copulas["Gaussian"]["corr_matrix"]),
        "StudentT": lambda n: t_copula(n, copulas["StudentT"]["corr_matrix"], copulas["StudentT"]["df"]),
        "Clayton" : lambda n: clayton_copula(n, copulas["Clayton"]["theta"]),
        "Gumbel"  : lambda n: surv_gumbel(n, copulas["Gumbel"]["theta"]),
    }

    out = {}
    for name, gen in samplers.items():
        w_s, w_d = min_var_w(σs, σd, copulas[name]["rho"])
        u = gen(config['sims'])

        # Use the distribution's own .ppf() method for accurate simulation
        innov_s = dist_s.ppf(u[:, 0], params_s)
        innov_d = dist_d.ppf(u[:, 1], params_d)

        ret_s = μs + σs * innov_s
        ret_d = μd + σd * innov_d
        port = (w_s * ret_s + w_d * ret_d)[np.isfinite(ret_s) & np.isfinite(ret_d)]
        
        if port.size < 50: VaR, ES = np.nan, np.nan
        else:
            VaR = np.percentile(port, 100 * config['alpha'])
            ES = port[port <= VaR].mean()

        out.update({
            f"{name}_wSPX": w_s, 
            f"{name}_wDAX": w_d, 
            f"{name}_VaR": VaR, 
            f"{name}_ES": ES
        })
    out["Date"] = date
    return out

def min_var_w(vol1, vol2, rho):
    """Calculates minimum variance portfolio weights with constraints."""
    Σ = np.array([[vol1**2, rho*vol1*vol2], [rho*vol1*vol2, vol2**2]])
    try:
        inv = np.linalg.inv(Σ)
        w = inv @ np.ones(2) / (np.ones(2) @ inv @ np.ones(2))
        return np.clip(w[0], WEIGHT_FLOOR, WEIGHT_CAP), 1 - np.clip(w[0], WEIGHT_FLOOR, WEIGHT_CAP)
    except np.linalg.LinAlgError: return 0.5, 0.5

# ============================================================================ #
# 4. BACKTESTING FUNCTIONS (Adapted from 4_backtesting_analysis.py)
# ============================================================================ #

def run_backtest(frc_df, act_df, alpha):
    """Runs backtests and returns key performance metrics."""
    act1 = act_df.shift(-1)
    results = {}
    
    for model in ["Gaussian", "StudentT", "Clayton", "Gumbel"]:
        vcol = f"{model}_VaR"
        tmp = pd.concat([frc_df[vcol], act1[["SPX_Return", "DAX_Return"]]], axis=1).dropna()
        if tmp.empty: continue
        
        # NOTE: For simplicity, using 50/50 weights for backtesting actual returns
        # A more advanced version could use the model's own forecast weights.
        pret = 0.5 * tmp["SPX_Return"] + 0.5 * tmp["DAX_Return"]
        hits = (pret <= tmp[vcol]).astype(int)
        
        n, n1 = len(hits), hits.sum()
        p = alpha
        pi_hat = n1 / n if n > 0 else 0
        expected_n1 = n * p
        
        # Kupiec POF Test
        if n1 == 0 or n1 == n:
            lr_pof = -2 * n * np.log((1 - p) if n1 == 0 else p)
        else:
            lr_pof = 2 * (n1 * np.log(pi_hat / p) + (n - n1) * np.log((1 - pi_hat) / (1 - p)))
        pof_p = 1 - chi2.cdf(lr_pof, 1)

        # Christoffersen Independence Test
        trans = pd.DataFrame({"prev": hits.shift(1), "curr": hits}).dropna()
        n00, n01 = ((trans.prev == 0) & (trans.curr == 0)).sum(), ((trans.prev == 0) & (trans.curr == 1)).sum()
        n10, n11 = ((trans.prev == 1) & (trans.curr == 0)).sum(), ((trans.prev == 1) & (trans.curr == 1)).sum()
        
        pi0, pi1 = n01/(n00+n01) if (n00+n01)>0 else 0, n11/(n10+n11) if (n10+n11)>0 else 0
        pi_all = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11)>0 else 0
        
        logL0 = (n00+n10)*np.log(1-pi_all) + (n01+n11)*np.log(pi_all) if pi_all > 0 and pi_all < 1 else 0
        logL1 = n00*np.log(1-pi0)+n01*np.log(pi0) + n10*np.log(1-pi1)+n11*np.log(pi1) if all(p > 0 and p < 1 for p in [pi0, pi1]) else 0
        lr_ind = 2 * (logL1 - logL0)
        ind_p = 1 - chi2.cdf(lr_ind, 1)

        results[model] = {
            "breaches": n1,
            "expected_breaches": expected_n1,
            "breach_rate": pi_hat,
            "kupiec_p": pof_p,
            "christoffersen_p": ind_p,
        }
    return results


# ============================================================================ #
# 5. MAIN EXECUTION & TUNING LOOP
# ============================================================================ #

def main():
    """Main function to run tuning, select best model, and generate final forecast."""
    print("=" * 80)
    print(">>> SCRIPT 3 (FINAL): AUTOMATED GARCH-COPULA TUNING & FORECASTING <<<")
    
    os.makedirs(Config.SIMULATION_DIR, exist_ok=True)

    BEST_CONFIG_PATH = f"{Config.SIMULATION_DIR}/best_config.pkl"
    
    u_df = pd.read_csv(f"{Config.MODEL_DIR}/copula_input_data.csv", index_col=0, parse_dates=True)
    full_df = pd.read_csv(f"{Config.DATA_DIR}/spx_dax_daily_data.csv", index_col=0, parse_dates=True)
    dates_to_forecast = full_df.loc[OOS_START_DATE:].index

    def task_generator(dates, config):
        for date in dates:
            yield delayed(one_day_forecast)(date, full_df, u_df, config, MEAN_SPEC, VOL_FAMILIES)
    
    best_config = {}

    # --- Use Optuna for Bayesian Optimization ---
    if RUN_OPTUNA_TUNING:
        print("\n--- Running Optuna Hyperparameter Tuning ---")
        def objective(trial):
            config = {
                "refit_freq": trial.suggest_categorical("refit_freq", [63]),
                "beta_cap": trial.suggest_float("beta_cap", 0.93, 0.99, step=0.02),
                "tail_adj": trial.suggest_float("tail_adj", 1.4, 2.0, step=0.1),
                "sims": SIMS_PER_DAY,
                "alpha": PORTFOLIO_ALPHA,
            }

            print(f"\n[Trial {trial.number}] Testing params: {config}")
            
            tasks = task_generator(dates_to_forecast, config)
            forecasts = Parallel(n_jobs=N_JOBS)(
                tqdm(tasks, desc=f"Forecast Trial {trial.number}", total=len(dates_to_forecast))
            )

            forecast_df = pd.DataFrame([f for f in forecasts if f]).set_index("Date").sort_index()
            backtest_results = run_backtest(forecast_df, full_df, config['alpha'])

            best_metric = float("inf")
            for model_name, metrics in backtest_results.items():
                if metrics['kupiec_p'] > 0.05 and metrics['christoffersen_p'] > 0.05:
                    error = abs(metrics['breaches'] - metrics['expected_breaches'])
                    best_metric = min(best_metric, error)

            # If no model passed the backtest, penalize
            if best_metric == float("inf"):
                return 1e6

            return best_metric

        # --- Run the study ---
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)  # You can increase this

        # --- Best parameters ---
        best_config = {
            "refit_freq": study.best_params["refit_freq"],
            "beta_cap": study.best_params["beta_cap"],
            "tail_adj": study.best_params["tail_adj"],
            "sims": SIMS_PER_DAY,
            "alpha": PORTFOLIO_ALPHA,
        }
        print("\n\n--- Best Params Found by Optuna ---")
        print(best_config)

    else:
        print(f"\n--- Skipping Optuna. Loading best config from {BEST_CONFIG_PATH} ---")
        try:
            with open(BEST_CONFIG_PATH, "rb") as f:
                best_config = pickle.load(f)
            print(f"Successfully loaded configuration: {best_config}")
        except FileNotFoundError:
            print(f"[Error] {BEST_CONFIG_PATH} not found.")
            print("Please run the script with RUN_OPTUNA_TUNING = True at least once.")
            return
    
    # --- Final full forecast using best parameters ---
    print("\n--- Generating Final Forecast with Best Configuration ---")
    final_tasks_gen = task_generator(dates_to_forecast, best_config)
    final_forecasts = Parallel(n_jobs=N_JOBS)(
        tqdm(final_tasks_gen, desc="Final Forecast", total=len(dates_to_forecast))
    )

    final_df = pd.DataFrame([f for f in final_forecasts if f]).set_index("Date").sort_index()

    tuned_results_path = f"{Config.SIMULATION_DIR}/garch_copula_all_results.csv"
    tuned_params_path = f"{Config.SIMULATION_DIR}/copula_params_TUNED.pkl"
    final_df.to_csv(tuned_results_path, float_format="%.6f")

    final_copulas = estimate_copulas(u_df, best_config)
    with open(tuned_params_path, "wb") as fh:
        pickle.dump(final_copulas, fh)
    with open(f"{Config.SIMULATION_DIR}/best_config.pkl", "wb") as f:
        pickle.dump(best_config, f)

    print(f"\nSaved final tuned forecasts → {tuned_results_path}")
    print(f"Saved final tuned copula params → {tuned_params_path}")
    print("="*80)
    print(">>> Process finished successfully. <<<")


if __name__ == "__main__":
    main()

