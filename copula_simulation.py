# =============================================================================
# SCRIPT 03: GARCH-COPULA MODEL FORECASTING (FINAL FIXED VERSION)
#
# Key fixes:
# 1. Weight calculation: Use static theoretical correlation for each copula
# 2. Quantile transformation: Gaussian uses normal dist, others use t-dist
# 3. Archimedean copulas: Use theoretical linear correlation
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, t
from arch import arch_model
from tqdm import tqdm
import warnings
from scipy.linalg import cholesky
from scipy.optimize import minimize
import pickle
from joblib import Parallel, delayed
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. STABLE COPULA SAMPLERS AND PARAMETER ESTIMATORS
# -----------------------------------------------------------------------------

def calc_min_var_weights(vol_spx, vol_dax, rho):
    """Calculates minimum variance portfolio weights."""
    var_spx, var_dax = vol_spx**2, vol_dax**2
    cov = rho * vol_spx * vol_dax
    
    # Handle perfect correlation case
    if abs(cov - vol_spx * vol_dax) < 1e-8:
        return (0.5, 0.5)
    
    w_spx = (var_dax - cov) / (var_spx + var_dax - 2 * cov)
    return np.clip(w_spx, 0.3, 0.7), 1 - np.clip(w_spx, 0.3, 0.7)

def sample_gaussian_copula(n_samples, corr_matrix):
    """Generates random samples from a Gaussian copula."""
    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        # Fallback for non-positive-definite matrix
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        reconstituted = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D = np.diag(1 / np.sqrt(np.diag(reconstituted)))
        corr_matrix = D @ reconstituted @ D
        L = cholesky(corr_matrix, lower=True)
    
    z = np.random.normal(0, 1, size=(n_samples, 2))
    z_correlated = z @ L.T
    return norm.cdf(z_correlated)

def sample_t_copula(n_samples, corr_matrix, df):
    """Generates random samples from a Student's t-copula."""
    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        return sample_gaussian_copula(n_samples, corr_matrix)  # Fallback to Gaussian
    
    g = np.random.chisquare(df, n_samples)
    z = np.random.normal(0, 1, size=(n_samples, 2))
    z_correlated = z @ L.T
    x = np.sqrt(df / g)[:, np.newaxis] * z_correlated
    return t.cdf(x, df=df)

def sample_clayton_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u2 = (u1**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    return np.column_stack((u1, u2))

def sample_gumbel_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u1_tilde = -np.log(u1)
    u2_tilde = -np.log(v)
    g_u1 = u1_tilde**theta
    g_u2 = u2_tilde**theta
    g_inv_sum = (g_u1 + g_u2)**(1/theta)
    return np.column_stack((np.exp(-g_inv_sum), np.exp(-g_inv_sum)))

def sample_survival_clayton_copula(n_samples, theta):
    """Samples from a survival Clayton copula for lower tail dependence."""
    u, v = sample_clayton_copula(n_samples, theta).T
    return np.column_stack((1 - u, 1 - v))

def sample_survival_gumbel_copula(n_samples, theta):
    """Samples from a survival Gumbel copula for upper tail dependence."""
    u, v = sample_gumbel_copula(n_samples, theta).T
    return np.column_stack((1 - u, 1 - v))

def fit_t_copula_mle(data):
    """Fits Student-t Copula parameters using MLE"""
    u = data.values
    kendall_tau = data.corr('kendall').iloc[0, 1]
    rho0 = np.sin(np.pi * kendall_tau / 2)
    bounds = [(-0.99, 0.99), (2.1, 30.0)]
    
    def neg_log_likelihood(params):
        rho, df = params
        if abs(rho) >= 0.99 or df <= 2.0: 
            return 1e10
        try:
            z = t.ppf(u, df)
            log_c = t.logpdf(z, df=df, loc=0, scale=1).sum(axis=1) - \
                    sm.distributions.multivariate_t.logpdf(z, df=df, shape=np.array([[1, rho], [rho, 1]]))
            return -np.sum(log_c)
        except:
            return 1e10
    
    best_ll = np.inf
    best_params = [rho0, 8]
    for init_df in [5, 10, 15]:
        res = minimize(neg_log_likelihood, [rho0, init_df], method='SLSQP', bounds=bounds)
        if res.fun < best_ll and res.success:
            best_ll = res.fun
            best_params = res.x
    
    rho_hat, df_hat = best_params
    return {'corr_matrix': np.array([[1, rho_hat], [rho_hat, 1]]), 'df': df_hat}

def fit_clayton_copula(data):
    """Fits Clayton Copula parameters using MLE"""
    u = data.values
    def neg_loglik(theta):
        u1, u2 = u[:, 0], u[:, 1]
        c = (theta[0] + 1) * (u1 * u2) ** (-theta[0] - 1) * (u1 ** (-theta[0]) + u2 ** (-theta[0]) - 1) ** (-2 - 1/theta[0])
        return -np.sum(np.log(np.maximum(c, 1e-20)))
    
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    theta0 = 2 * kendall_tau / (1 - kendall_tau) if (1 - kendall_tau) != 0 else 0.01
    bounds = [(0.01, 20)]
    result = minimize(neg_loglik, [theta0], method='L-BFGS-B', bounds=bounds)
    return max(result.x[0], 0.1)  # Ensure positive theta

def fit_gumbel_copula(data):
    """Fits Gumbel Copula parameters using MLE"""
    u = data.values
    def neg_loglik(theta):
        u1, u2 = u[:, 0], u[:, 1]
        v = (-np.log(u1)) ** theta[0] + (-np.log(u2)) ** theta[0]
        c = np.exp(-v ** (1/theta[0])) * (v ** (-2 + 2/theta[0])) * ((np.log(u1)*np.log(u2)) ** (theta[0]-1)) / (u1*u2) * (1 + (theta[0]-1)*v ** (-1/theta[0]))
        return -np.sum(np.log(np.maximum(c, 1e-20)))
    
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    theta0 = 1 / (1 - kendall_tau) if (1 - kendall_tau) != 0 else 1.01
    bounds = [(1.01, 20)]
    result = minimize(neg_loglik, [theta0], method='L-BFGS-B', bounds=bounds)
    return max(result.x[0], 1.01)  # Ensure theta > 1

def get_copula_parameters(data):
    """Estimates parameters for various copulas using MLE"""
    print("Estimating dependence parameters with MLE on in-sample data...")
    pearson_corr = data.corr(method='pearson').values
    t_copula_params = fit_t_copula_mle(data)
    clayton_theta = fit_clayton_copula(data)
    gumbel_theta = fit_gumbel_copula(data)
    
    # Calculate theoretical linear correlations
    clayton_rho = 1 - (clayton_theta/(clayton_theta+2)) if clayton_theta > 0 else 0
    gumbel_rho = 1 - (1/gumbel_theta) if gumbel_theta > 1 else 0
    
    params = {
        'Gaussian': {
            'corr_matrix': pearson_corr,
            'rho': pearson_corr[0, 1]
        },
        'StudentT': {
            'corr_matrix': t_copula_params['corr_matrix'],
            'df': t_copula_params['df'],
            'rho': t_copula_params['corr_matrix'][0, 1]
        },
        'Clayton': {
            'theta': clayton_theta,
            'rho': clayton_rho
        },
        'Gumbel': {
            'theta': gumbel_theta,
            'rho': gumbel_rho
        }
    }
    
    print("\n--- Static Copula Parameters Estimated ---")
    print(f"  Student-t: ρ = {params['StudentT']['rho']:.4f}, ν = {params['StudentT']['df']:.2f}")
    print(f"  Clayton:   θ = {params['Clayton']['theta']:.4f}, ρ = {params['Clayton']['rho']:.4f}")
    print(f"  Gumbel:    θ = {params['Gumbel']['theta']:.4f}, ρ = {params['Gumbel']['rho']:.4f}")
    
    return params

# -----------------------------------------------------------------------------
# 2. GARCH MODEL FITTING AND ROLLING FORECAST (FINAL FIXED VERSION)
# -----------------------------------------------------------------------------

def fit_garch_model(returns, max_retries=3):
    """Fits a GARCH(1,1)-t model with constant mean"""
    model = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    
    for attempt in range(max_retries):
        try:
            res = model.fit(disp='off', options={'maxiter': 1000})
            return res
        except Exception:
            if attempt == max_retries - 1:
                # Fallback to historical volatility
                hist_vol = returns.std()
                class FallbackResult:
                    def __init__(self, vol):
                        self.params = {'mu': returns.mean(), 'nu': 5}
                        self.conditional_volatility = np.full_like(returns, vol)
                    def forecast(self, horizon=1, reindex=False):
                        class VarianceForecast:
                            def __init__(self, vol): self.variance = pd.DataFrame([[vol**2]])
                        return VarianceForecast(hist_vol)
                return FallbackResult(hist_vol)
    return None

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    """Performs full simulation and risk calculation for a single day"""
    # Use expanding window for GARCH fitting
    window_data = full_data.loc[:t_index]
    
    # Re-fit GARCH models
    res_spx = fit_garch_model(window_data['SPX_Return'])
    res_dax = fit_garch_model(window_data['DAX_Return'])
    
    # Forecast one-day-ahead volatility
    sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    sigma_t1_dax = np.sqrt(res_dax.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    
    # Forecast one-day-ahead mean
    mu_t1_spx = res_spx.params['mu']
    mu_t1_dax = res_dax.params['mu']

    # Get degrees of freedom from GARCH fit
    nu_spx = res_spx.params.get('nu', 5)
    nu_dax = res_dax.params.get('nu', 5)

    daily_forecasts = {}
    
    # Define copula samplers
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Clayton': lambda n, p: sample_survival_clayton_copula(n, p['theta']),
        'Gumbel': lambda n, p: sample_survival_gumbel_copula(n, p['theta'])
    }
    
    for name, params in copula_params.items():
        try:
            # === CRITICAL FIX 1: Use static theoretical correlation for weights ===
            rho = params['rho']
            weight_spx, weight_dax = calc_min_var_weights(sigma_t1_spx, sigma_t1_dax, rho)
            
            # === CRITICAL FIX 2: Simulate using copula parameters ===
            simulated_uniforms = samplers[name](n_simulations, params)
            u_spx = np.clip(simulated_uniforms[:, 0], 1e-6, 1 - 1e-6)
            u_dax = np.clip(simulated_uniforms[:, 1], 1e-6, 1 - 1e-6)
            
            # === CRITICAL FIX 3: Correct quantile transformation ===
            if name == 'Gaussian':
                # Gaussian uses normal distribution
                z_spx = norm.ppf(u_spx)
                z_dax = norm.ppf(u_dax)
            elif name == 'StudentT':
                # Student-t uses copula degrees of freedom
                z_spx = t.ppf(u_spx, df=params['df'])
                z_dax = t.ppf(u_dax, df=params['df'])
            else:
                # Archimedean copulas use fixed df t-distribution
                z_spx = t.ppf(u_spx, df=5)
                z_dax = t.ppf(u_dax, df=5)
            
            # Calculate asset returns
            r_spx_sim = mu_t1_spx + sigma_t1_spx * z_spx
            r_dax_sim = mu_t1_dax + sigma_t1_dax * z_dax
            
            # Calculate portfolio returns
            r_portfolio_sim = weight_spx * r_spx_sim + weight_dax * r_dax_sim
            
            # Calculate VaR and ES
            var_99 = np.percentile(r_portfolio_sim, 1)
            es_99 = r_portfolio_sim[r_portfolio_sim < var_99].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99, 'ES_99': es_99,
                'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax,
                'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax,
                'Rho': rho  # Save correlation for debugging
            }
        except Exception as e:
            print(f"{name} Copula error on {t_index.date()}: {str(e)}")
            # Fallback to historical simulation
            hist_returns = window_data['SPX_Return'] * weight_spx + window_data['DAX_Return'] * weight_dax
            daily_forecasts[name] = {
                'VaR_99': np.percentile(hist_returns, 1), 
                'ES_99': hist_returns[hist_returns < np.percentile(hist_returns, 1)].mean(),
                'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax, 
                'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax,
                'Rho': rho
            }
    return t_index, daily_forecasts

def _one_day_wrapper(day, full_data, copula_params):
    """Wrapper function for parallel processing"""
    # Get index of day before forecast date
    t_idx = full_data.index[full_data.index.get_loc(day) - 1]
    
    date, forecasts = run_simulation_for_day(t_idx, full_data, copula_params)
    
    flat = {'Date': day}
    for model, vals in forecasts.items():
        flat[f'{model}_VaR_99'] = vals['VaR_99']
        flat[f'{model}_ES_99'] = vals['ES_99']
        flat[f'{model}_Weight_SPX'] = vals['Weight_SPX']
        flat[f'{model}_Weight_DAX'] = vals['Weight_DAX']
        flat[f'{model}_Vol_SPX'] = vals['Vol_SPX']
        flat[f'{model}_Vol_DAX'] = vals['Vol_DAX']
        flat[f'{model}_Rho'] = vals.get('Rho', np.nan)  # Save correlation
    return flat

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        print("\n" + "=" * 80)
        print(">>> SCRIPT 03: GARCH-COPULA ROLLING FORECAST (FINAL FIXED) <<<")
        
        # --- Step 1: Load Data ---
        print("\nLoading in-sample data ('copula_input_data.csv')...")
        copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
        
        print("Loading full dataset ('spx_dax_daily_data.csv')...")
        full_data = pd.read_csv('spx_dax_daily_data.csv', index_col='Date', parse_dates=True)
        full_data['SPX_Return'] = full_data['SPX_Return'] 
        full_data['DAX_Return'] = full_data['DAX_Return'] 

        # --- Step 2: Estimate Static Copula Parameters ---
        copula_params = get_copula_parameters(copula_input_data)

        # --- Step 3: Run Rolling Forecast ---
        out_of_sample_start = '2020-01-01'
        forecast_dates = full_data.loc[out_of_sample_start:].index
        
        print("\nStarting out-of-sample rolling forecast...")
        print(f"Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        print(f"Number of forecast days: {len(forecast_dates)}")
        
        # Use parallel processing
        all_forecasts = Parallel(n_jobs=-1)(
            delayed(_one_day_wrapper)(d, full_data, copula_params) 
            for d in tqdm(forecast_dates, desc="Forecasting VaR/ES")
        )
        
        # --- Step 4: Save Results ---
        print("\nForecasts completed. Saving results...")
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        forecasts_df.sort_index(inplace=True)
        
        forecast_output_file = 'garch_copula_forecasts_final.csv'
        forecasts_df.to_csv(forecast_output_file)
        print(f"All forecasts saved to '{forecast_output_file}'.")
        
        # Save copula parameters
        with open('copula_params_final.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        print("Copula parameters saved to 'copula_params_final.pkl'.")

        print("\n" + "=" * 80)
        print(">>> SCRIPT 03 COMPLETED SUCCESSFULLY <<<")
        print("=" * 80 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}")
        print("Please ensure you've run SCRIPT 01 and SCRIPT 02 successfully")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()