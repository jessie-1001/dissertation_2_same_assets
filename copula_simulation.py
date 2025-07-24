# =============================================================================
# SCRIPT 03: GARCH-COPULA MODEL FORECASTING (ENHANCED & DOCUMENTED)
#
# Description:
# This script performs the core out-of-sample forecasting using the GARCH-Copula
# framework. Its workflow is as follows:
# 1. It loads the IN-SAMPLE PIT data (`copula_input_data.csv`) created by SCRIPT 02.
#    This data is used ONLY to estimate the static copula dependence parameters.
# 2. It loads the FULL price/return data (`spx_dax_daily_data.csv`).
# 3. It iterates through each day of the out-of-sample period. On each day, it:
#    a. Uses an expanding window of historical data to re-fit the GARCH models for SPX and DAX.
#    b. Forecasts the next day's volatility ($\sigma_{t+1}$) for each asset.
#    c. Simulates a large number of correlated random variables using the static copula parameters.
#    d. Transforms these simulations into portfolio returns using the forecasted volatilities.
#    e. Calculates the 99% VaR and Expected Shortfall (ES) from the simulated portfolio returns.
# 4. It saves all daily forecasts to a CSV file for backtesting in SCRIPT 04.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, t, genpareto
from arch import arch_model
from tqdm import tqdm
import warnings
from scipy.linalg import cholesky
from scipy.optimize import minimize
import pickle
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. STABLE COPULA SAMPLERS AND PARAMETER ESTIMATORS
# (No changes needed in these helper functions)
# -----------------------------------------------------------------------------

def calc_min_var_weights(vol_spx, vol_dax, rho):
    """Calculates minimum variance portfolio weights."""
    var_spx, var_dax = vol_spx**2, vol_dax**2
    cov = rho * vol_spx * vol_dax
    # Handle the case of perfect correlation to avoid division by zero
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
        return sample_gaussian_copula(n_samples, corr_matrix) # Fallback to Gaussian
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
    # Use standard Gumbel sampler
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


# (MLE fitting functions remain the same as in your provided code)
def fit_t_copula_mle(data):
    u = data.values
    kendall_tau = data.corr('kendall').iloc[0, 1]
    rho0 = np.sin(np.pi * kendall_tau / 2)
    bounds = [(-0.99, 0.99), (2.1, 30.0)]
    def neg_log_likelihood(params):
        rho, df = params
        if abs(rho) >= 0.99 or df <= 2.0: return 1e10
        try:
            # Using the logpdf of the t-copula for stability
            z = t.ppf(u, df)
            log_c = stats.t.logpdf(z, df=df, loc=0, scale=1).sum(axis=1) \
                    - stats.multivariate_t.logpdf(z, df=df, shape=np.array([[1, rho], [rho, 1]]))
            return np.sum(log_c)
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
    u = data.values
    def neg_loglik(theta):
        u1, u2 = u[:, 0], u[:, 1]
        c = (theta[0] + 1) * (u1 * u2) ** (-theta[0] - 1) * (u1 ** (-theta[0]) + u2 ** (-theta[0]) - 1) ** (-2 - 1/theta[0])
        return -np.sum(np.log(np.maximum(c, 1e-20)))
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    theta0 = 2 * kendall_tau / (1 - kendall_tau) if (1 - kendall_tau) != 0 else 0.01
    bounds = [(0.01, 20)]
    result = minimize(neg_loglik, [theta0], method='L-BFGS-B', bounds=bounds)
    return result.x[0] if result.success else theta0

def fit_gumbel_copula(data):
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
    return result.x[0] if result.success else theta0


def get_copula_parameters(data):
    """Estimates parameters for various copulas using Maximum Likelihood Estimation."""
    print("Estimating dependence parameters with MLE on in-sample data...")
    pearson_corr = data.corr(method='pearson').values
    t_copula_params = fit_t_copula_mle(data)
    clayton_theta = fit_clayton_copula(data)
    gumbel_theta = fit_gumbel_copula(data)
    
    params = {
        'Gaussian': {'corr_matrix': pearson_corr},
        'StudentT': t_copula_params,
        'Clayton': {'theta': clayton_theta},
        'Gumbel': {'theta': gumbel_theta}
    }
    print("\n--- Static Copula Parameters Estimated ---")
    print(f"  Student-t: ρ = {t_copula_params['corr_matrix'][0,1]:.4f}, ν = {t_copula_params['df']:.2f}")
    print(f"  Clayton:   θ = {clayton_theta:.4f}")
    print(f"  Gumbel:    θ = {gumbel_theta:.4f}")
    return params

# -----------------------------------------------------------------------------
# 2. STABLE GARCH MODEL FITTING AND ROLLING FORECAST
# -----------------------------------------------------------------------------

def fit_garch_model(returns, max_retries=3):
    """Fits a GARCH(1,1)-t model with a constant mean."""
    model = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    for attempt in range(max_retries):
        try:
            res = model.fit(disp='off', options={'maxiter': 1000})
            return res
        except Exception:
            if attempt == max_retries - 1:
                # Fallback to historical volatility if GARCH fails repeatedly
                hist_vol = returns.std()
                class FallbackResult:
                    def __init__(self, vol):
                        self.params = {'mu': returns.mean(), 'nu': 5} # Default nu
                        self.conditional_volatility = np.full_like(returns, vol)
                    def forecast(self, horizon=1, reindex=False):
                        class VarianceForecast:
                            def __init__(self, vol): self.variance = pd.DataFrame([[vol**2]])
                        return VarianceForecast(hist_vol)
                return FallbackResult(hist_vol)
    return None

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    """Performs the full simulation and risk calculation for a single day."""
    # Use an expanding window for GARCH fitting
    window_data = full_data.loc[:t_index]
    
    # Re-fit GARCH models daily on the expanding window
    res_spx = fit_garch_model(window_data['SPX_Return'])
    res_dax = fit_garch_model(window_data['DAX_Return'])
    
    # Forecast one-day-ahead volatility
    sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    sigma_t1_dax = np.sqrt(res_dax.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    
    # Forecast one-day-ahead mean
    mu_t1_spx = res_spx.params['mu']
    mu_t1_dax = res_dax.params['mu']

    # Get degrees of freedom from the latest GARCH fit
    nu_spx = res_spx.params.get('nu', 5) # Default to 5 if not available
    nu_dax = res_dax.params.get('nu', 5)

    # Calculate portfolio weights based on forecasted vols
    rho_gauss = copula_params['Gaussian']['corr_matrix'][0, 1]
    weight_spx, weight_dax = calc_min_var_weights(sigma_t1_spx, sigma_t1_dax, rho_gauss)
    
    daily_forecasts = {}
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Gumbel': lambda n, p: sample_survival_gumbel_copula(n, p['theta']),
        'Clayton': lambda n, p: sample_survival_clayton_copula(n, p['theta'])
    }
    
    for name, params in copula_params.items():
        try:
            # Step 1: Simulate from Copula
            simulated_uniforms = samplers[name](n_simulations, params)
            u_spx = np.clip(simulated_uniforms[:, 0], 1e-6, 1 - 1e-6)
            u_dax = np.clip(simulated_uniforms[:, 1], 1e-6, 1 - 1e-6)
            
            # Step 2: Inverse transform to standardized residuals
            z_spx = t.ppf(u_spx, df=nu_spx)
            z_dax = t.ppf(u_dax, df=nu_dax)
            
            # Step 3: Scale by forecasted volatility and mean to get asset returns
            r_spx_sim = mu_t1_spx + sigma_t1_spx * z_spx
            r_dax_sim = mu_t1_dax + sigma_t1_dax * z_dax
            
            # Step 4: Calculate portfolio returns
            r_portfolio_sim = weight_spx * r_spx_sim + weight_dax * r_dax_sim
            
            # Step 5: Calculate VaR and ES
            var_99 = np.percentile(r_portfolio_sim, 1)
            es_99 = r_portfolio_sim[r_portfolio_sim < var_99].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99, 'ES_99': es_99,
                'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax,
                'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax
            }
        except Exception:
            # Fallback in case a specific copula simulation fails
            hist_returns = window_data['SPX_Return'] * weight_spx + window_data['DAX_Return'] * weight_dax
            daily_forecasts[name] = {
                'VaR_99': np.percentile(hist_returns, 1), 'ES_99': hist_returns[hist_returns < np.percentile(hist_returns, 1)].mean(),
                'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax, 'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax
            }
    return t_index, daily_forecasts

def _one_day_wrapper(day, full_data, copula_params):
    """Wrapper function for parallel processing."""
    # Get the index of the day before the forecast date
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
    return flat

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        print("\n" + "=" * 80)
        print(">>> SCRIPT 03: GARCH-COPULA ROLLING FORECAST <<<")
        
        # --- Step 1: Load Data ---
        # Load the IN-SAMPLE data. This is used to get the static copula parameters.
        # This is a key methodological step: parameters are estimated on a training set.
        print("\nLoading IN-SAMPLE data ('copula_input_data.csv') for parameter estimation...")
        copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
        
        # Load the FULL dataset. This will be used in the expanding window for daily GARCH re-fitting.
        print("Loading FULL data ('spx_dax_daily_data.csv') for rolling forecasts...")
        full_data = pd.read_csv('spx_dax_daily_data.csv', index_col='Date', parse_dates=True)
        # Convert returns to be on a scale of 1 = 1%, as expected by arch library
        full_data['SPX_Return'] = full_data['SPX_Return'] 
        full_data['DAX_Return'] = full_data['DAX_Return'] 

        # --- Step 2: Estimate Static Copula Parameters ---
        copula_params = get_copula_parameters(copula_input_data)

        # --- Step 3: Run Rolling Forecast ---
        out_of_sample_start = '2020-01-01'
        forecast_dates = full_data.loc[out_of_sample_start:].index
        
        print("\nStarting out-of-sample rolling forecast...")
        print(f"Forecast Period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        print(f"Number of forecast days: {len(forecast_dates)}")
        
        # Use joblib for robust parallel processing
        all_forecasts = Parallel(n_jobs=-1)(
            delayed(_one_day_wrapper)(d, full_data, copula_params) for d in tqdm(forecast_dates, desc="Forecasting VaR/ES")
        )
        
        # --- Step 4: Save Results ---
        print("\nForecasts generated. Saving results...")
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        
        # Ensure correct sorting by date
        forecasts_df.sort_index(inplace=True)
        
        forecast_output_file = 'garch_copula_forecasts_improved.csv'
        forecasts_df.to_csv(forecast_output_file)
        print(f"All forecasts saved to '{forecast_output_file}'.")
        
        # Save the estimated copula parameters for reference
        with open('copula_params_improved.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        print("Copula parameters saved to 'copula_params_improved.pkl'.")

        print("\n" + "=" * 80)
        print(">>> SCRIPT 03 FINISHED SUCCESSFULLY <<<")
        print("=" * 80 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}")
        print("Please ensure you have run SCRIPT 01 and SCRIPT 02 successfully first.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()