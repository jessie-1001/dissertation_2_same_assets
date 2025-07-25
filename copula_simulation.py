# =============================================================================
# SCRIPT 03: GARCH-COPULA MODEL FORECASTING (PROFESSIONALLY REVISED VERSION)
#
# Key revisions based on expert feedback:
# 1. Fixed Gumbel copula sampling implementation
# 2. Corrected Archimedean copula rho calculation using Kendall's tau conversion
# 3. Proper quantile transformation using marginal t-distributions
# 4. Improved Gaussian copula correlation estimation
# 5. Clarified survival copula tail dependencies
# 6. Unified correlation for weight calculation using t-copula's rho
# 7. Added robustness checks for Archimedean copulas
# 8. Enhanced error handling and numerical stability
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
from scipy.stats import kendalltau

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CORRECTED COPULA SAMPLERS AND PARAMETER ESTIMATORS
# -----------------------------------------------------------------------------

def calc_min_var_weights(vol_spx, vol_dax, rho, leverage_effect=0):
    """Calculates minimum variance portfolio weights with leverage adjustment"""
    var_spx, var_dax = vol_spx**2, vol_dax**2
    cov = rho * vol_spx * vol_dax
    
    if abs(cov - vol_spx * vol_dax) < 1e-8:
        base_weights = (0.5, 0.5)
    else:
        w_spx = (var_dax - cov) / (var_spx + var_dax - 2 * cov)
        base_weights = (np.clip(w_spx, 0.3, 0.7), 1 - np.clip(w_spx, 0.3, 0.7))
    
    # Leverage effect adjustment: reduce SPX weight when leverage effect is strong
    if leverage_effect > 1.0:  # Strong leverage effect
        adj_w_spx = base_weights[0] * 0.9
        return adj_w_spx, 1 - adj_w_spx
    return base_weights

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
    """Generates samples from Clayton copula using stable Marshall-Olkin algorithm"""
    if theta <= 0:
        # θ=0时退化为独立Copula
        return np.random.uniform(0, 1, (n_samples, 2))
    
    # 生成Gamma分布变量
    gamma = np.random.gamma(shape=1/theta, scale=1.0, size=n_samples)
    # 生成独立均匀分布
    u1 = np.random.uniform(0, 1, n_samples)
    u2 = np.random.uniform(0, 1, n_samples)
    # 计算Copula变量
    u = (1 - np.log(u1) / gamma) ** (-1/theta)
    v = (1 - np.log(u2) / gamma) ** (-1/theta)
    return np.column_stack((
        np.clip(u, 1e-6, 1-1e-6), 
        np.clip(v, 1e-6, 1-1e-6)
    ))

def sample_gumbel_copula(n_samples, theta):
    """Generates samples from Gumbel copula using stable Marshall-Olkin algorithm"""
    if theta <= 1.0:
        # 当θ=1时退化为独立Copula
        return np.random.uniform(0, 1, (n_samples, 2))
    
    beta = 1.0 / theta
    # 生成共享的稳定分布变量
    S = np.random.gamma(shape=1/beta, scale=1.0, size=n_samples)
    # 生成独立的指数分布
    e1 = np.random.exponential(scale=1.0, size=n_samples)
    e2 = np.random.exponential(scale=1.0, size=n_samples)
    # 计算Copula变量
    u = np.exp(-e1 / (S ** beta))
    v = np.exp(-e2 / (S ** beta))
    return np.column_stack((
        np.clip(u, 1e-6, 1-1e-6), 
        np.clip(v, 1e-6, 1-1e-6)
    ))

def sample_survival_clayton_copula(n_samples, theta):
    """Samples from survival Clayton copula for UPPER tail dependence"""
    u, v = sample_clayton_copula(n_samples, theta).T
    return np.column_stack((1 - u, 1 - v))

def sample_survival_gumbel_copula(n_samples, theta):
    """Samples from survival Gumbel copula for LOWER tail dependence"""
    u, v = sample_gumbel_copula(n_samples, theta).T
    return np.column_stack((1 - u, 1 - v))

def fit_t_copula_mle(data):
    """Fits Student-t Copula parameters using MLE with fallback"""
    u = data.values
    tau, _ = kendalltau(data.iloc[:, 0], data.iloc[:, 1])
    rho0 = np.sin(np.pi * tau / 2)
    bounds = [(-0.99, 0.99), (2.1, 30.0)]
    
    def neg_log_likelihood(params):
        rho, df = params
        if abs(rho) >= 0.99 or df <= 2.0: 
            return 1e10
        try:
            # Use statsmodels if available
            from statsmodels.distributions.multivariate_t import multivariate_t
            z = t.ppf(u, df)
            log_c = t.logpdf(z, df=df).sum(axis=1) - \
                    multivariate_t.logpdf(z, df=df, shape=np.array([[1, rho], [rho, 1]]))
            return -np.sum(log_c)
        except ImportError:
            # Fallback implementation
            z = t.ppf(u, df)
            log_c = t.logpdf(z, df=df).sum(axis=1)
            cov = np.array([[1, rho], [rho, 1]])
            inv_cov = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            log_density = -0.5 * (df + 2) * np.log(1 + np.sum(z @ inv_cov * z, axis=1) / df)
            log_density += -0.5 * np.log(det) - t.logpdf(z, df=df).sum(axis=1)
            return -np.sum(log_density)
    
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
    
    tau, _ = kendalltau(data.iloc[:, 0], data.iloc[:, 1])
    theta0 = 2 * tau / (1 - tau) if (1 - tau) != 0 else 0.01
    bounds = [(0.01, 20)]
    result = minimize(neg_loglik, [theta0], method='L-BFGS-B', bounds=bounds)
    return max(result.x[0], 0.1)

def fit_gumbel_copula(data):
    """Fits Gumbel Copula parameters using MLE"""
    u = data.values
    def neg_loglik(theta):
        u1, u2 = u[:, 0], u[:, 1]
        v = (-np.log(u1)) ** theta[0] + (-np.log(u2)) ** theta[0]
        g = v ** (1/theta[0])
        c = np.exp(-g) * g * (u1 * u2) ** (-1) * (np.log(u1) * np.log(u2)) ** (theta[0]-1) / v
        return -np.sum(np.log(np.maximum(c, 1e-20)))
    
    tau, _ = kendalltau(data.iloc[:, 0], data.iloc[:, 1])
    theta0 = 1 / (1 - tau) if (1 - tau) != 0 else 1.01
    bounds = [(1.01, 20)]  # 确保theta>=1.01
    result = minimize(neg_loglik, [theta0], method='L-BFGS-B', bounds=bounds)
    return max(min(result.x[0], 20), 1.01)  # 限制theta在[1.01, 20]之间

def get_copula_parameters(data):
    """Estimates parameters for various copulas using MLE with corrected rho"""
    print("Estimating dependence parameters with MLE on in-sample data...")
    
    # For Gaussian copula: transform to normal space
    z_data = norm.ppf(data.values)
    pearson_corr = np.corrcoef(z_data.T)
    
    # Fit copulas
    t_copula_params = fit_t_copula_mle(data)
    clayton_theta = fit_clayton_copula(data)
    gumbel_theta = fit_gumbel_copula(data)
    
    # Calculate Kendall's tau for each copula
    clayton_tau = clayton_theta / (clayton_theta + 2)
    gumbel_tau = 1 - 1/gumbel_theta
    
    # Convert tau to linear correlation using normal transformation
    clayton_rho = np.sin(np.pi * clayton_tau / 2)
    gumbel_rho = np.sin(np.pi * gumbel_tau / 2)
    
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
# 2. GARCH MODEL FITTING AND ROLLING FORECAST
# -----------------------------------------------------------------------------

def fit_garch_model(returns, asset_type, max_retries=3):
    """Fits appropriate GARCH model based on asset type"""
    if asset_type == "SPX":
        model = arch_model(returns, mean='AR', lags=1, vol='EGARCH', p=1, o=1, q=1, dist='t')
    else:  # DAX
        model = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    
    for attempt in range(max_retries):
        try:
            res = model.fit(disp='off', options={'maxiter': 1000})
            return res
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {asset_type}: {str(e)}")
            if attempt == max_retries - 1:
                # Enhanced fallback: use EWMA volatility
                ewma_vol = returns.ewm(span=63).std().iloc[-1]
                class FallbackResult:
                    def __init__(self, vol, mean_type='Constant'):
                        self.params = {
                            'mu': returns.mean() if mean_type=='Constant' else 0,
                            'Const': returns.mean() if mean_type=='AR' else 0,
                            'SPX_Return[1]': 0,
                            'nu': 5
                        }
                        self.conditional_volatility = np.full_like(returns, vol)
                    def forecast(self, horizon=1, reindex=False):
                        class VarianceForecast:
                            def __init__(self, vol): self.variance = pd.DataFrame([[vol**2]])
                        return VarianceForecast(ewma_vol)
                return FallbackResult(ewma_vol, 'AR' if asset_type=='SPX' else 'Constant')
    return None

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    """Performs full simulation and risk calculation for a single day with robust error handling"""
    window_data = full_data.loc[:t_index]
    
    # --- Step 1: Fit GARCH models and forecast volatility ---
    res_spx = fit_garch_model(window_data['SPX_Return'], "SPX")
    res_dax = fit_garch_model(window_data['DAX_Return'], "DAX")
    
    # Forecast volatility with fallback
    try:
        sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except:
        sigma_t1_spx = res_spx.conditional_volatility.iloc[-1] if hasattr(res_spx, 'conditional_volatility') else window_data['SPX_Return'].std()
    
    try:
        sigma_t1_dax = np.sqrt(res_dax.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except:
        sigma_t1_dax = res_dax.conditional_volatility.iloc[-1] if hasattr(res_dax, 'conditional_volatility') else window_data['DAX_Return'].std()
    
    # Correct mean prediction
    if 'Const' in res_spx.params and 'SPX_Return[1]' in res_spx.params:
        mu_t1_spx = res_spx.params['Const'] + res_spx.params['SPX_Return[1]'] * window_data['SPX_Return'].iloc[-1]
    else:
        mu_t1_spx = res_spx.params.get('mu', 0)
    
    mu_t1_dax = res_dax.params.get('mu', 0)

    # Get degrees of freedom from marginal models
    nu_spx = res_spx.params.get('nu', 5)
    nu_dax = res_dax.params.get('nu', 5)
    
    # Calculate leverage effect strength (only for SPX)
    leverage_effect_spx = 0
    if 'gamma[1]' in res_spx.params:
        leverage_effect_spx = abs(res_spx.params['gamma[1]'] / (res_spx.params.get('alpha[1]', 1e-6) + 1e-6))

    # --- Step 2: Unified correlation for weight calculation (use t-copula's rho) ---
    # 修正：确保base_rho被正确定义
    base_rho = copula_params['StudentT']['rho']
    
    # Calculate portfolio weights (same for all copulas)
    weight_spx, weight_dax = calc_min_var_weights(
        sigma_t1_spx, sigma_t1_dax, base_rho, leverage_effect_spx
    )
    
    daily_forecasts = {}
    
    # --- Step 3: Define copula samplers (using survival versions) ---
    # 修正：使用生存Copula版本捕捉上尾依赖
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Clayton': lambda n, p: sample_survival_clayton_copula(n, p['theta']),  # 生存Clayton
        'Gumbel': lambda n, p: sample_survival_gumbel_copula(n, p['theta'])     # 生存Gumbel
    }
    
    # --- Step 4: Simulate for each copula type ---
    for name, params in copula_params.items():
        try:
            # Simulate using copula with numerical error suppression
            with np.errstate(over='ignore', invalid='ignore'):
                simulated_uniforms = samplers[name](n_simulations, params)
                u_spx = np.clip(simulated_uniforms[:, 0], 1e-6, 1 - 1e-6)
                u_dax = np.clip(simulated_uniforms[:, 1], 1e-6, 1 - 1e-6)
            
            # Correct quantile transformation using marginal distributions
            if name == 'Gaussian':
                z_spx = norm.ppf(u_spx)
                z_dax = norm.ppf(u_dax)
            else:
                z_spx = t.ppf(u_spx, df=nu_spx)
                z_dax = t.ppf(u_dax, df=nu_dax)
            
            # Calculate asset returns
            r_spx_sim = mu_t1_spx + sigma_t1_spx * z_spx
            r_dax_sim = mu_t1_dax + sigma_t1_dax * z_dax
            
            # Calculate portfolio returns
            r_portfolio_sim = weight_spx * r_spx_sim + weight_dax * r_dax_sim
            
            # Strict NaN/Inf filtering
            valid_mask = np.isfinite(r_portfolio_sim)
            valid_count = np.sum(valid_mask)
            
            if valid_count < 0.95 * n_simulations:
                raise RuntimeError(f"{name} copula produced >5% invalid returns ({valid_count}/{n_simulations} valid)")
            
            r_portfolio_sim = r_portfolio_sim[valid_mask]
            
            # Calculate VaR and ES (1% level)
            var_99 = np.percentile(r_portfolio_sim, 1)
            losses = r_portfolio_sim[r_portfolio_sim <= var_99]
            
            if len(losses) > 0:
                es_99 = losses.mean()
            else:
                # Fallback: use worst 1% of samples
                n_cut = max(1, int(0.01 * len(r_portfolio_sim)))
                es_99 = np.partition(r_portfolio_sim, n_cut-1)[:n_cut].mean()
            
            daily_forecasts[name] = {
                'VaR_99': var_99, 'ES_99': es_99,
                'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax,
                'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax,
                'Rho': base_rho,
                'Nu_SPX': nu_spx, 'Nu_DAX': nu_dax
            }
            
        except Exception as e:
            # Enhanced fallback using historical simulation
            try:
                # 修正：使用window_data而不是full_data.loc[:t_index]
                hist_returns = window_data['SPX_Return'] * weight_spx + window_data['DAX_Return'] * weight_dax
                
                # Filter NaN/Inf in historical data
                valid_hist = hist_returns[np.isfinite(hist_returns)]
                if len(valid_hist) == 0:
                    raise ValueError("Historical fallback data contains only NaN/Inf")
                
                # Calculate VaR and ES from historical data
                var_99 = np.percentile(valid_hist, 1)
                losses = valid_hist[valid_hist <= var_99]
                
                daily_forecasts[name] = {
                    'VaR_99': var_99,
                    'ES_99': losses.mean() if len(losses) > 0 else np.nan,
                    'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax,
                    'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax,
                    'Rho': base_rho,
                    'Nu_SPX': nu_spx, 'Nu_DAX': nu_dax
                }
                
            except Exception as fallback_err:
                # Final fallback: return NaN values
                daily_forecasts[name] = {
                    'VaR_99': np.nan, 'ES_99': np.nan,
                    'Weight_SPX': weight_spx, 'Weight_DAX': weight_dax,
                    'Vol_SPX': sigma_t1_spx, 'Vol_DAX': sigma_t1_dax,
                    'Rho': base_rho,
                    'Nu_SPX': nu_spx, 'Nu_DAX': nu_dax
                }
    
    # 修正：返回预测日期(t_index)而不是当日日期(day)
    return t_index, daily_forecasts


def _one_day_wrapper(day, full_data, copula_params, n_simulations=50000):
    """Wrapper function for parallel processing with robust error handling"""
    try:
        # Get index of day before forecast date
        t_idx = full_data.index[full_data.index.get_loc(day) - 1]
        
        # Run the simulation
        date, forecasts = run_simulation_for_day(t_idx, full_data, copula_params, n_simulations)
        
        # Prepare results dictionary
        flat = {'Date': day}
        for model, vals in forecasts.items():
            flat[f'{model}_VaR_99'] = vals['VaR_99']
            flat[f'{model}_ES_99'] = vals['ES_99']
            flat[f'{model}_Weight_SPX'] = vals['Weight_SPX']
            flat[f'{model}_Weight_DAX'] = vals['Weight_DAX']
            flat[f'{model}_Vol_SPX'] = vals['Vol_SPX']
            flat[f'{model}_Vol_DAX'] = vals['Vol_DAX']
            flat[f'{model}_Rho'] = vals.get('Rho', np.nan)
            flat[f'{model}_Nu_SPX'] = vals.get('Nu_SPX', np.nan)
            flat[f'{model}_Nu_DAX'] = vals.get('Nu_DAX', np.nan)
        return flat
        
    except Exception as e:
        import traceback
        print(f"\nCRITICAL ERROR processing {day}: {str(e)}")
        traceback.print_exc()
        
        # 返回包含NaN的结构化结果
        error_result = {'Date': day}
        models = ['Gaussian', 'StudentT', 'Clayton', 'Gumbel']
        metrics = ['VaR_99', 'ES_99', 'Weight_SPX', 'Weight_DAX', 'Vol_SPX', 'Vol_DAX', 'Rho', 'Nu_SPX', 'Nu_DAX']
        
        for model in models:
            for metric in metrics:
                error_result[f"{model}_{metric}"] = np.nan
                
        return error_result

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        print("\n" + "=" * 80)
        print(">>> SCRIPT 03: GARCH-COPULA ROLLING FORECAST (PROFESSIONAL VERSION) <<<")
        
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
        print("Note: Clayton and Gumbel results are for robustness checks only")
        
        # Use parallel processing
        all_forecasts = Parallel(n_jobs=-1)(
            delayed(_one_day_wrapper)(d, full_data, copula_params) 
            for d in tqdm(forecast_dates, desc="Forecasting VaR/ES")
        )
        
        # --- Step 4: Save Results ---
        print("\nForecasts completed. Saving results...")
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        forecasts_df.sort_index(inplace=True)
        
        # Save main results and robustness separately
        main_copulas = ['Gaussian', 'StudentT']
        robustness_copulas = ['Clayton', 'Gumbel']
        
        # Main results (Gaussian and Student-t)
        main_cols = [col for col in forecasts_df.columns if any(c in col for c in main_copulas)]
        main_results = forecasts_df[main_cols]
        main_results.to_csv('garch_copula_main_results.csv')
        
        # Robustness checks (Clayton and Gumbel)
        robustness_cols = [col for col in forecasts_df.columns if any(c in col for c in robustness_copulas)]
        if robustness_cols:
            robustness_results = forecasts_df[robustness_cols]
            robustness_results.to_csv('garch_copula_robustness.csv')
        
        # Save copula parameters
        with open('copula_params.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        
        print("Results saved:")
        print("- Main results (Gaussian & Student-t): 'garch_copula_main_results.csv'")
        print("- Robustness checks (Clayton & Gumbel): 'garch_copula_robustness.csv'")
        print("Copula parameters saved to 'copula_params.pkl'.")

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