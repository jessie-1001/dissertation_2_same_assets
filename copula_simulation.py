# SCRIPT 03: GARCH-COPULA MODEL WITH STABILITY ENHANCEMENTS
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
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# 1. STABLE COPULA SAMPLERS
def calc_min_var_weights(vol_spx, vol_dax, rho):
    var_spx, var_dax = vol_spx**2, vol_dax**2
    cov = rho * vol_spx * vol_dax
    w_spx = (var_dax - cov) / (var_spx + var_dax - 2 * cov)
    w_spx = np.clip(w_spx, 0.2, 0.8)
    return w_spx, 1 - w_spx

def sample_gaussian_copula(n_samples, corr_matrix):
    try:
        L = cholesky(corr_matrix, lower=True)
        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_correlated = z @ L.T
        return norm.cdf(z_correlated)
    except np.linalg.LinAlgError:
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
    try:
        L = cholesky(corr_matrix, lower=True)
        g = np.random.chisquare(df, n_samples)
        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_correlated = z @ L.T
        x = np.sqrt(df / g)[:, np.newaxis] * z_correlated
        return t.cdf(x, df=df)
    except np.linalg.LinAlgError:
        return sample_gaussian_copula(n_samples, corr_matrix)

def sample_clayton_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    u2 = (u1**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    return np.column_stack((u1, u2))

def sample_gumbel_copula(n_samples, theta):
    u1 = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)
    gamma_rv = np.random.gamma(1/theta, 1, n_samples)
    e1 = -np.log(u1) / gamma_rv
    e2 = -np.log(v) / gamma_rv
    u1_sim = np.exp(-e1**(1/theta))
    u2_sim = np.exp(-e2**(1/theta))
    return np.column_stack((u1_sim, u2_sim))

def sample_survival_gumbel_copula(n_samples, theta):
    return 1 - sample_gumbel_copula(n_samples, theta)

def sample_survival_clayton_copula(n_samples, theta):
    return 1 - sample_clayton_copula(n_samples, theta)

# 2. ROBUST PARAMETER ESTIMATION
def fit_t_copula_mle(data):
    u = data.values
    kendall_tau = data.corr('kendall').iloc[0, 1]
    rho0 = np.sin(np.pi * kendall_tau / 2)
    bounds = [(-0.99, 0.99), (2.1, 30.0)]
    constraints = [{'type': 'ineq', 'fun': lambda x: 30 - x[1]}]
    def neg_log_likelihood(params):
        rho, df = params
        if abs(rho) >= 0.99 or df <= 2.0:
            return 1e10
        try:
            cov = np.array([[1, rho], [rho, 1]])
            inv_cov = np.linalg.inv(cov)
            det_cov = 1 - rho ** 2
            z = t.ppf(u, df)
            quad_form = np.sum(z @ inv_cov * z, axis=1)
            log_copula = (
                -0.5 * np.log(det_cov)
                - (df + 2)/2 * np.log(1 + quad_form/df)
                + (df/2 + 1) * np.log(1 + (z**2).sum(axis=1)/df)
                - (df/2 + 1) * np.log(1 + z[:, 0]**2/df)
                - (df/2 + 1) * np.log(1 + z[:, 1]**2/df)
            )
            return -np.sum(log_copula)
        except:
            return 1e10
    best_ll = np.inf
    best_params = [rho0, 8]
    for init_df in [5, 10, 15]:
        res = minimize(neg_log_likelihood, [rho0, init_df],
                      method='SLSQP', bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-6})
        if res.fun < best_ll and res.success:
            best_ll = res.fun
            best_params = res.x
    rho_hat, df_hat = best_params
    return {'corr_matrix': np.array([[1, rho_hat], [rho_hat, 1]]), 'df': df_hat}

def get_copula_parameters(data, silent=False):
    if not silent:
        print("Estimating dependence parameters...")
    kendall_tau = data.corr(method='kendall').iloc[0, 1]
    pearson_corr = data.corr(method='pearson').values
    theta_clayton = max(0.01, 2 * kendall_tau / (1 - kendall_tau)) if (1 - kendall_tau) != 0 else 0.01
    theta_gumbel = max(1.01, 1 / (1 - kendall_tau)) if (1 - kendall_tau) != 0 else 1.01
    t_copula_params = fit_t_copula_mle(data)
    params = {
        'Gaussian': {'corr_matrix': pearson_corr},
        'StudentT': t_copula_params,
        'Gumbel': {'theta': theta_gumbel},
        'Clayton': {'theta': theta_clayton}
    }
    if not silent:
        param_data = {
            'Gaussian': {'ρ (Pearson)': pearson_corr[0, 1]},
            'StudentT': {'ρ (MLE)': t_copula_params['corr_matrix'][0, 1], 'ν (DoF, MLE)': t_copula_params['df']},
            'Gumbel': {'θ (from Kendall)': theta_gumbel},
            'Clayton': {'θ (from Kendall)': theta_clayton}
        }
        param_df = pd.DataFrame(param_data).T.fillna('')
        print("\n\n" + "=" * 80)
        print(">>> OUTPUT FOR DISSERTATION: TABLE 4.4 <<<")
        print(param_df.to_markdown(floatfmt=".4f"))
        print("=" * 80 + "\n")
    return params

# 3. STABLE GARCH MODELING
def fit_garch_model(returns, model_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            if model_type == "SPX":
                model = arch_model(
                    returns,
                    mean='Constant',
                    vol='Garch',
                    p=1,
                    o=1,
                    q=1,
                    dist='t',
                    rescale=False
                )
            else:
                model = arch_model(
                    returns,
                    mean='Constant',
                    vol='Garch',
                    p=1,
                    q=1,
                    dist='t',
                    rescale=False
                )
            res = model.fit(disp='off', options={'maxiter': 1000, 'ftol': 1e-5}, update_freq=0)
            return res
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"GARCH fitting failed after {max_retries} attempts: {e}. Using fallback volatility.")
                hist_vol = returns.std()
                class FallbackResult:
                    def __init__(self, vol, nu=5):
                        self.params = {'Const': 0, 'nu': nu}
                        self.conditional_volatility = np.array([vol] * len(returns))
                    def forecast(self, *args, **kwargs):
                        class VarianceForecast:
                            def __init__(self, vol):
                                self.variance = pd.DataFrame([vol ** 2])
                        return VarianceForecast(hist_vol)
                return FallbackResult(hist_vol)
    return None

def run_simulation_for_day(t_index, full_data, copula_params, n_simulations=50000):
    window_data = full_data.loc[:t_index]
    res_spx = fit_garch_model(window_data['SPX_Return'].values * 100, "SPX")
    res_dax = fit_garch_model(window_data['DAX_Return'].values * 100, "DAX")
    try:
        sigma_t1_spx = np.sqrt(res_spx.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except Exception as e:
        print(f"SPX volatility forecast failed: {e}. Using last conditional vol.")
        sigma_t1_spx = res_spx.conditional_volatility[-1]
    try:
        sigma_t1_dax = np.sqrt(res_dax.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
    except Exception as e:
        print(f"DAX volatility forecast failed: {e}. Using last conditional vol.")
        sigma_t1_dax = res_dax.conditional_volatility[-1]
    rho_today = copula_params['Gaussian']['corr_matrix'][0, 1]
    weight_spx, weight_dax = calc_min_var_weights(sigma_t1_spx, sigma_t1_dax, rho_today)
    mu_spx_next = 0
    mu_dax_next = 0
    daily_forecasts = {}
    samplers = {
        'Gaussian': lambda n, p: sample_gaussian_copula(n, p['corr_matrix']),
        'StudentT': lambda n, p: sample_t_copula(n, p['corr_matrix'], p['df']),
        'Gumbel': lambda n, p: sample_survival_gumbel_copula(n, p['theta']),
        'Clayton': lambda n, p: sample_survival_clayton_copula(n, p['theta'])
    }
    nu_spx = res_spx.params.get('nu', 5) if hasattr(res_spx, 'params') else 5
    nu_dax = res_dax.params.get('nu', 5) if hasattr(res_dax, 'params') else 5
    for name, params in copula_params.items():
        try:
            simulated_uniforms = samplers[name](n_simulations, params)
            u_spx = np.clip(simulated_uniforms[:, 0], 1e-4, 1 - 1e-4)
            u_dax = np.clip(simulated_uniforms[:, 1], 1e-4, 1 - 1e-4)
            z_spx = t.ppf(u_spx, df=nu_spx)
            z_dax = t.ppf(u_dax, df=nu_dax)
            r_spx_sim = (mu_spx_next + sigma_t1_spx * z_spx) / 100
            r_dax_sim = (mu_dax_next + sigma_t1_dax * z_dax) / 100
            r_portfolio_sim = weight_spx * r_spx_sim + weight_dax * r_dax_sim
            r_portfolio_sim = np.clip(r_portfolio_sim, -5, 5)
            var_99 = np.percentile(r_portfolio_sim, 1)
            exceed = var_99 - r_portfolio_sim[r_portfolio_sim <= var_99]
            if exceed.size > 30:
                try:
                    shape, loc, scale = stats.genpareto.fit(exceed, floc=0)
                    es_99 = var_99 - (scale + shape * exceed.mean()) / (1 - shape)
                except Exception as e:
                    print(f"GPD fitting failed: {e}. Using sample mean for ES.")
                    es_99 = r_portfolio_sim[r_portfolio_sim <= var_99].mean()
            else:
                es_99 = r_portfolio_sim[r_portfolio_sim <= var_99].mean()
            daily_forecasts[name] = {
                'VaR_99': var_99,
                'ES_99': es_99,
                'Weight_SPX': weight_spx,
                'Weight_DAX': weight_dax,
                'Vol_SPX': sigma_t1_spx,
                'Vol_DAX': sigma_t1_dax
            }
        except Exception as e:
            print(f"Simulation for {name} failed: {e}. Using fallback values.")
            hist_returns = window_data['SPX_Return'] * weight_spx + window_data['DAX_Return'] * weight_dax
            var_99_fallback = np.percentile(hist_returns, 1)
            es_99_fallback = hist_returns[hist_returns <= var_99_fallback].mean()
            daily_forecasts[name] = {
                'VaR_99': var_99_fallback,
                'ES_99': es_99_fallback,
                'Weight_SPX': weight_spx,
                'Weight_DAX': weight_dax,
                'Vol_SPX': sigma_t1_spx,
                'Vol_DAX': sigma_t1_dax
            }
    return daily_forecasts

def _one_day(day, full_data, copula_params):
    t_idx = full_data.index[full_data.index.get_loc(day) - 1]
    forecasts = run_simulation_for_day(t_idx, full_data, copula_params, n_simulations=50000)
    flat = {'Date': day}
    for model, vals in forecasts.items():
        flat[f'{model}_VaR_99'] = vals['VaR_99']
        flat[f'{model}_ES_99'] = vals['ES_99']
        flat[f'{model}_Weight_SPX'] = vals['Weight_SPX']
        flat[f'{model}_Weight_DAX'] = vals['Weight_DAX']
        flat[f'{model}_Vol_SPX'] = vals['Vol_SPX']
        flat[f'{model}_Vol_DAX'] = vals['Vol_DAX']
    return flat

# 5. MAIN EXECUTION BLOCK
if __name__ == '__main__':
    try:
        print("\n" + "=" * 80)
        print(">>> GARCH-COPULA MODEL FOR SPX & DAX <<<")
        copula_input_data = pd.read_csv('copula_input_data.csv', index_col='Date', parse_dates=True).dropna()
        full_data = pd.read_csv('spx_dax_daily_data.csv', index_col='Date', parse_dates=True)
        print("Estimating copula parameters using in-sample data...")
        copula_params = get_copula_parameters(copula_input_data)
        print("Copula parameters estimation complete.\n")
        out_of_sample_start = '2020-01-01'
        forecast_dates = full_data.loc[out_of_sample_start:].index
        print(f"Out-of-sample period: {out_of_sample_start} to {forecast_dates[-1].date()}")
        print(f"Number of forecast days: {len(forecast_dates)}\n")
        all_forecasts = Parallel(n_jobs=8, prefer="processes")(
            delayed(_one_day)(d, full_data, copula_params) for d in tqdm(forecast_dates, desc="Forecasting VaR/ES"))
        forecasts_df = pd.DataFrame(all_forecasts).set_index('Date')
        forecast_output_file = 'garch_copula_forecasts.csv'
        forecasts_df.to_csv(forecast_output_file)
        print(f"\nAll forecasts saved to '{forecast_output_file}'.")
        with open('copula_params.pkl', 'wb') as f:
            pickle.dump(copula_params, f)
        print("Copula parameters saved to 'copula_params.pkl'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
