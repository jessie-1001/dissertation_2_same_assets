# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION AND DIAGNOSTICS (REVISED & CORRECTED)
# =============================================================================

import pandas as pd
import numpy as np
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.diagnostic import het_arch
import warnings
import traceback

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def fit_and_diagnose_garch(return_series, asset_name, silent=False):
    """
    Fits a GARCH(1,1)-t model and performs diagnostic checks.
    The 'silent' flag can suppress print output for secondary tasks.
    """
    if not silent:
        print(f"\n--- Fitting model for {asset_name} ---")

    # The primary model from literature: mean='Constant', vol='Garch', p=1, q=1, dist='t'
    model = arch_model(return_series, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    model_desc = "GARCH(1,1)-t, mean=Constant"
    if not silent:
        print(f"Model: {model_desc}")

    # Use multiple attempts for robust optimization
    best_result = None
    best_aic = np.inf
    for i in range(3):
        try:
            result = model.fit(update_freq=0, disp='off')
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
        except Exception as e:
            if not silent:
                print(f"   [!] Optimization attempt {i+1} failed: {str(e)}")
            continue

    if best_result is None:
        if not silent:
            print("   [!] All optimization attempts failed, using one final attempt.")
        best_result = model.fit(update_freq=0, disp='off')

    result = best_result

    if not silent:
        # --- Parameter Table ---
        params = result.params
        pvalues = result.pvalues
        param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
        param_table['P-value'] = param_table['P-value'].apply(lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}")
        print(f"\n--- Parameter Estimates for {asset_name} ---")
        print(param_table.to_markdown(floatfmt=".4f"))

        # --- Diagnostic Tests ---
        std_resid = pd.Series(result.std_resid).dropna()
        lags_to_test = [5, 10, 20]
        diag_rows = []
        for lag in lags_to_test:
            lb1 = sm.stats.acorr_ljungbox(std_resid, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
            lb2 = sm.stats.acorr_ljungbox(std_resid**2, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
            diag_rows.append({'Test': f'Ljung-Box on Std Residuals (Lags={lag})', 'P-value': lb1})
            diag_rows.append({'Test': f'Ljung-Box on Sq Std Residuals (Lags={lag})', 'P-value': lb2})

        diag_table = pd.DataFrame(diag_rows)
        print(f"\n--- Diagnostic Tests for {asset_name} ---")
        print(diag_table.to_markdown(index=False, floatfmt=".4f"))

        arch_test = het_arch(std_resid)
        arch_fstat, arch_pval = arch_test[0], arch_test[1]

        if any(p < 0.05 for p in diag_table['P-value']) or arch_pval < 0.05:
            print("\nWARNING: Model shows signs of misspecification (p-value < 0.05).")
        else:
            print("\nSUCCESS: Model appears well-specified for this asset.")

        print(f"\nARCH-LM Test: F-stat = {arch_fstat:.4f}, p-value = {arch_pval:.4f}")
        print("-" * 80)

    return result

# ===== MAIN SCRIPT EXECUTION =====
if __name__ == '__main__':
    print("\n" + "="*80)
    print(">>> MARGINAL MODEL ESTIMATION FOR SPX & DAX <<<")

    try:
        input_file = 'spx_dax_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

        # --- 1. In-Sample Fitting (for Copula Parameter Estimation) ---
        print("\n>>> STEP 1: Processing In-Sample Data for Forecasting Parameters <<<")
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end]

        spx_returns_in_sample = in_sample_data['SPX_Return']
        dax_returns_in_sample = in_sample_data['DAX_Return']

        print(f"In-sample size: {len(spx_returns_in_sample)} observations")
        print(f"In-sample date range: {spx_returns_in_sample.index[0].date()} to {spx_returns_in_sample.index[-1].date()}")

        result_spx_in_sample = fit_and_diagnose_garch(spx_returns_in_sample, 'S&P 500 (In-Sample)')
        result_dax_in_sample = fit_and_diagnose_garch(dax_returns_in_sample, 'DAX (In-Sample)')

        # PIT Transform on in-sample standardized residuals
        std_resid_spx = pd.Series(result_spx_in_sample.std_resid).dropna()
        nu_spx = result_spx_in_sample.params['nu']
        u_spx = t.cdf(std_resid_spx, df=nu_spx)

        std_resid_dax = pd.Series(result_dax_in_sample.std_resid).dropna()
        nu_dax = result_dax_in_sample.params['nu']
        u_dax = t.cdf(std_resid_dax, df=nu_dax)

        # Create and save the in-sample PIT data
        pit_df_in_sample = pd.DataFrame({'u_spx': u_spx, 'u_dax': u_dax}).dropna().clip(1e-6, 1 - 1e-6)
        pit_df_in_sample.index.name = 'Date'
        pit_df_in_sample.to_csv("copula_input_data.csv", index=True)
        print("\nSUCCESS: In-sample data for copula parameter estimation saved to 'copula_input_data.csv'.")
        print("This file will be used by SCRIPT 03.")

        # --- 2. Full-Sample Fitting (for Visualization & Analysis) ---
        print("\n>>> STEP 2: Processing Full-Sample Data for Backtesting Visualization <<<")
        
        # Fit GARCH models on the entire dataset
        # We use silent=True to avoid cluttering the output, as diagnostics were already run
        result_spx_full = fit_and_diagnose_garch(data['SPX_Return'], 'S&P 500 (Full-Sample)', silent=True)
        result_dax_full = fit_and_diagnose_garch(data['DAX_Return'], 'DAX (Full-Sample)', silent=True)
        
        # PIT Transform on full-sample standardized residuals
        full_std_resid_spx = pd.Series(result_spx_full.std_resid).dropna()
        full_nu_spx = result_spx_full.params['nu']
        full_u_spx = t.cdf(full_std_resid_spx, df=full_nu_spx)

        full_std_resid_dax = pd.Series(result_dax_full.std_resid).dropna()
        full_nu_dax = result_dax_full.params['nu']
        full_u_dax = t.cdf(full_std_resid_dax, df=full_nu_dax)
        
        # Create and save the full-sample PIT data
        pit_df_full = pd.DataFrame({'u_spx': full_u_spx, 'u_dax': full_u_dax}).dropna().clip(1e-6, 1 - 1e-6)
        pit_df_full.index.name = 'Date'
        pit_df_full.to_csv("copula_input_data_full.csv", index=True)
        print("\nSUCCESS: Full-sample data for visualization saved to 'copula_input_data_full.csv'.")
        print("This file should be used by SCRIPT 04 for plotting.")


    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please run SCRIPT 01 first.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("Script 02 finished successfully.")
    print("="*80 + "\n")