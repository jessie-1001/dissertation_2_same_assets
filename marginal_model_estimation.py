# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION (FINAL CORRECTED VERSION)
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
    if not silent:
        print(f"\n--- Fitting model for {asset_name} ---")
    model = arch_model(return_series, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    if not silent:
        print(f"Model: GARCH(1,1)-t, mean=Constant")
    try:
        result = model.fit(update_freq=0, disp='off')
    except Exception:
        result = model.fit(update_freq=0, disp='off', starting_values=result.params)

    if not silent:
        params = result.params
        pvalues = result.pvalues
        param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
        param_table['P-value'] = param_table['P-value'].apply(lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}")
        print(f"\n--- Parameter Estimates for {asset_name} ---")
        print(param_table.to_markdown(floatfmt=".4f"))

        std_resid = pd.Series(result.std_resid, index=return_series.index).dropna()
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
        if any(p < 0.05 for p in diag_table['P-value']) and diag_table['Test'].str.contains('Std Residuals').any():
            print("\nWARNING: Model shows signs of misspecification (p-value < 0.05).")
        else:
            print("\nSUCCESS: Model appears well-specified for this asset.")
        print(f"\nARCH-LM Test: F-stat = {arch_fstat:.4f}, p-value = {arch_pval:.4f}")
        print("-" * 80)
    return result

if __name__ == '__main__':
    print("\n" + "="*80)
    print(">>> SCRIPT 02: MARGINAL MODEL ESTIMATION FOR SPX & DAX <<<")

    try:
        input_file = 'spx_dax_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

        print("\n>>> STEP 1: Processing In-Sample Data for Forecasting Parameters <<<")
        in_sample_end = '2019-12-31'
        in_sample_data = data.loc[:in_sample_end].dropna()
        result_spx_in_sample = fit_and_diagnose_garch(in_sample_data['SPX_Return'], 'S&P 500 (In-Sample)')
        result_dax_in_sample = fit_and_diagnose_garch(in_sample_data['DAX_Return'], 'DAX (In-Sample)')
        std_resid_spx_in = pd.Series(result_spx_in_sample.std_resid, index=in_sample_data.index).dropna()
        std_resid_dax_in = pd.Series(result_dax_in_sample.std_resid, index=in_sample_data.index).dropna()
        u_spx_in = pd.Series(t.cdf(std_resid_spx_in, df=result_spx_in_sample.params['nu']), index=std_resid_spx_in.index)
        u_dax_in = pd.Series(t.cdf(std_resid_dax_in, df=result_dax_in_sample.params['nu']), index=std_resid_dax_in.index)
        pit_df_in_sample = pd.DataFrame({'u_spx': u_spx_in, 'u_dax': u_dax_in}).dropna().clip(1e-6, 1 - 1e-6)
        pit_df_in_sample.to_csv("copula_input_data.csv", index=True)
        print("\nSUCCESS: In-sample data for copula parameter estimation saved to 'copula_input_data.csv'.")
        print("This file will be used by SCRIPT 03.")

        print("\n>>> STEP 2: Processing Full-Sample Data for Backtesting Visualization <<<")
        result_spx_full = fit_and_diagnose_garch(data['SPX_Return'], 'S&P 500 (Full-Sample)', silent=True)
        result_dax_full = fit_and_diagnose_garch(data['DAX_Return'], 'DAX (Full-Sample)', silent=True)

        # <<< FIX: Correctly create Series with index BEFORE making the final DataFrame >>>
        std_resid_spx_full = pd.Series(result_spx_full.std_resid, index=data.index).dropna()
        std_resid_dax_full = pd.Series(result_dax_full.std_resid, index=data.index).dropna()
        u_spx_full = pd.Series(t.cdf(std_resid_spx_full, df=result_spx_full.params['nu']), index=std_resid_spx_full.index)
        u_dax_full = pd.Series(t.cdf(std_resid_dax_full, df=result_dax_full.params['nu']), index=std_resid_dax_full.index)
        
        pit_df_full = pd.DataFrame({'u_spx': u_spx_full, 'u_dax': u_dax_full}).dropna().clip(1e-6, 1 - 1e-6)
        pit_df_full.to_csv("copula_input_data_full.csv", index=True)
        print("\nSUCCESS: Full-sample data for visualization saved to 'copula_input_data_full.csv'.")
        print("This file will be used by SCRIPT 04 for plotting.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("Script 02 finished successfully.")
    print("="*80 + "\n")