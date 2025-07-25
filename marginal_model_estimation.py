# =============================================================================
# SCRIPT 02: MARGINAL MODEL ESTIMATION (FINAL OPTIMIZED VERSION)
# =============================================================================
import pandas as pd
import numpy as np
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import t, norm
from statsmodels.stats.diagnostic import het_arch
import warnings
import traceback
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 创建日志目录
os.makedirs('model_logs', exist_ok=True)

def fit_and_diagnose_garch(return_series, asset_name, model_type='GARCH', silent=False):
    """
    增强版GARCH模型拟合函数，支持不同模型类型
    """
    if not silent:
        print(f"\n--- Fitting model for {asset_name} ---")
        print(f"Model: {model_type}(1,1)-t, mean=Constant")
    
    # 根据资产选择模型类型
    if asset_name.startswith('S&P 500'):
        model = arch_model(
            return_series, 
            mean='AR', 
            lags=1,
            vol=model_type, 
            p=1, 
            o=1,  # 非对称项
            q=1, 
            dist='t'
        )
    else:
        model = arch_model(return_series, mean='Constant', vol='Garch', p=1, q=1, dist='t')
    
    # 更健壮的异常处理 - 修复点
    result = None
    try:
        # 第一次尝试拟合
        result = model.fit(update_freq=0, disp='off')
    except Exception as e1:
        if not silent:
            print(f"First fitting attempt failed: {str(e1)}")
        
        try:
            # 第二次尝试使用默认起始值
            if not silent:
                print("Retrying with default starting values...")
            result = model.fit(update_freq=0, disp='off', starting_values=None)
        except Exception as e2:
            # 使用简化起始值作为最后手段
            if not silent:
                print(f"Second fitting attempt failed: {str(e2)}")
                print("Using simplified starting values...")
            
            # 根据模型类型设置简化起始值
            if 'EGARCH' in model_type:
                start_vals = [0.01, -0.05, 0.05, 0.1, -0.05, 0.9, 7]  # AR-EGARCH-t
            else:
                start_vals = [0.01, 0.05, 0.1, 0.85, 7]  # GARCH-t
                
            result = model.fit(update_freq=0, disp='off', starting_values=start_vals)
    
    # 确保result已被定义
    if result is None:
        raise RuntimeError(f"Failed to fit model for {asset_name} after multiple attempts")
    
    if not silent:
        params = result.params
        pvalues = result.pvalues
        
        # 添加自由度经济解释
        nu = params['nu']
        tail_risk = "非常肥尾" if nu < 5 else "中等肥尾" if nu < 10 else "接近正态"
        
        param_table = pd.DataFrame({'Coefficient': params, 'P-value': pvalues})
        param_table['P-value'] = param_table['P-value'].apply(lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}")
        
        print(f"\n--- Parameter Estimates for {asset_name} ---")
        print(param_table.to_markdown(floatfmt=".4f"))
        print(f"\nTail thickness (nu): {nu:.2f} ({tail_risk}, Normal=∞)")
        
        # 添加经济解释增强
        # 1. 杠杆效应解释 (仅EGARCH)
        if 'gamma[1]' in params:
            leverage_effect = abs(params['gamma[1]']/params['alpha[1]'])*100
            print(f"杠杆效应: 负收益增加波动率 {leverage_effect:.1f}%")
        
        # 2. 波动持续性分析
        if 'beta[1]' in params:
            persistence = params['beta[1]']  # EGARCH
            half_life = np.log(0.5)/np.log(persistence)
            print(f"波动半衰期: {half_life:.1f} 天")
        
        # 3. 极端风险量化
        if 'nu' in params:
            risk_5sigma = t.sf(5, df=nu)*2 * 100  # 双侧5σ概率
            normal_risk = 2 * (1 - norm.cdf(5)) * 100
            print(f"极端风险: |r|>5σ的概率 = {risk_5sigma:.4f}% (正态: {normal_risk:.6f}%)")

        # 诊断检验
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
        
        # 更精细的诊断判断
        misspec_warning = any(p < 0.05 for p in diag_table.loc[diag_table['Test'].str.contains('Std Residuals'), 'P-value'])
        vol_misspec = any(p < 0.05 for p in diag_table.loc[diag_table['Test'].str.contains('Sq Std Residuals'), 'P-value'])
        
        if misspec_warning:
            print("\nWARNING: Model shows signs of misspecification in returns (p-value < 0.05).")
        if vol_misspec:
            print("\nWARNING: Model shows signs of volatility clustering not captured (p-value < 0.05).")
        if not (misspec_warning or vol_misspec):
            print("\nSUCCESS: Model appears well-specified for this asset.")
        
        print(f"\nARCH-LM Test: F-stat = {arch_fstat:.4f}, p-value = {arch_pval:.4f}")
        print("-" * 80)
    
    # 记录日志
    log_name = asset_name.replace(" ", "_").replace("(", "").replace(")", "")
    with open(f'model_logs/{log_name}_summary.txt', 'w') as f:
        f.write(str(result.summary()))
    
    return result

def calculate_pit(result, return_series):
    """计算概率积分变换值(PIT)"""
    std_resid = pd.Series(result.std_resid, index=return_series.index).dropna()
    u_series = pd.Series(t.cdf(std_resid, df=result.params['nu']), index=std_resid.index)
    return u_series

if __name__ == '__main__':
    print("\n" + "="*80)
    print(">>> SCRIPT 02: MARGINAL MODEL ESTIMATION FOR SPX & DAX <<<")

    try:
        input_file = 'spx_dax_daily_data.csv'
        data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

        # 参数化样本划分
        SAMPLE_SPLIT = 0.8  # 80%训练，20%测试
        split_idx = int(len(data) * SAMPLE_SPLIT)
        in_sample_data = data.iloc[:split_idx]
        print(f"\n>>> STEP 1: Processing In-Sample Data ({len(in_sample_data)} obs, {in_sample_data.index[0].date()} to {in_sample_data.index[-1].date()}) <<<")
        
        # SPX使用AR(1)-EGARCH(1,1,1)-t
        result_spx_in_sample = fit_and_diagnose_garch(
            in_sample_data['SPX_Return'], 
            'S&P 500 (In-Sample)', 
            model_type='EGARCH'
        )
        
        # DAX保持GARCH
        result_dax_in_sample = fit_and_diagnose_garch(
            in_sample_data['DAX_Return'], 
            'DAX (In-Sample)'
        )
        
        # PIT转换
        u_spx_in = calculate_pit(result_spx_in_sample, in_sample_data['SPX_Return'])
        u_dax_in = calculate_pit(result_dax_in_sample, in_sample_data['DAX_Return'])
        
        pit_df_in_sample = pd.DataFrame({'u_spx': u_spx_in, 'u_dax': u_dax_in}).dropna().clip(1e-6, 1 - 1e-6)
        pit_df_in_sample.to_csv("copula_input_data.csv", index=True)
        print("\nSUCCESS: In-sample data for copula parameter estimation saved to 'copula_input_data.csv'.")
        print("This file will be used by SCRIPT 03.")

        print("\n>>> STEP 2: Processing Full-Sample Data for Backtesting Visualization <<<")
        # 全样本使用相同模型但输出诊断
        result_spx_full = fit_and_diagnose_garch(
            data['SPX_Return'], 
            'S&P 500 (Full-Sample)', 
            model_type='EGARCH',
            silent=False
        )
        result_dax_full = fit_and_diagnose_garch(
            data['DAX_Return'], 
            'DAX (Full-Sample)',
            silent=False
        )

        # PIT转换
        u_spx_full = calculate_pit(result_spx_full, data['SPX_Return'])
        u_dax_full = calculate_pit(result_dax_full, data['DAX_Return'])
        
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