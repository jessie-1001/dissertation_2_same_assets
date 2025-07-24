# =============================================================================
# SCRIPT 01: DATA ACQUISITION AND PROCESSING (ENHANCED) - MODIFIED FOR SPX & DAX
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis
import seaborn as sns
import os

def data_processing_and_summary():
    """
    Enhanced function to download, process, save, and summarize financial data
    for S&P 500 and DAX indices.
    """
    # --- 1. Set up parameters ---
    tickers = ['^GSPC', '^GDAXI']  # S&P 500 and DAX indices
    start_date = '2007-01-01'
    end_date = '2025-06-01'
    output_file = 'spx_dax_daily_data.csv'
    plot_dir = 'data_quality_plots/'
    os.makedirs(plot_dir, exist_ok=True)

    # --- 2. Download historical price data ---
    print(f"Downloading daily data for {tickers} from {start_date} to {end_date}...")
    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        print(f"Data download complete. Retrieved {len(price_data)} days of data.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    # --- 3. Clean and process the data ---
    price_data.rename(columns={'^GSPC': 'SPX', '^GDAXI': 'DAX'}, inplace=True)
    price_data_aligned = price_data.dropna()
    
    print(f"\nData alignment:")
    print(f"Original rows: {len(price_data)}, After alignment: {len(price_data_aligned)}")
    print(f"Removed {len(price_data) - len(price_data_aligned)} rows due to missing values.")
    
    # Calculate log returns
    return_data = 100 * np.log(price_data_aligned / price_data_aligned.shift(1))
    return_data.rename(columns={'SPX': 'SPX_Return', 'DAX': 'DAX_Return'}, inplace=True)
    
    # Combine price and return data
    final_data = pd.concat([price_data_aligned, return_data], axis=1).dropna()
    print(f"Final dataset size: {len(final_data)} rows after removing leading NaN.")
    
    # Save data
    final_data.to_csv(output_file)
    print(f"\nCleaned data saved successfully to '{output_file}'.")

    # --- 4. Generate and print Descriptive Statistics Table (Table 4.1) ---
    print("\n\n" + "="*80)
    print(">>> DESCRIPTIVE STATISTICS: SPX & DAX RETURNS <<<")
    
    returns = final_data[['SPX_Return', 'DAX_Return']]
    
    # Calculate descriptive statistics
    desc_stats = returns.describe().T
    
    # Add skewness, kurtosis, and Jarque-Bera test results
    desc_stats['Skewness'] = returns.skew()
    desc_stats['Kurtosis'] = returns.kurtosis()  # Excess kurtosis
    
    # Jarque-Bera test
    jb_spx = jarque_bera(returns['SPX_Return'].dropna())
    jb_dax = jarque_bera(returns['DAX_Return'].dropna())
    desc_stats['Jarque-Bera'] = [jb_spx[0], jb_dax[0]]
    desc_stats['JB p-value'] = [jb_spx[1], jb_dax[1]]
    
    # Format table
    desc_stats_formatted = desc_stats[[
        'mean', 'std', 'min', 'max', 'Skewness', 'Kurtosis', 'Jarque-Bera', 'JB p-value'
    ]]
    desc_stats_formatted.columns = [
        'Mean', 'Std. Dev.', 'Min', 'Max', 'Skewness', 'Excess Kurtosis', 
        'Jarque-Bera', 'p-value'
    ]
    
    print("\n--- Descriptive Statistics of Daily Log-Returns ---")
    print(desc_stats_formatted.to_markdown(floatfmt=".4f"))
    print("="*80 + "\n")

    # --- 5. Enhanced Data Quality Checks ---
    print("\n" + "="*80)
    print(">>> ENHANCED DATA QUALITY CHECKS <<<")
    
    # 5.1 Extreme value detection
    print("\n--- Extreme Value Detection ---")
    extreme_spx = final_data[(final_data['SPX_Return'] < -0.05) | (final_data['SPX_Return'] > 0.05)]
    extreme_dax = final_data[(final_data['DAX_Return'] < -0.05) | (final_data['DAX_Return'] > 0.05)]
    
    print(f"SPX extreme returns (<-5% or >5%): {len(extreme_spx)} events")
    print(f"DAX extreme returns (<-5% or >5%): {len(extreme_dax)} events")
    
    # 5.2 Data continuity check
    print("\n--- Data Continuity Check ---")
    trading_days = final_data.asfreq('B').index
    date_diff = trading_days.to_series().diff().dt.days
    gap_days = date_diff[date_diff > 3]  # Ignore weekends
        
    if not gap_days.empty:
        print(f"Data gaps detected ({len(gap_days)} gaps):")
        gap_summary = gap_days.value_counts().reset_index()
        gap_summary.columns = ['Gap Size (days)', 'Count']
        print(gap_summary.to_markdown(index=False))
    else:
        print("No significant data gaps detected (all consecutive trading days).")
    
    # 5.3 Visualizations
    print("\n--- Generating Data Quality Visualizations ---")
    plt.figure(figsize=(14, 10))
    
    # Price series
    plt.subplot(2, 2, 1)
    final_data['SPX'].plot(title='S&P 500 Price Series', color='blue')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    final_data['DAX'].plot(title='DAX Index', color='green')
    plt.grid(True)
    
    # Return distributions
    plt.subplot(2, 2, 3)
    sns.histplot(final_data['SPX_Return'], kde=True, color='blue', bins=50)
    plt.title('S&P 500 Return Distribution')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.subplot(2, 2, 4)
    sns.histplot(final_data['DAX_Return'], kde=True, color='green', bins=50)
    plt.title('DAX Return Distribution')
    plt.axvline(x=0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}price_and_return_distributions.png")
    plt.close()
    print(f"- Saved price and return distributions to {plot_dir}price_and_return_distributions.png")
    
    # Extreme events
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    final_data['SPX_Return'].plot(title='S&P 500 Daily Returns', color='blue', alpha=0.7)
    plt.scatter(extreme_spx.index, extreme_spx['SPX_Return'], color='red', s=30, label='Extreme Returns')
    plt.axhline(y=-0.05, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    final_data['DAX_Return'].plot(title='DAX Daily Returns', color='green', alpha=0.7)
    plt.scatter(extreme_dax.index, extreme_dax['DAX_Return'], color='red', s=30, label='Extreme Returns')
    plt.axhline(y=-0.05, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}extreme_return_events.png")
    plt.close()
    print(f"- Saved extreme return events plot to {plot_dir}extreme_return_events.png")
    
    # 5.4 Data quality report
    print("\n" + "="*80)
    print(">>> DATA QUALITY REPORT SUMMARY <<<")
    print(f"- Total observations: {len(final_data)}")
    print(f"- Date range: {final_data.index[0].date()} to {final_data.index[-1].date()}")
    print(f"- SPX missing values: {final_data['SPX'].isnull().sum()}")
    print(f"- DAX missing values: {final_data['DAX'].isnull().sum()}")
    print(f"- SPX extreme returns: {len(extreme_spx)} ({len(extreme_spx)/len(final_data)*100:.2f}%)")
    print(f"- DAX extreme returns: {len(extreme_dax)} ({len(extreme_dax)/len(final_data)*100:.2f}%)")
    print(f"- Data gaps: {len(gap_days)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    data_processing_and_summary()