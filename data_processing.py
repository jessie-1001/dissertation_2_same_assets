# =============================================================================
# DATA ACQUISITION & PROCESSING – SPX and DAX  (end‑date fixed to 2025‑06‑01)
# =============================================================================
"""
Downloads daily closes for the S&P 500 (^GSPC) and DAX (^GDAXI),
cleans the data, computes log‑returns, performs data‑quality checks,
and writes results to CSV and PNG.

Key points
----------
* End‑date is intentionally fixed at 2025‑06‑01 for thesis reproducibility.
* Robust log‑return computation via log1p(pct_change).
* Dual extreme‑move detection: ±5 % and ±3 standard deviations.
* True gaps detected with a business‑day calendar.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera

START_DATE = "2007-01-01"
END_DATE   = "2025-06-01"          # thesis cut‑off
TICKERS    = ["^GSPC", "^GDAXI"]
OUTPUT_CSV = "spx_dax_daily_data.csv"
PLOT_DIR   = "data_quality_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def data_processing_and_summary() -> None:
    # ---------------------------------------------------------------- download
    print(f"Downloading {TICKERS} from {START_DATE} to {END_DATE} …")
    close = (
        yf.download(TICKERS, start=START_DATE, end=END_DATE,
                    progress=False)["Close"]
        .rename(columns={"^GSPC": "SPX", "^GDAXI": "DAX"})
    )
    print(f"Downloaded {len(close)} rows.")

    # ---------------------------------------------------------- clean & align
    close = close.dropna()
    print(f"Rows after NA removal: {len(close)}")

    # --------------------------------------------------- log‑return (% scale)
    ret = 100 * np.log1p(close.pct_change())
    ret = ret.rename(columns={"SPX": "SPX_Return", "DAX": "DAX_Return"})

    # ------------------------------------------------ combine & persist
    full = pd.concat([close, ret], axis=1).dropna()
    full.to_csv(OUTPUT_CSV)
    print(f"Cleaned data saved → {OUTPUT_CSV}")

    # ----------------------------------------- descriptive statistics table
    returns = full[["SPX_Return", "DAX_Return"]]
    desc = returns.describe().T
    desc["Skewness"]        = returns.skew()
    desc["Excess Kurtosis"] = returns.kurtosis()
    jb_spx = jarque_bera(returns["SPX_Return"])
    jb_dax = jarque_bera(returns["DAX_Return"])
    desc["Jarque‑Bera"] = [jb_spx.statistic, jb_dax.statistic]
    desc["JB p‑value"]  = [jb_spx.pvalue,     jb_dax.pvalue]
    print("\nDescriptive statistics (daily log‑returns, %):\n",
          desc[["mean", "std", "min", "max",
                "Skewness", "Excess Kurtosis",
                "Jarque‑Bera", "JB p‑value"]].to_markdown(floatfmt=".4f"))

    # ---------------------------------------------------- extreme‑move count
    thr_spx = 3 * returns["SPX_Return"].std()
    thr_dax = 3 * returns["DAX_Return"].std()
    n_ext_spx_5  = (returns["SPX_Return"].abs() > 5).sum()
    n_ext_dax_5  = (returns["DAX_Return"].abs() > 5).sum()
    n_ext_spx_3s = (returns["SPX_Return"].abs() > thr_spx).sum()
    n_ext_dax_3s = (returns["DAX_Return"].abs() > thr_dax).sum()
    print(f"\nExtreme moves | |r|>5 %  SPX: {n_ext_spx_5}  DAX: {n_ext_dax_5}")
    print(f"               | |r|>3 σ SPX: {n_ext_spx_3s} DAX: {n_ext_dax_3s}")

    # --------------------------------------------------------- gap analysis
    bdays = pd.bdate_range(START_DATE, END_DATE)
    missing = bdays.difference(full.index)
    multi_day_gaps = (
        missing.to_series().diff().dt.days.fillna(1).loc[lambda s: s > 1]
    )
    if not multi_day_gaps.empty:
        print("\nPotential data gaps (may include holidays):")
        print(multi_day_gaps.value_counts().to_markdown())
    else:
        print("\nNo business‑day gaps detected.")

    # -------------------------------------------------------------- visuals
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1); close["SPX"].plot(title="S&P 500 level"); plt.grid(True)
    plt.subplot(2, 2, 2); close["DAX"].plot(title="DAX level", color="green"); plt.grid(True)
    plt.subplot(2, 2, 3); sns.histplot(returns["SPX_Return"], kde=True,
                                       bins=60, color="blue")
    plt.title("S&P 500 return distribution")
    plt.subplot(2, 2, 4); sns.histplot(returns["DAX_Return"], kde=True,
                                       bins=60, color="green")
    plt.title("DAX return distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/price_return_overview.png")
    plt.close()
    print(f"Plots saved → {PLOT_DIR}/price_return_overview.png")


if __name__ == "__main__":
    data_processing_and_summary()
