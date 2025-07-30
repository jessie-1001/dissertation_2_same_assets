#!/usr/bin/env python3
# =============================================================================
# DATA ACQUISITION & QUALITY CHECK – SPX and DAX (Leakage-Free)
# =============================================================================
"""
Downloads daily closes for S&P 500 (^GSPC) and DAX (^GDAXI),
computes daily log-returns, performs descriptive stats &
extreme-move detection, outputs CSV + diagnostic plots.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera

from config import Config

# ------------------------------ CONFIG SECTION ------------------------------ #
# Define output file paths using the central config
OUTPUT_CSV              = os.path.join(Config.DATA_DIR, "spx_dax_daily_data.csv")
EXTREME_EVENTS_CSV      = os.path.join(Config.DATA_DIR, "extreme_return_events.csv")
OVERVIEW_PNG            = os.path.join(Config.DATA_DIR, "price_return_overview.png")
EXTREME_RETURNS_PNG     = os.path.join(Config.DATA_DIR, "extreme_returns.png")

# Other script-specific settings (can be moved to config.py if preferred)
THRESHOLD_MODE          = "dual"    # "fixed"  or "dual"
FIXED_PCT               = 5         # 5  ->  ±5%
K_SIGMA                 = 3         # used when THRESHOLD_MODE="dual"
SHOW_GAP_SUMMARY        = True      # list multi-day data gaps?
DRAW_THRESHOLD_LINES    = True      # draw ±threshold on plots?
# ---------------------------------------------------------------------------- #

os.makedirs(Config.DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------

def download_prices() -> pd.DataFrame:
    print(f"Downloading {Config.TICKERS} from {Config.START_DATE} to {Config.END_DATE} …")
    close = (
        yf.download(
            Config.TICKERS,
            start=Config.START_DATE,
            end=Config.END_DATE,
            progress=False,
            auto_adjust=True,
            threads=False,
            repair=True,
        )["Close"]
        .rename(columns={"^GSPC": "SPX", "^GDAXI": "DAX"})
    )
    print(f"Downloaded {len(close)} rows.")
    return close.dropna()

def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    ret = 100 * np.log1p(close.pct_change())
    return ret.rename(columns={"SPX": "SPX_Return", "DAX": "DAX_Return"})

def descriptive_stats(returns: pd.DataFrame) -> None:
    desc = returns.describe().T
    desc["Skewness"]        = returns.skew()
    desc["Excess_Kurtosis"] = returns.kurtosis()

    jb_spx = jarque_bera(returns["SPX_Return"].dropna())
    jb_dax = jarque_bera(returns["DAX_Return"].dropna())
    desc["Jarque_Bera"] = [jb_spx.statistic, jb_dax.statistic]
    desc["JB_p_value"]  = [jb_spx.pvalue,     jb_dax.pvalue]

    # Format for printing
    printable = (
        desc[
            [
                "mean",
                "std",
                "min",
                "max",
                "Skewness",
                "Excess_Kurtosis",
                "Jarque_Bera",
                "JB_p_value",
            ]
        ]
        .rename(
            columns={
                "mean": "Mean",
                "std": "Std. Dev.",
                "min": "Min",
                "max": "Max",
                "Excess_Kurtosis": "Excess Kurtosis",
                "Jarque_Bera": "Jarque-Bera",
                "JB_p_value": "p-value",
            }
        )
    )

    print("\n" + "=" * 80)
    print(">>> DESCRIPTIVE STATISTICS: SPX & DAX RETURNS <<<\n")
    print(printable.to_markdown(floatfmt=".4f"))
    print("=" * 80 + "\n")

def gap_summary(full_index: pd.DatetimeIndex) -> None:
    if not SHOW_GAP_SUMMARY:
        return
    bdays = pd.bdate_range(Config.START_DATE, Config.END_DATE)
    missing = bdays.difference(full_index)
    gaps = missing.to_series().diff().dt.days.fillna(1).loc[lambda s: s > 1]
    if gaps.empty:
        print("No multi-day gaps detected.")
        return
    print("\n--- Data Gaps ---")
    gap_tbl = gaps.value_counts().rename_axis("Gap Size (days)").to_frame("Count")
    print(gap_tbl.to_markdown())

def overview_plot(close: pd.DataFrame, returns: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 10))

    # Price levels
    plt.subplot(2, 2, 1)
    close["SPX"].plot(color="royalblue")
    plt.title("S&P 500 Price Series")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    close["DAX"].plot(color="seagreen")
    plt.title("DAX Index")
    plt.grid(True)

    # Distributions
    plt.subplot(2, 2, 3)
    sns.histplot(returns["SPX_Return"], bins=60, kde=True, color="royalblue")
    plt.title("S&P 500 Return Distribution")
    plt.axvline(0, ls="--", color="red")

    plt.subplot(2, 2, 4)
    sns.histplot(returns["DAX_Return"], bins=60, kde=True, color="seagreen")
    plt.title("DAX Return Distribution")
    plt.axvline(0, ls="--", color="red")

    plt.tight_layout()
    plt.savefig(OVERVIEW_PNG)
    plt.close()
    print(f"• Overview plot saved → {OVERVIEW_PNG}")

def detect_extremes(
    returns: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, float, float]:
    """Return cond_spx, cond_dax boolean Series + thresholds."""
    thr_spx = K_SIGMA * returns["SPX_Return"].std()
    thr_dax = K_SIGMA * returns["DAX_Return"].std()

    if THRESHOLD_MODE == "fixed":
        cond_spx = returns["SPX_Return"].abs() > FIXED_PCT
        cond_dax = returns["DAX_Return"].abs() > FIXED_PCT
    elif THRESHOLD_MODE == "dual":
        cond_spx = (returns["SPX_Return"].abs() > FIXED_PCT) | (
            returns["SPX_Return"].abs() > thr_spx
        )
        cond_dax = (returns["DAX_Return"].abs() > FIXED_PCT) | (
            returns["DAX_Return"].abs() > thr_dax
        )
    else:
        raise ValueError("THRESHOLD_MODE must be 'fixed' or 'dual'.")

    return cond_spx, cond_dax, thr_spx, thr_dax

def extreme_plot(
    returns: pd.DataFrame,
    cond_spx: pd.Series,
    cond_dax: pd.Series,
    thr_spx: float,
    thr_dax: float,
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # S&P
    ax[0].plot(
        returns.index,
        returns["SPX_Return"],
        color="royalblue",
        lw=0.7,
        alpha=0.8,
        label="SPX_Return",
    )
    ax[0].scatter(
        returns.index[cond_spx],
        returns.loc[cond_spx, "SPX_Return"],
        color="red",
        s=20,
        zorder=5,
        label="Extreme Returns",
    )
    if DRAW_THRESHOLD_LINES:
        ax[0].axhline(FIXED_PCT, ls="--", color="grey", lw=0.8)
        ax[0].axhline(-FIXED_PCT, ls="--", color="grey", lw=0.8)
        if THRESHOLD_MODE == "dual":
            ax[0].axhline(thr_spx, ls=":", color="grey", lw=0.8)
            ax[0].axhline(-thr_spx, ls=":", color="grey", lw=0.8)
    ax[0].axhline(0, ls="--", color="black", lw=0.6, alpha=0.5)
    ax[0].set_title("S&P 500 Daily Returns")
    ax[0].legend()

    # DAX
    ax[1].plot(
        returns.index,
        returns["DAX_Return"],
        color="seagreen",
        lw=0.7,
        alpha=0.8,
        label="DAX_Return",
    )
    ax[1].scatter(
        returns.index[cond_dax],
        returns.loc[cond_dax, "DAX_Return"],
        color="red",
        s=20,
        zorder=5,
        label="Extreme Returns",
    )
    if DRAW_THRESHOLD_LINES:
        ax[1].axhline(FIXED_PCT, ls="--", color="grey", lw=0.8)
        ax[1].axhline(-FIXED_PCT, ls="--", color="grey", lw=0.8)
        if THRESHOLD_MODE == "dual":
            ax[1].axhline(thr_dax, ls=":", color="grey", lw=0.8)
            ax[1].axhline(-thr_dax, ls=":", color="grey", lw=0.8)
    ax[1].axhline(0, ls="--", color="black", lw=0.6, alpha=0.5)
    ax[1].set_title("DAX Daily Returns")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(EXTREME_RETURNS_PNG)
    plt.close()
    print(f"• Extreme-return plot saved → {EXTREME_RETURNS_PNG}")

def main() -> None:
    close = download_prices()
    returns = compute_returns(close)
    full = pd.concat([close, returns], axis=1).dropna()
    full.to_csv(OUTPUT_CSV)
    print(f"• Cleaned data saved → {OUTPUT_CSV}")
    descriptive_stats(returns)

    gap_summary(full.index)

    overview_plot(close, returns)

    cond_spx, cond_dax, thr_spx, thr_dax = detect_extremes(returns)
    events = returns.loc[cond_spx | cond_dax].copy()
    events["Extreme_SPX"] = cond_spx.loc[events.index]
    events["Extreme_DAX"] = cond_dax.loc[events.index]
    events.to_csv(EXTREME_EVENTS_CSV)
    print(f"• Extreme-event table saved → {EXTREME_EVENTS_CSV}")
    print(
        f"  > Extreme counts | SPX: {cond_spx.sum()}  DAX: {cond_dax.sum()} "
        f"(mode = {THRESHOLD_MODE})"
    )

    extreme_plot(returns, cond_spx, cond_dax, thr_spx, thr_dax)

    print("\nProcess completed.")

if __name__ == "__main__":
    main()

