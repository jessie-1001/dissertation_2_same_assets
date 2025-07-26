# =============================================================================
# SCRIPT 05: ADVANCED VISUALISATION DASHBOARD  (ASCII‑clean version)
#
# Generates four diagnostic figures that complement SCRIPT‑04 back‑testing:
#   1. Rolling VaR hit‑rate curves (99 % & 95 %).
#   2. Heat‑map of back‑test metrics.
#   3. PIT‑Uniform QQ plot.
#   4. Volatility & weight paths with crisis shading.
#
# Required CSVs (produced by earlier scripts):
#   - spx_dax_daily_data.csv
#   - garch_copula_main_results.csv
#   - garch_copula_robustness.csv
#   - copula_input_data_full.csv
#   - backtesting_summary_final.csv
#
# Author : <your‑name>
# =============================================================================
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import norm

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk", font_scale=0.9)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def align_realized(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Align +1‑day realized portfolio return with Student‑t weights."""
    w = forecast[["StudentT_Weight_SPX", "StudentT_Weight_DAX"]]
    shifted = actual.shift(-1)[["SPX_Return", "DAX_Return"]]
    return (shifted.loc[w.index] * w.values).sum(axis=1)


def rolling_pct(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean expressed in percent."""
    return series.rolling(window, min_periods=1).mean() * 100.0


def add_crisis_shading(ax: plt.Axes, windows: dict, **kwargs) -> None:
    """Shade crisis periods on a matplotlib axis."""
    for start, end in windows.values():
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   color="red", alpha=0.15, **kwargs)


def uniform_qq(data: np.ndarray, ax: plt.Axes) -> None:
    """Uniform QQ plot with 95 % Kolmogorov–Smirnov envelope."""
    n = len(data)
    emp = np.sort(data)
    theo = (np.arange(1, n + 1) - 0.5) / n
    ax.plot(theo, emp, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", c="k")
    # KS envelope
    eps = 1.36 / np.sqrt(n)
    ax.fill_between([0, 1], [0 + eps, 1 - eps], [0 - eps, 1 + eps],
                    color="gray", alpha=0.2)
    ax.set_xlabel("Theoretical U(0,1) Quantile")
    ax.set_ylabel("Empirical PIT Quantile")
    ax.set_title("PIT–Uniform QQ Plot")


# -----------------------------------------------------------------------------
# Figure 1 – rolling VaR hit‑rates
# -----------------------------------------------------------------------------
def plot_roll_hit(forecast: pd.DataFrame,
                  actual: pd.DataFrame,
                  window: int = 250) -> None:
    realized = align_realized(forecast, actual)
    var99 = forecast["StudentT_VaR_99"]
    # crude 95 % VaR proxy
    var95 = forecast["StudentT_VaR_99"] * 0.6

    breaches99 = (realized <= var99).astype(int)
    breaches95 = (realized <= var95).astype(int)

    roll99 = rolling_pct(breaches99, window)
    roll95 = rolling_pct(breaches95, window)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(roll99.index, roll99.values, label="Hit-Rate 99 %", lw=2)
    ax.step(roll95.index, roll95.values, label="Hit-Rate 95 %",
            lw=2, color="#E17C05")

    ax.axhline(1, ls="--", color="k", lw=1, label="Expected 1 %")
    ax.axhline(5, ls="--", color="grey", lw=1, label="Expected 5 %")
    ax.set_title(f"{window}-Day Rolling VaR Violation Rate")
    ax.set_ylabel("% of Breaches")
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig("viz_roll_hit_rate.png", dpi=300)


# -----------------------------------------------------------------------------
# Figure 2 – heat‑map of back‑test metrics
# -----------------------------------------------------------------------------
def plot_heatmap(tbl: pd.DataFrame) -> None:
    metrics = tbl.copy()
    metrics["|ES Z|"] = metrics["ES Z-score"].abs()

    hm = metrics.set_index("Model")[["Kupiec p-val",
                                     "Christoffersen p-val",
                                     "|ES Z|"]]

    # Normalise so larger = worse
    hm_norm = hm.copy()
    hm_norm["Kupiec p-val"] = 1 - hm_norm["Kupiec p-val"].clip(0, 1)
    hm_norm["Christoffersen p-val"] = 1 - hm_norm["Christoffersen p-val"].clip(0, 1)
    hm_norm["|ES Z|"] = hm_norm["|ES Z|"] / hm_norm["|ES Z|"].max()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(hm_norm,
                annot=hm.round(2),
                fmt=".2f",
                cmap="Reds",
                cbar_kws={"label": "Deeper Red = Worse"},
                linewidths=.5,
                ax=ax)
    ax.set_title("Back-test Metrics (Deeper Red = Worse)")
    plt.tight_layout()
    fig.savefig("viz_backtest_heatmap.png", dpi=300)


# -----------------------------------------------------------------------------
# Figure 3 – PIT QQ
# -----------------------------------------------------------------------------
def plot_qq(pit_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    uniform_qq(pit_df["u_spx"].values, ax)
    plt.tight_layout()
    fig.savefig("viz_pit_qq.png", dpi=300)


# -----------------------------------------------------------------------------
# Figure 4 – volatility & weights
# -----------------------------------------------------------------------------
def plot_vol_and_weights(forecast: pd.DataFrame) -> None:
    crisis = {
        "COVID-19": ("2020-02-19", "2020-04-07"),
        "Banking23": ("2023-03-08", "2023-03-24")
    }

    sig_spx = forecast["StudentT_Vol_SPX"]
    sig_dax = forecast["StudentT_Vol_DAX"]
    w_spx = forecast["StudentT_Weight_SPX"]
    w_dax = forecast["StudentT_Weight_DAX"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # top: vol
    ax1.plot(sig_spx.index, sig_spx, label=r"$\sigma_{\mathrm{SPX}}$", lw=1.3)
    ax1.plot(sig_dax.index, sig_dax, label=r"$\sigma_{\mathrm{DAX}}$", lw=1.3)
    ax1.set_ylabel(r"$\sigma$ (%)")
    ax1.set_title("One-Day-Ahead Volatility Forecasts")
    add_crisis_shading(ax1, crisis)

    # bottom: weights
    ax2.step(w_spx.index, w_spx, label="SPX weight", lw=1.2)
    ax2.step(w_dax.index, w_dax, label="DAX weight", lw=1.2, color="#E17C05")
    ax2.set_ylabel("Weight")
    ax2.set_title("Minimum-Variance Portfolio Weights")
    add_crisis_shading(ax2, crisis)
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig("viz_vol_and_weights.png", dpi=300)


# -----------------------------------------------------------------------------
# Master
# -----------------------------------------------------------------------------
def main() -> None:
    print("Creating refined visualisations …")

    # Load data
    actual = pd.read_csv("spx_dax_daily_data.csv",
                         index_col="Date", parse_dates=True)
    fc_main = pd.read_csv("garch_copula_main_results.csv",
                          index_col="Date", parse_dates=True)
    fc_rob = pd.read_csv("garch_copula_robustness.csv",
                         index_col="Date", parse_dates=True)
    forecast = pd.concat([fc_main, fc_rob], axis=1)
    pit_full = pd.read_csv("copula_input_data_full.csv",
                           index_col="Date", parse_dates=True)
    backtest_tbl = pd.read_csv("backtesting_summary_final.csv")

    # Generate figures
    plot_roll_hit(forecast, actual)
    plot_heatmap(backtest_tbl)
    plot_qq(pit_full)
    plot_vol_and_weights(forecast)

    print("Done – all PNGs saved.")


if __name__ == "__main__":
    main()
