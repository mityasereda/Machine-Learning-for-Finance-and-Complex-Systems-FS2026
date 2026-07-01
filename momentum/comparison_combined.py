"""
Combined side-by-side plot for META and MSFT (with market impact).
Single shared legend below the panels — larger and more readable in PDF.

Run from the momentum/ directory:
    python3 comparison_combined.py
"""

import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

RESULTS_DIR    = "results"
RL_RESULTS_DIR = "backtest_rl_results"
OUTPUT_DIR     = "results"
DAILY_DIR      = "../data/wrds/processed/daily"

ASSETS = ["META", "MSFT"]

STRATEGIES = {
    "no_robust":             {"label": "RL (Vanilla PPO)",           "color": "#2196F3"},
    "robust_p1N2":           {"label": "Robust RL (Elliptic, p1N2)", "color": "#E53935"},
    "robust_dynamic_radius": {"label": "Robust RL (Dynamic Radius)", "color": "#F3D321"},
    "robust_p1":             {"label": "Robust RL (Ball, p1)",       "color": "#43A047"},
    "momentum":              {"label": "Momentum (Classical)",        "color": "#FB8C00"},
    "bnh":                   {"label": "Buy & Hold",                  "color": "#7B1FA2"},
}


def load_momentum(ticker):
    no_impact   = pd.read_csv(f"{RESULTS_DIR}/{ticker}_momentum_stats_no_impact.csv",   index_col=0, parse_dates=True)
    with_impact = pd.read_csv(f"{RESULTS_DIR}/{ticker}_momentum_stats_with_impact.csv", index_col=0, parse_dates=True)
    return no_impact, with_impact


def load_rl_results(ticker):
    results = {}
    pattern = re.compile(rf"^{ticker}_best_model_(.*?)_(no_impact|with_impact)\.pkl$")
    for fname in sorted(os.listdir(RL_RESULTS_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        model_str, impact_mode = m.group(1), m.group(2)
        if   "robust_dynamic_radius" in model_str: stype = "robust_dynamic_radius"
        elif "robust_p1N2" in model_str:           stype = "robust_p1N2"
        elif "robust_p1"   in model_str:           stype = "robust_p1"
        elif "no_robust"   in model_str:           stype = "no_robust"
        else: continue
        with open(f"{RL_RESULTS_DIR}/{fname}", "rb") as f:
            results[(stype, impact_mode)] = pickle.load(f)
    return results


def load_buyandhold(ticker, start_date, initial_aum=100_000_000.0):
    df = pd.read_parquet(f"{DAILY_DIR}/{ticker}_daily.parquet")
    df.index = pd.to_datetime(df["caldt"])
    prices = df["close"].sort_index().loc[start_date:].dropna()
    return initial_aum * prices / prices.iloc[0]


def cum_return_from_aum(series):
    return (series / series.iloc[0] - 1) * 100


def plot_panel(ax, ticker, mom_wi, rl_results, rl_dates):
    bnh = load_buyandhold(ticker, rl_dates[0])
    bnh_cum = cum_return_from_aum(bnh) - cum_return_from_aum(bnh).iloc[0]

    ax.plot(bnh_cum.index, bnh_cum.values,
            color=STRATEGIES["bnh"]["color"], linewidth=1.5, alpha=0.85, zorder=1)

    cum = cum_return_from_aum(mom_wi["AUM"])
    cum = cum.loc[rl_dates[0]:] - cum.loc[rl_dates[0]:].iloc[0]
    ax.plot(cum.index, cum.values, color=STRATEGIES["momentum"]["color"],
            linewidth=1.8, zorder=2)

    for stype in ["no_robust", "robust_p1N2", "robust_dynamic_radius", "robust_p1"]:
        key = (stype, "with_impact")
        if key not in rl_results:
            continue
        ax.plot(rl_dates, rl_results[key]["cumulative_returns"] * 100,
                color=STRATEGIES[stype]["color"], linewidth=1.8, zorder=3)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_title(ticker, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, ticker in zip(axes, ASSETS):
        _, mom_wi   = load_momentum(ticker)
        rl_results  = load_rl_results(ticker)

        stored_dates = next(
            (v["dates"] for v in rl_results.values() if v.get("dates") is not None), None
        )
        if stored_dates is not None:
            rl_dates = stored_dates
        else:
            dates  = mom_wi.index
            rl_len = next((len(v["cumulative_returns"]) for v in rl_results.values()), None)
            offset = (len(dates) - rl_len) if rl_len else 0
            rl_dates = dates[offset:]

        plot_panel(ax, ticker, mom_wi, rl_results, rl_dates)

    axes[0].set_ylabel("Cumulative Return (%)", fontsize=11)

    legend_handles = [
        Line2D([0], [0], color=STRATEGIES[k]["color"], linewidth=2.5, label=STRATEGIES[k]["label"])
        for k in ["momentum", "no_robust", "robust_p1N2", "robust_dynamic_radius", "robust_p1", "bnh"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle("Strategy Comparison — with Market Impact", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/META_MSFT_combined_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
