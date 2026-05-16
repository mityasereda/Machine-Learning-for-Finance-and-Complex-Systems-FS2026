"""
Beta grid search for Robust RL (p1N2, elliptic uncertainty set).
Run from the momentum/ directory: python gridsearch/beta_gridsearch.py

Trains the p1N2 model for each beta in BETA_GRID with fixed foci,
backtests on all assets, and selects the best beta using a composite
score: 0.7 * mean_sharpe + 0.3 * min_sharpe across assets (no market impact).
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Run from momentum/ so all existing imports resolve normally
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_data
from train import train
from backtest_rl import backtest_rl
from comparison_presentation import compute_stats
from seed_utils import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_GRID = [1e-4, 1e-3, 1e-2]
ASSETS    = ["META", "MSFT", "SPY"]

FIXED_FOCI = {
    "focus_buy":   [-1.5e-5, 0,  1.5e-5],
    "focus_buy_2": [-4.5e-5, 0,  4.5e-5],
    "focus_sell":  [ 1.5e-5, 0, -1.5e-5],
    "focus_sell_2":[ 4.5e-5, 0, -4.5e-5],
}

TRAIN_FROM     = "2021-05-09"
TRAIN_UNTIL    = "2022-05-09"
BACKTEST_FROM  = "2022-06-09"
BACKTEST_UNTIL = "2022-12-09"

GRIDSEARCH_DIR = "gridsearch"
MODELS_DIR     = f"{GRIDSEARCH_DIR}/models"
BACKTEST_DIR   = f"{GRIDSEARCH_DIR}/backtest_results"
RESULTS_DIR    = f"{GRIDSEARCH_DIR}/results"
MOMENTUM_RESULTS_DIR = "results"

BETA_COLORS = {1e-4: "#2196F3", 1e-3: "#FB8C00", 1e-2: "#E53935"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def pkl_path(ticker, beta, impact_mode):
    return f"{BACKTEST_DIR}/{ticker}_beta{beta}_{impact_mode}.pkl"


def model_path(ticker, beta):
    return f"{MODELS_DIR}/{ticker}_best_model_robust_p1N2_beta{beta}.pth"


def cum_return_from_aum(series):
    return (series / series.iloc[0] - 1) * 100


# ---------------------------------------------------------------------------
# Training + backtesting
# ---------------------------------------------------------------------------

def run_gridsearch(config):
    for beta in BETA_GRID:
        robust_params = {
            "robust_type": "p1N2",
            "beta": beta,
            "epsilon": 1e-3,
            "u_dim": 3,
            **FIXED_FOCI,
        }
        print(f"\n{'='*70}")
        print(f"  Beta = {beta}")
        print(f"{'='*70}")

        for ticker in ASSETS:
            print(f"\n  [{ticker}] Training...")
            df_intra, df_daily = get_data(ticker, TRAIN_FROM, TRAIN_UNTIL)
            mpath = train(config, df_intra, df_daily, ticker,
                          robust_params=robust_params,
                          model_dir=MODELS_DIR)

            print(f"  [{ticker}] Backtesting...")
            df_intra_bt, df_daily_bt = get_data(ticker, BACKTEST_FROM, BACKTEST_UNTIL)

            for impact_mode, flag in [("no_impact", False), ("with_impact", True)]:
                out = pkl_path(ticker, beta, impact_mode)
                if os.path.exists(out):
                    print(f"  [{ticker}] {impact_mode} pkl exists, skipping backtest.")
                    continue
                results = backtest_rl(config, df_intra_bt, df_daily_bt, mpath,
                                      consider_market_impact=flag)
                with open(out, "wb") as f:
                    pickle.dump(results, f)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_results():
    """Returns all_results[beta][ticker][impact_mode] = data dict."""
    all_results = {}
    for beta in BETA_GRID:
        all_results[beta] = {}
        for ticker in ASSETS:
            all_results[beta][ticker] = {}
            for impact_mode in ["no_impact", "with_impact"]:
                path = pkl_path(ticker, beta, impact_mode)
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    all_results[beta][ticker][impact_mode] = pickle.load(f)
    return all_results


def get_rl_dates(ticker, all_results):
    """Align RL steps to momentum dates using length offset."""
    mom = pd.read_csv(f"{MOMENTUM_RESULTS_DIR}/{ticker}_momentum_stats_no_impact.csv",
                      index_col=0, parse_dates=True)
    dates  = mom.index
    # find first available result length
    for beta in BETA_GRID:
        if "no_impact" in all_results[beta].get(ticker, {}):
            rl_len = len(all_results[beta][ticker]["no_impact"]["cumulative_returns"])
            offset = len(dates) - rl_len
            return dates[offset:], mom
    return dates, mom


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_all_stats(all_results):
    """
    Returns stats[beta][ticker][impact_mode] = (total_ret, sharpe, max_dd, ann_vol).
    Uses compute_stats() from comparison_presentation.py (pv pct-change based).
    """
    stats = {}
    for beta in BETA_GRID:
        stats[beta] = {}
        for ticker in ASSETS:
            stats[beta][ticker] = {}
            for impact_mode in ["no_impact", "with_impact"]:
                data = all_results[beta].get(ticker, {}).get(impact_mode)
                if data is None:
                    continue
                pv  = data["portfolio_values"]
                ret = np.diff(pv) / pv[:-1]
                stats[beta][ticker][impact_mode] = compute_stats(pv, ret)
    return stats


def select_best_beta(stats):
    """
    Composite score = 0.7 * mean_sharpe + 0.3 * min_sharpe across assets (no impact).
    Returns (best_beta, scores_dict).
    """
    scores = {}
    for beta in BETA_GRID:
        sharpes = [
            stats[beta][t]["no_impact"][1]
            for t in ASSETS
            if "no_impact" in stats[beta].get(t, {})
        ]
        if not sharpes:
            scores[beta] = float("-inf")
            continue
        scores[beta] = 0.7 * np.mean(sharpes) + 0.3 * np.min(sharpes)
    best = max(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_stats_txt(stats, best_beta, scores):
    lines = []

    for ticker in ASSETS:
        lines.append(f"{'='*72}")
        lines.append(f"  {ticker} — Beta Grid Search Statistics (Robust RL p1N2)")
        lines.append(f"{'='*72}")
        header = f"  {'Beta':<10} {'MI':>6}  {'Total Ret':>10}  {'Sharpe':>8}  {'Max DD':>9}  {'Ann Vol':>8}"
        lines.append(header)
        lines.append(f"  {'-'*66}")

        for beta in BETA_GRID:
            for impact_mode in ["no_impact", "with_impact"]:
                s = stats[beta].get(ticker, {}).get(impact_mode)
                if s is None:
                    continue
                total_ret, sharpe, max_dd, ann_vol = s
                mi_label = "with" if impact_mode == "with_impact" else "no"
                lines.append(
                    f"  {beta:<10}  {mi_label:>5}  {total_ret:>+9.2f}%"
                    f"  {sharpe:>8.3f}  {max_dd:>8.2f}%  {ann_vol:>7.2f}%"
                )
            lines.append("")
        lines.append("")

    # Summary
    lines.append(f"{'='*72}")
    lines.append("  BEST BETA SELECTION")
    lines.append(f"  Metric: 0.7 * mean_sharpe + 0.3 * min_sharpe (no market impact)")
    lines.append(f"{'='*72}")
    lines.append(f"  {'Beta':<10}  {'Mean Sharpe':>12}  {'Min Sharpe':>11}  {'Composite':>10}")
    lines.append(f"  {'-'*50}")
    for beta in BETA_GRID:
        sharpes = [
            stats[beta][t]["no_impact"][1]
            for t in ASSETS
            if "no_impact" in stats[beta].get(t, {})
        ]
        mean_s = np.mean(sharpes) if sharpes else float("nan")
        min_s  = np.min(sharpes)  if sharpes else float("nan")
        marker = "  <-- BEST" if beta == best_beta else ""
        lines.append(f"  {beta:<10}  {mean_s:>12.3f}  {min_s:>11.3f}  {scores[beta]:>10.3f}{marker}")

    lines.append(f"\n  Best beta: {best_beta}  (composite score: {scores[best_beta]:.3f})")
    lines.append(f"{'='*72}\n")

    out_path = f"{RESULTS_DIR}/beta_gridsearch_stats.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Stats -> {out_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_asset(all_results, stats):
    for ticker in ASSETS:
        rl_dates, mom_df = get_rl_dates(ticker, all_results)

        fig, ax = plt.subplots(figsize=(13, 6))

        # SPY benchmark
        spy_cum = cum_return_from_aum(mom_df["AUM_SPX"].dropna())
        spy_cum = spy_cum.loc[rl_dates[0]:]
        spy_cum = spy_cum - spy_cum.iloc[0]
        ax.plot(spy_cum.index, spy_cum.values,
                color="#9E9E9E", linewidth=1.2, linestyle="-", alpha=0.6, zorder=1,
                label="S&P 500")

        # One line per beta × impact mode
        for beta in BETA_GRID:
            color = BETA_COLORS[beta]
            for impact_mode in ["no_impact", "with_impact"]:
                data = all_results[beta].get(ticker, {}).get(impact_mode)
                if data is None:
                    continue
                ls = "--" if impact_mode == "no_impact" else "-"
                ax.plot(rl_dates, data["cumulative_returns"] * 100,
                        color=color, linewidth=1.8, linestyle=ls, zorder=2)

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.set_title(f"{ticker}  —  Robust RL (p1N2) Beta Comparison", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)

        beta_handles = [
            Line2D([0], [0], color=BETA_COLORS[b], linewidth=2.5, label=f"β = {b}")
            for b in BETA_GRID
        ] + [Line2D([0], [0], color="#9E9E9E", linewidth=1.5, label="S&P 500")]
        style_handles = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-",  label="With market impact"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="No market impact"),
        ]
        leg1 = ax.legend(handles=beta_handles, loc="upper left", fontsize=9,
                         framealpha=0.9, title="Beta", title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, loc="lower left", fontsize=9,
                  framealpha=0.9, title="Market Impact", title_fontsize=9)

        plt.tight_layout()
        out = f"{RESULTS_DIR}/{ticker}_beta_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot  -> {out}")


def plot_summary(stats, best_beta):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_assets  = len(ASSETS)
    n_betas   = len(BETA_GRID)
    bar_width = 0.22
    x = np.arange(n_assets)

    for i, beta in enumerate(BETA_GRID):
        sharpes = []
        for ticker in ASSETS:
            s = stats[beta].get(ticker, {}).get("no_impact")
            sharpes.append(s[1] if s is not None else 0.0)
        offset = (i - n_betas / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, sharpes, bar_width,
                      label=f"β = {beta}", color=BETA_COLORS[beta], alpha=0.85)
        for bar, val in zip(bars, sharpes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if val >= 0 else -0.08),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS, fontsize=11)
    ax.set_ylabel("Sharpe Ratio (no market impact)", fontsize=11)
    ax.set_title(f"Beta Grid Search Summary — Robust RL (p1N2)\n"
                 f"Best overall beta: {best_beta}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    out = f"{RESULTS_DIR}/beta_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot  -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    set_seed(config.get("seed", 42))

    for d in [MODELS_DIR, BACKTEST_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    print("Running beta grid search...")
    run_gridsearch(config)

    print("\nLoading results and computing statistics...")
    all_results = load_all_results()
    stats       = compute_all_stats(all_results)
    best_beta, scores = select_best_beta(stats)

    print(f"\nBest beta: {best_beta}  (composite score: {scores[best_beta]:.3f})")

    print("\nGenerating plots...")
    plot_per_asset(all_results, stats)
    plot_summary(stats, best_beta)

    write_stats_txt(stats, best_beta, scores)
    print("\nDone.")


if __name__ == "__main__":
    main()
