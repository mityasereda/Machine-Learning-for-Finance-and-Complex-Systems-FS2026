"""
Beta grid search for Robust RL — Elliptic (p1N2) and Ball (p1) uncertainty sets.
Run from the momentum/ directory: python gridsearch/beta_gridsearch.py

Trains both model types for each beta in BETA_GRID, backtests on all assets,
and selects the best beta per model type using a composite score:
  0.7 * mean_sharpe + 0.3 * min_sharpe  across assets (no market impact).
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

MODEL_TYPES = {
    "p1N2": {
        "label":        "Robust RL (Elliptic, p1N2)",
        "params_base":  {"robust_type": "p1N2", "epsilon": 1e-3, "u_dim": 3, **FIXED_FOCI},
    },
    "p1": {
        "label":        "Robust RL (Ball, p1)",
        "params_base":  {"robust_type": "p1",   "epsilon": 1e-3, "u_dim": 3},
    },
}

TRAIN_FROM     = "2021-05-09"
TRAIN_UNTIL    = "2022-05-09"
BACKTEST_FROM  = "2022-06-09"
BACKTEST_UNTIL = "2022-12-09"

_MOMENTUM_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRIDSEARCH_DIR       = os.path.join(_MOMENTUM_DIR, "gridsearch")
MODELS_DIR           = os.path.join(GRIDSEARCH_DIR, "models")
BACKTEST_DIR         = os.path.join(GRIDSEARCH_DIR, "backtest_results")
RESULTS_DIR          = os.path.join(GRIDSEARCH_DIR, "results")
MOMENTUM_RESULTS_DIR = os.path.join(_MOMENTUM_DIR, "results")

BETA_COLORS  = {1e-4: "#2196F3", 1e-3: "#FB8C00", 1e-2: "#E53935"}
MODEL_COLORS = {"p1N2": "#6A1B9A", "p1": "#00695C"}   # used in summary only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pkl_path(model_type, ticker, beta, impact_mode):
    return f"{BACKTEST_DIR}/{ticker}_{model_type}_beta{beta}_{impact_mode}.pkl"


def cum_return_from_aum(series):
    return (series / series.iloc[0] - 1) * 100


def get_rl_dates(ticker, all_results):
    """Align RL steps to momentum dates using length offset."""
    mom = pd.read_csv(
        f"{MOMENTUM_RESULTS_DIR}/{ticker}_momentum_stats_no_impact.csv",
        index_col=0, parse_dates=True
    )
    dates = mom.index
    for mtype in MODEL_TYPES:
        for beta in BETA_GRID:
            data = all_results.get(mtype, {}).get(beta, {}).get(ticker, {}).get("no_impact")
            if data is not None:
                rl_len = len(data["cumulative_returns"])
                return dates[len(dates) - rl_len:], mom
    return dates, mom


# ---------------------------------------------------------------------------
# Training + backtesting
# ---------------------------------------------------------------------------

def run_gridsearch(config):
    for model_type, minfo in MODEL_TYPES.items():
        for beta in BETA_GRID:
            robust_params = {**minfo["params_base"], "beta": beta}
            print(f"\n{'='*70}")
            print(f"  Model: {minfo['label']}  |  Beta = {beta}")
            print(f"{'='*70}")

            for ticker in ASSETS:
                # Skip if all pkls already exist
                if all(
                    os.path.exists(pkl_path(model_type, ticker, beta, im))
                    for im in ["no_impact", "with_impact"]
                ):
                    print(f"  [{ticker}] pkls exist, skipping.")
                    continue

                print(f"  [{ticker}] Training...")
                df_intra, df_daily = get_data(ticker, TRAIN_FROM, TRAIN_UNTIL)
                mpath = train(config, df_intra, df_daily, ticker,
                              robust_params=robust_params,
                              model_dir=MODELS_DIR)

                print(f"  [{ticker}] Backtesting...")
                df_intra_bt, df_daily_bt = get_data(ticker, BACKTEST_FROM, BACKTEST_UNTIL)

                for impact_mode, flag in [("no_impact", False), ("with_impact", True)]:
                    out = pkl_path(model_type, ticker, beta, impact_mode)
                    results = backtest_rl(config, df_intra_bt, df_daily_bt, mpath,
                                         consider_market_impact=flag)
                    with open(out, "wb") as f:
                        pickle.dump(results, f)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_results():
    """Returns all_results[model_type][beta][ticker][impact_mode] = data dict."""
    all_results = {mt: {} for mt in MODEL_TYPES}
    for model_type in MODEL_TYPES:
        for beta in BETA_GRID:
            all_results[model_type][beta] = {}
            for ticker in ASSETS:
                all_results[model_type][beta][ticker] = {}
                for impact_mode in ["no_impact", "with_impact"]:
                    path = pkl_path(model_type, ticker, beta, impact_mode)
                    if not os.path.exists(path):
                        continue
                    with open(path, "rb") as f:
                        all_results[model_type][beta][ticker][impact_mode] = pickle.load(f)
    return all_results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_all_stats(all_results):
    """Returns stats[model_type][beta][ticker][impact_mode] = (total_ret, sharpe, max_dd, ann_vol)."""
    stats = {mt: {} for mt in MODEL_TYPES}
    for model_type in MODEL_TYPES:
        for beta in BETA_GRID:
            stats[model_type][beta] = {}
            for ticker in ASSETS:
                stats[model_type][beta][ticker] = {}
                for impact_mode in ["no_impact", "with_impact"]:
                    data = all_results[model_type][beta].get(ticker, {}).get(impact_mode)
                    if data is None:
                        continue
                    pv  = data["portfolio_values"]
                    ret = np.diff(pv) / pv[:-1]
                    stats[model_type][beta][ticker][impact_mode] = compute_stats(pv, ret)
    return stats


def select_best_beta(stats, model_type):
    """Composite score = 0.7 * mean_sharpe + 0.3 * min_sharpe (no impact)."""
    scores = {}
    for beta in BETA_GRID:
        sharpes = [
            stats[model_type][beta][t]["no_impact"][1]
            for t in ASSETS
            if "no_impact" in stats[model_type][beta].get(t, {})
        ]
        scores[beta] = (0.7 * np.mean(sharpes) + 0.3 * np.min(sharpes)) if sharpes else float("-inf")
    best = max(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def write_stats_csv(stats, best_betas, scores_by_type):
    rows = []
    for model_type, minfo in MODEL_TYPES.items():
        for beta in BETA_GRID:
            for ticker in ASSETS:
                for impact_mode in ["no_impact", "with_impact"]:
                    s = stats[model_type][beta].get(ticker, {}).get(impact_mode)
                    if s is None:
                        continue
                    total_ret, sharpe, max_dd, ann_vol = s
                    rows.append({
                        "model_type":    minfo["label"],
                        "beta":          beta,
                        "ticker":        ticker,
                        "market_impact": "with" if impact_mode == "with_impact" else "no",
                        "total_return":  round(total_ret, 4),
                        "sharpe_ratio":  round(sharpe, 4),
                        "max_drawdown":  round(max_dd, 4),
                        "ann_volatility":round(ann_vol, 4),
                    })

    csv_path = f"{RESULTS_DIR}/beta_gridsearch_stats.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Stats -> {csv_path}")

    # Summary txt for best beta selection
    lines = []
    for model_type, minfo in MODEL_TYPES.items():
        best_beta = best_betas[model_type]
        scores    = scores_by_type[model_type]
        lines.append(f"{'='*72}")
        lines.append(f"  {minfo['label']} — Best Beta Selection")
        lines.append(f"  Metric: 0.7 * mean_sharpe + 0.3 * min_sharpe (no market impact)")
        lines.append(f"{'='*72}")
        lines.append(f"  {'Beta':<10}  {'Mean Sharpe':>12}  {'Min Sharpe':>11}  {'Composite':>10}")
        lines.append(f"  {'-'*50}")
        for beta in BETA_GRID:
            sharpes = [
                stats[model_type][beta][t]["no_impact"][1]
                for t in ASSETS
                if "no_impact" in stats[model_type][beta].get(t, {})
            ]
            mean_s = np.mean(sharpes) if sharpes else float("nan")
            min_s  = np.min(sharpes)  if sharpes else float("nan")
            marker = "  <-- BEST" if beta == best_beta else ""
            lines.append(
                f"  {beta:<10}  {mean_s:>12.3f}  {min_s:>11.3f}  {scores[beta]:>10.3f}{marker}"
            )
        lines.append(f"\n  Best beta: {best_beta}  (composite score: {scores[best_beta]:.3f})\n")

    txt_path = f"{RESULTS_DIR}/best_beta_selection.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Stats -> {txt_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_asset(all_results, stats):
    """One figure per asset, two subplots (p1N2 top, p1 bottom)."""
    for ticker in ASSETS:
        rl_dates, mom_df = get_rl_dates(ticker, all_results)
        spy_cum = cum_return_from_aum(mom_df["AUM_SPX"].dropna())
        spy_cum = spy_cum.loc[rl_dates[0]:]
        spy_cum = spy_cum - spy_cum.iloc[0]

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        fig.suptitle(f"{ticker}  —  Beta Grid Search", fontsize=13, fontweight="bold")

        for ax, (model_type, minfo) in zip(axes, MODEL_TYPES.items()):
            ax.plot(spy_cum.index, spy_cum.values,
                    color="#9E9E9E", linewidth=1.2, linestyle="-", alpha=0.6, zorder=1)

            for beta in BETA_GRID:
                color = BETA_COLORS[beta]
                for impact_mode in ["no_impact", "with_impact"]:
                    data = all_results[model_type][beta].get(ticker, {}).get(impact_mode)
                    if data is None:
                        continue
                    ls = "--" if impact_mode == "no_impact" else "-"
                    ax.plot(rl_dates, data["cumulative_returns"] * 100,
                            color=color, linewidth=1.8, linestyle=ls, zorder=2)

            ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
            ax.set_ylabel("Cumulative Return (%)", fontsize=10)
            ax.set_title(minfo["label"], fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.4)

            beta_handles = [
                Line2D([0], [0], color=BETA_COLORS[b], linewidth=2.5, label=f"β = {b}")
                for b in BETA_GRID
            ] + [Line2D([0], [0], color="#9E9E9E", linewidth=1.5, label="S&P 500")]
            style_handles = [
                Line2D([0], [0], color="black", linewidth=2, linestyle="-",  label="With MI"),
                Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="No MI"),
            ]
            leg1 = ax.legend(handles=beta_handles, loc="upper left", fontsize=8,
                             framealpha=0.9, title="Beta", title_fontsize=8)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc="lower left", fontsize=8,
                      framealpha=0.9, title="Market Impact", title_fontsize=8)

        axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
        axes[-1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        out = f"{RESULTS_DIR}/{ticker}_beta_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot  -> {out}")


def plot_summary(stats, best_betas):
    """Two-panel bar chart: one panel per model type, Sharpe ratio per beta per asset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Beta Grid Search Summary — Sharpe Ratio (no market impact)",
                 fontsize=12, fontweight="bold")

    n_assets  = len(ASSETS)
    n_betas   = len(BETA_GRID)
    bar_width = 0.22
    x = np.arange(n_assets)

    for ax, (model_type, minfo) in zip(axes, MODEL_TYPES.items()):
        best_beta = best_betas[model_type]
        for i, beta in enumerate(BETA_GRID):
            sharpes = []
            for ticker in ASSETS:
                s = stats[model_type][beta].get(ticker, {}).get("no_impact")
                sharpes.append(s[1] if s is not None else 0.0)
            offset = (i - n_betas / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, sharpes, bar_width,
                          label=f"β = {beta}", color=BETA_COLORS[beta], alpha=0.85)
            for bar, val in zip(bars, sharpes):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.02 if val >= 0 else -0.1),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ASSETS, fontsize=11)
        ax.set_ylabel("Sharpe Ratio", fontsize=10)
        ax.set_title(f"{minfo['label']}\nBest β = {best_beta}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
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

    best_betas     = {}
    scores_by_type = {}
    for model_type in MODEL_TYPES:
        best_beta, scores = select_best_beta(stats, model_type)
        best_betas[model_type]     = best_beta
        scores_by_type[model_type] = scores
        print(f"  Best beta ({model_type}): {best_beta}  (composite score: {scores[best_beta]:.3f})")

    print("\nGenerating plots...")
    plot_per_asset(all_results, stats)
    plot_summary(stats, best_betas)

    write_stats_csv(stats, best_betas, scores_by_type)
    print("\nDone.")


if __name__ == "__main__":
    main()
