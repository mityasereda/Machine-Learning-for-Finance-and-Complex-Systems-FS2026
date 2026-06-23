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

ASSETS = ["META", "MSFT", "SPY"]

STRATEGIES = {
    "no_robust":   {"label": "RL (Vanilla PPO)",           "color": "#2196F3"},
    "robust_p1N2": {"label": "Robust RL (Elliptic, p1N2)", "color": "#E53935"},
    "robust_p1":   {"label": "Robust RL (Ball, p1)",       "color": "#43A047"},
    "momentum":    {"label": "Momentum (Classical)",        "color": "#FB8C00"},
    "bnh":         {"label": "Buy & Hold",                   "color": "#7B1FA2"},
}

TABLE_ROW_ORDER = [
    ("bnh",         "benchmark"),
    ("momentum",    "no_impact"),
    ("momentum",    "with_impact"),
    ("no_robust",   "no_impact"),
    ("no_robust",   "with_impact"),
    ("robust_p1N2", "no_impact"),
    ("robust_p1N2", "with_impact"),
    ("robust_p1",   "no_impact"),
    ("robust_p1",   "with_impact"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_momentum(ticker):
    no_impact   = pd.read_csv(f"{RESULTS_DIR}/{ticker}_momentum_stats_no_impact.csv",   index_col=0, parse_dates=True)
    with_impact = pd.read_csv(f"{RESULTS_DIR}/{ticker}_momentum_stats_with_impact.csv", index_col=0, parse_dates=True)
    return no_impact, with_impact


def load_rl_results(ticker):
    """Returns dict (strategy_type, impact_mode) -> full data dict."""
    results = {}
    pattern = re.compile(rf"^{ticker}_best_model_(.*?)_(no_impact|with_impact)\.pkl$")
    for fname in sorted(os.listdir(RL_RESULTS_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        model_str, impact_mode = m.group(1), m.group(2)
        if   "robust_p1N2" in model_str: stype = "robust_p1N2"
        elif "robust_p1"   in model_str: stype = "robust_p1"
        elif "no_robust"   in model_str: stype = "no_robust"
        else: continue
        with open(f"{RL_RESULTS_DIR}/{fname}", "rb") as f:
            results[(stype, impact_mode)] = pickle.load(f)
    return results


def load_buyandhold(ticker, start_date, initial_aum=10_000_000.0):
    df = pd.read_parquet(f"{DAILY_DIR}/{ticker}_daily.parquet")
    df.index = pd.to_datetime(df["caldt"])
    prices = df["close"].sort_index().loc[start_date:].dropna()
    return initial_aum * prices / prices.iloc[0]


def cum_return_from_aum(series):
    return (series / series.iloc[0] - 1) * 100


# ---------------------------------------------------------------------------
# Statistics  (always derived from actual portfolio value pct-changes)
# ---------------------------------------------------------------------------

def compute_stats(pv_array, daily_ret_array):
    """
    pv_array:        portfolio values (numpy array)
    daily_ret_array: actual daily pct returns (numpy array), used only for
                     Sharpe / vol. For RL we compute these from pv directly.
    Returns: (total_ret_pct, sharpe, max_dd_pct, ann_vol_pct)
    """
    total_ret = (pv_array[-1] / pv_array[0] - 1) * 100

    # Use actual pct-changes from portfolio values for risk metrics
    pct = np.diff(pv_array) / pv_array[:-1]
    sharpe  = np.sqrt(252) * np.mean(pct) / (np.std(pct) + 1e-10)
    ann_vol = np.std(pct) * np.sqrt(252) * 100

    peak   = np.maximum.accumulate(pv_array)
    max_dd = np.min((pv_array - peak) / peak) * 100

    return total_ret, sharpe, max_dd, ann_vol


def compute_all_stats(ticker, mom_no, mom_wi, rl_results, rl_dates):
    rows = {}

    # Stock buy-and-hold benchmark
    bnh = load_buyandhold(ticker, rl_dates[0]).values
    rows[("bnh", "benchmark")] = compute_stats(bnh, np.diff(bnh) / bnh[:-1])

    # Momentum — align to RL start date, reconstruct AUM array
    for impact_mode, df in [("no_impact", mom_no), ("with_impact", mom_wi)]:
        aum = df["AUM"].loc[rl_dates[0]:].values
        ret = df["ret"].loc[rl_dates[0]:].dropna().values
        rows[("momentum", impact_mode)] = compute_stats(aum, ret)

    # RL — compute pct-changes from portfolio_values (not daily_returns)
    for (stype, impact_mode), data in rl_results.items():
        pv  = data["portfolio_values"]
        ret = np.diff(pv) / pv[:-1]
        rows[(stype, impact_mode)] = compute_stats(pv, ret)

    return rows


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def collect_stats_rows(ticker, stats):
    rows = []
    for stype, impact_mode in TABLE_ROW_ORDER:
        key = (stype, impact_mode)
        if key not in stats:
            continue
        total_ret, sharpe, max_dd, ann_vol = stats[key]
        mi_label = {"with_impact": "with", "no_impact": "no", "benchmark": "—"}.get(impact_mode, "no")
        rows.append({
            "ticker":         ticker,
            "strategy":       STRATEGIES[stype]["label"],
            "market_impact":  mi_label,
            "total_return":   round(total_ret, 4),
            "sharpe_ratio":   round(sharpe, 4),
            "max_drawdown":   round(max_dd, 4),
            "ann_volatility": round(ann_vol, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_asset(ticker, mom_no, mom_wi, rl_results, rl_dates):
    fig, ax = plt.subplots(figsize=(13, 6))

    # Stock buy-and-hold benchmark
    bnh = load_buyandhold(ticker, rl_dates[0])
    bnh_cum = cum_return_from_aum(bnh) - cum_return_from_aum(bnh).iloc[0]
    ax.plot(bnh_cum.index, bnh_cum.values,
            color=STRATEGIES["bnh"]["color"], linewidth=1.2, linestyle="-", alpha=0.8, zorder=1)

    # Momentum
    for impact_mode, df in [("no_impact", mom_no), ("with_impact", mom_wi)]:
        cum = cum_return_from_aum(df["AUM"])
        cum = cum.loc[rl_dates[0]:]
        cum = cum - cum.iloc[0]
        ls  = "--" if impact_mode == "no_impact" else "-"
        ax.plot(cum.index, cum.values, color=STRATEGIES["momentum"]["color"],
                linewidth=1.8, linestyle=ls, zorder=2)

    # RL strategies
    for stype in ["no_robust", "robust_p1N2", "robust_p1"]:
        for impact_mode in ["no_impact", "with_impact"]:
            key = (stype, impact_mode)
            if key not in rl_results:
                continue
            ls = "--" if impact_mode == "no_impact" else "-"
            ax.plot(rl_dates, rl_results[key]["cumulative_returns"] * 100,
                    color=STRATEGIES[stype]["color"], linewidth=1.8, linestyle=ls, zorder=3)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title(f"{ticker}  —  Strategy Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)

    strategy_handles = [
        Line2D([0], [0], color=STRATEGIES[k]["color"], linewidth=2.5, label=STRATEGIES[k]["label"])
        for k in ["momentum", "no_robust", "robust_p1N2", "robust_p1"]
    ] + [Line2D([0], [0], color=STRATEGIES["bnh"]["color"], linewidth=2.5,
                label=f"{ticker} Buy & Hold")]
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-",  label="With market impact"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="No market impact"),
    ]
    leg1 = ax.legend(handles=strategy_handles, loc="upper left", fontsize=9,
                     framealpha=0.9, title="Strategy", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=9,
              framealpha=0.9, title="Market Impact", title_fontsize=9)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{ticker}_strategy_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []

    for ticker in ASSETS:
        print(f"Processing {ticker}...")
        mom_no, mom_wi = load_momentum(ticker)
        rl_results     = load_rl_results(ticker)

        # Use actual trading dates from RL backtest if available; otherwise approximate via index offset
        stored_dates = next(
            (v["dates"] for v in rl_results.values() if v.get("dates") is not None), None
        )
        if stored_dates is not None:
            rl_dates = stored_dates
        else:
            dates  = mom_no.index
            rl_len = next((len(v["cumulative_returns"]) for v in rl_results.values()), None)
            offset = (len(dates) - rl_len) if rl_len else 0
            rl_dates = dates[offset:]

        stats = compute_all_stats(ticker, mom_no, mom_wi, rl_results, rl_dates)
        all_rows.extend(collect_stats_rows(ticker, stats))

        path = plot_asset(ticker, mom_no, mom_wi, rl_results, rl_dates)
        print(f"  Plot  -> {path}")

    csv_path = f"{OUTPUT_DIR}/strategy_statistics.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"  Stats -> {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
