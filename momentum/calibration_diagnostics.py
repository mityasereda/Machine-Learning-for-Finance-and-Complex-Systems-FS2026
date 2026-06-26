"""
Calibration diagnostics: collect elliptic L1 norms of market-impact residuals
from the 20% calibration holdout and produce summary statistics and plots.

Run from the momentum/ directory:
    python3 calibration_diagnostics.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from data import get_data
from rl_environment import TradingEnvironment
from rl_model import PPOTrainer
from train_dynamic_radius import split_train_calibration, elliptic_l1_norm
from seed_utils import set_seed

ASSETS      = ["META", "MSFT", "SPY"]
OUTPUT_DIR  = "calibration_results"
ROBUST_PARAMS_BASE = {
    "robust_type": "p1N2",
    "beta": 1e-4,
    "epsilon": 1e-3,
    "u_dim": 3,
    "focus_buy":   [-1.5e-5, 0,  1.5e-5],
    "focus_buy_2": [-4.5e-5, 0,  4.5e-5],
    "focus_sell":  [ 1.5e-5, 0, -1.5e-5],
    "focus_sell_2":[ 4.5e-5, 0, -4.5e-5],
}
QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def collect_residual_norms(config, df_intra, df_daily, ticker, nominal_model_path,
                           ctx_intra=None, ctx_daily=None):
    """
    Run the nominal model over the calibration data in two parallel environments
    (with and without market impact) and return a DataFrame with one row per
    trade: date, side, residual (fractional), and elliptic L1 norm.
    """
    granularity = config["backtesting"].get("granularity", "day")

    if ctx_intra is not None and not ctx_intra.empty:
        env_intra = pd.concat([ctx_intra, df_intra], ignore_index=True)
        env_daily = pd.concat([ctx_daily, df_daily], ignore_index=True)
    else:
        env_intra, env_daily = df_intra, df_daily

    nominal_env = TradingEnvironment(
        env_intra, env_daily, config,
        initial_cash=config["backtesting"]["initial_aum"],
        consider_market_impact=False,
        ticker=ticker, robust_params=None, granularity=granularity,
    )
    realised_env = TradingEnvironment(
        env_intra, env_daily, config,
        initial_cash=config["backtesting"]["initial_aum"],
        consider_market_impact=True,
        ticker=ticker, robust_params=None, granularity=granularity,
    )

    trainer = PPOTrainer(
        state_dim=nominal_env.observation_space.shape[0],
        action_dim=nominal_env.action_space.shape[0],
        hidden_dim=config["rl"]["hidden_dim"],
    )
    trainer.load(nominal_model_path)

    state = nominal_env.reset()
    realised_env.reset()

    records = []
    done = False
    while not done:
        action = trainer.select_action(state)
        next_state, _, done, nominal_info   = nominal_env.step(action)
        _,          _, rdone, realised_info = realised_env.step(action)

        if realised_info["price_impact"] != 0:
            residual = (
                (realised_info["effective_price"] - nominal_info["effective_price"])
                / nominal_info["effective_price"]
            )
            side = "buy" if realised_info["price_impact"] > 0 else "sell"
            norm = elliptic_l1_norm(residual, side, ROBUST_PARAMS_BASE)
            records.append({
                "date":     nominal_info.get("date", None),
                "side":     side,
                "residual": residual,
                "norm":     norm,
            })

        state = next_state
        done  = done or rdone

    return pd.DataFrame(records)


def plot_histogram(df, ticker, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df["norm"], bins=30, color="#2196F3", edgecolor="white", alpha=0.85)

    colors = ["#FB8C00", "#E53935", "#7B1FA2", "#43A047", "#000000"]
    for q, c in zip(QUANTILES, colors):
        val = np.quantile(df["norm"], q)
        ax.axvline(val, color=c, linewidth=1.5, linestyle="--",
                   label=f"p{int(q*100)} = {val:.4f}")

    ax.set_xlabel("Elliptic L1 norm of market-impact residual", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{ticker} — Calibration residual norm distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_residual_norm_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_timeseries(df, ticker, output_dir):
    df_ts = df.dropna(subset=["date"]).copy()
    if df_ts.empty:
        return None

    df_ts["date"] = pd.to_datetime(df_ts["date"])
    df_ts = df_ts.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4))

    buys  = df_ts[df_ts["side"] == "buy"]
    sells = df_ts[df_ts["side"] == "sell"]
    ax.scatter(buys["date"],  buys["norm"],  color="#E53935", s=20, label="Buy",  alpha=0.7, zorder=3)
    ax.scatter(sells["date"], sells["norm"], color="#2196F3", s=20, label="Sell", alpha=0.7, zorder=3)

    for q, c in zip([0.90, 0.95], ["#FB8C00", "#7B1FA2"]):
        val = np.quantile(df_ts["norm"], q)
        ax.axhline(val, color=c, linewidth=1.2, linestyle="--", label=f"p{int(q*100)} = {val:.4f}")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Elliptic L1 norm", fontsize=11)
    ax.set_title(f"{ticker} — Residual norms over calibration period", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_residual_norm_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def compute_summary(df, ticker):
    norms = df["norm"].values
    row = {"ticker": ticker, "n_trades": len(norms)}
    row["mean"]   = np.mean(norms)
    row["std"]    = np.std(norms)
    row["min"]    = np.min(norms)
    row["median"] = np.median(norms)
    for q in QUANTILES:
        row[f"p{int(q*100)}"] = np.quantile(norms, q)
    row["max"] = np.max(norms)
    return row


def main():
    config = load_config()
    set_seed(config.get("seed", 42))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from_date  = "2021-05-09"
    until_date = "2022-05-09"

    summary_rows = []

    for ticker in ASSETS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        nominal_model_path = os.path.join("dynamic_radius_models", f"{ticker}_best_model_no_robust.pth")
        if not os.path.exists(nominal_model_path):
            print(f"  Nominal model not found at {nominal_model_path} — skipping.")
            print(f"  Run train.py first to generate the dynamic radius models.")
            continue

        print("  Loading data and splitting 80/20...")
        df_intra, df_daily = get_data(ticker, from_date, until_date)
        _, _, calibration_intra, calibration_daily, ctx_intra, ctx_daily = split_train_calibration(df_intra, df_daily)

        print("  Running calibration rollout...")
        df_norms = collect_residual_norms(
            config, calibration_intra, calibration_daily, ticker, nominal_model_path,
            ctx_intra=ctx_intra, ctx_daily=ctx_daily,
        )

        if df_norms.empty:
            print("  No trades with market impact during calibration — skipping.")
            continue

        print(f"  Collected {len(df_norms)} trade residuals.")

        # Summary stats
        row = compute_summary(df_norms, ticker)
        summary_rows.append(row)
        print(f"  Calibrated beta (p90) = {row['p90']:.6f}")

        # Plots
        p = plot_histogram(df_norms, ticker, OUTPUT_DIR)
        print(f"  Histogram  -> {p}")
        p = plot_timeseries(df_norms, ticker, OUTPUT_DIR)
        if p:
            print(f"  Time series -> {p}")

    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(OUTPUT_DIR, "calibration_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary stats -> {csv_path}")
        print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
