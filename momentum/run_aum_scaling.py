"""
AUM scaling experiment — no retraining needed.

For each AUM level, re-evaluates vanilla PPO and dynamic radius (q90) with and
without market impact, then plots the return degradation (no_impact - with_impact)
as a function of AUM to replicate the spirit of the AUM-scaling analysis in
Ma & Huang (2025).

Run from the momentum/ directory:
    python3 run_aum_scaling.py
"""

import os
import re
import shutil
import sys
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt

ASSETS      = ["META", "MSFT", "SPY"]
AUM_LEVELS  = [100_000, 500_000, 1_000_000, 10_000_000, 100_000_000, 500_000_000]
AUM_LABELS  = ["$100K", "$500K", "$1M", "$10M", "$100M", "$500M"]
CONFIG_PATH = "config.yaml"
SOURCE_DIR  = "sweep_results_no_context_buffer/q90"
OUTPUT_DIR  = "aum_scaling_results"

MODELS = {
    "Vanilla PPO":       ("models",                "no_robust"),
    "Dynamic Radius":    ("dynamic_radius_models", "robust_dynamic_radius"),
}


def patch_aum(aum: float) -> None:
    with open(CONFIG_PATH) as f:
        text = f.read()
    text = re.sub(r"(initial_aum\s*:\s*)[\d_,\.]+", rf"\g<1>{aum:.1f}", text)
    with open(CONFIG_PATH, "w") as f:
        f.write(text)


def restore_models() -> None:
    for model_dir, _ in MODELS.values():
        src = os.path.join(SOURCE_DIR, model_dir)
        if not os.path.isdir(src):
            print(f"  WARNING: {src} not found")
            continue
        os.makedirs(model_dir, exist_ok=True)
        for fname in os.listdir(src):
            if fname.endswith(".pth"):
                shutil.copy2(os.path.join(src, fname), os.path.join(model_dir, fname))


def run_backtests() -> None:
    script = """
import sys
sys.path.insert(0, '.')
from backtest_rl import final_backtest_rl
import os

assets = ["META", "MSFT", "SPY"]
models = {
    "models":                "{ticker}_best_model_no_robust.pth",
    "dynamic_radius_models": "{ticker}_best_model_robust_dynamic_radius.pth",
}
os.makedirs("backtest_rl_results", exist_ok=True)
for ticker in assets:
    for model_dir, tmpl in models.items():
        path = os.path.join(model_dir, tmpl.format(ticker=ticker))
        if not os.path.exists(path):
            print(f"  Skipping {path}")
            continue
        print(f"  {ticker} / {model_dir}")
        final_backtest_rl(ticker, path)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def compute_stats(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pv  = np.array(data["portfolio_values"])
    ret = (pv[-1] / pv[0] - 1) * 100
    daily = np.diff(pv) / pv[:-1]
    sharpe = np.sqrt(252) * np.mean(daily) / (np.std(daily) + 1e-10)
    return ret, sharpe


def load_metrics() -> dict:
    """Returns {(label, ticker, impact): (return_pct, sharpe)}"""
    results = {}
    pkl_dir = "backtest_rl_results"
    strategy_map = {
        "no_robust":             "Vanilla PPO",
        "robust_dynamic_radius": "Dynamic Radius",
    }
    for fname in os.listdir(pkl_dir):
        if not fname.endswith(".pkl"):
            continue
        for key, label in strategy_map.items():
            for impact in ("no_impact", "with_impact"):
                if f"best_model_{key}_{impact}" in fname:
                    ticker = fname.split("_")[0]
                    results[(label, ticker, impact)] = compute_stats(
                        os.path.join(pkl_dir, fname)
                    )
    return results


def make_plot(data, ylabel, title, outpath):
    colors = {"Vanilla PPO": "#2196F3", "Dynamic Radius": "#F3D321"}
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(14, 5), sharey=False)
    for ax, ticker in zip(axes, ASSETS):
        for label, color in colors.items():
            ax.plot(AUM_LEVELS, data[label][ticker], marker="o", linewidth=2,
                    color=color, label=label)
        ax.set_xscale("log")
        ax.set_xticks(AUM_LEVELS)
        ax.set_xticklabels(AUM_LABELS, rotation=30, ha="right")
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xlabel("AUM")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    restore_models()

    ret_deg   = {label: {t: [] for t in ASSETS} for label in MODELS}
    sharpe_deg = {label: {t: [] for t in ASSETS} for label in MODELS}

    for aum, aum_label in zip(AUM_LEVELS, AUM_LABELS):
        print(f"\n=== AUM = {aum_label} ===")
        patch_aum(aum)
        run_backtests()
        m = load_metrics()
        for label in MODELS:
            for ticker in ASSETS:
                r_no,  s_no  = m.get((label, ticker, "no_impact"),   (np.nan, np.nan))
                r_wi,  s_wi  = m.get((label, ticker, "with_impact"), (np.nan, np.nan))
                ret_deg[label][ticker].append(r_no - r_wi)
                sharpe_deg[label][ticker].append(s_no - s_wi)

    patch_aum(100_000_000)

    make_plot(ret_deg,
              ylabel="Return degradation (no MI − with MI, %)",
              title="Return Degradation vs AUM — Vanilla PPO vs Dynamic Radius",
              outpath=os.path.join(OUTPUT_DIR, "aum_scaling_return.png"))

    make_plot(sharpe_deg,
              ylabel="Sharpe degradation (no MI − with MI)",
              title="Sharpe Degradation vs AUM — Vanilla PPO vs Dynamic Radius",
              outpath=os.path.join(OUTPUT_DIR, "aum_scaling_sharpe.png"))


if __name__ == "__main__":
    main()
