import pickle
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = RESULTS_DIR / "rl_model_comparison.csv"


def parse_filename(path: Path) -> tuple[str, str, str]:
    stem = path.stem

    if stem.endswith("_no_impact"):
        impact = "no_impact"
        base = stem[: -len("_no_impact")]
    elif stem.endswith("_with_impact"):
        impact = "with_impact"
        base = stem[: -len("_with_impact")]
    else:
        raise ValueError(f"Unexpected result file name: {path.name}")

    if "_best_model_" not in base:
        return None
    ticker, model = base.split("_best_model_", 1)
    return ticker, normalize_model_name(model), impact


def normalize_model_name(model: str) -> str:
    import re
    if model == "no_robust":
        return "vanilla_ppo"
    m = re.match(r"robust_(p1N2|p1)_beta(.+)", model)
    if m:
        robust_type, beta = m.group(1), m.group(2)
        label = "elliptic" if robust_type == "p1N2" else "ball"
        return f"robust_{label}_beta{beta}"
    return model


def load_metrics(path: Path) -> dict:
    import numpy as np
    with path.open("rb") as handle:
        result = pickle.load(handle)

    pv  = result["portfolio_values"]
    ret = np.diff(pv) / pv[:-1]
    sharpe = float(np.sqrt(252) * np.mean(ret) / (np.std(ret) + 1e-10))

    peak   = np.maximum.accumulate(pv)
    max_dd = float(np.min((pv - peak) / peak) * 100.0)

    return {
        "return_pct": float(result["cumulative_returns"][-1] * 100.0),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
    }


def build_comparison_table() -> pd.DataFrame:
    rows = []

    for path in sorted(RESULTS_DIR.glob("*.pkl")):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        ticker, model, impact = parsed
        metrics = load_metrics(path)
        metrics.update(
            {
                "ticker": ticker,
                "model": model,
                "impact": impact,
            }
        )
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    wide = df.pivot(index=["ticker", "model"], columns="impact")
    wide.columns = [f"{metric}_{impact}" for metric, impact in wide.columns]
    wide = wide.reset_index()

    def _model_order(name):
        if name == "vanilla_ppo":          return 0
        if name.startswith("robust_elliptic"): return 1
        if name.startswith("robust_ball"):     return 2
        return 99

    wide["model_order"] = wide["model"].map(_model_order)
    wide = wide.sort_values(["ticker", "model_order"]).drop(columns=["model_order"])

    ordered_columns = [
        "ticker",
        "model",
        "return_pct_no_impact",
        "return_pct_with_impact",
        "sharpe_ratio_no_impact",
        "sharpe_ratio_with_impact",
        "max_drawdown_pct_no_impact",
        "max_drawdown_pct_with_impact",
    ]
    return wide[ordered_columns]


def main():
    comparison = build_comparison_table()

    if comparison.empty:
        print(f"No result pickle files found in {RESULTS_DIR}")
        return

    comparison.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for ticker, group in comparison.groupby("ticker", sort=False):
        print(f"\n{ticker}")
        print(group.to_string(index=False))

    print(f"\nSaved comparison table to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
