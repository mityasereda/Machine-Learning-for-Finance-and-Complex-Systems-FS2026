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

    ticker, model = base.split("_best_model_", 1)
    return ticker, normalize_model_name(model), impact


def normalize_model_name(model: str) -> str:
    mapping = {
        "no_robust": "non_robust",
        "robust_p1N2_beta0": "robust_p1N2",
        "robust_p1_beta0": "robust_p1",
    }
    return mapping.get(model, model)


def load_metrics(path: Path) -> dict:
    with path.open("rb") as handle:
        result = pickle.load(handle)

    return {
        "return_pct": float(result["cumulative_returns"][-1] * 100.0),
        "sharpe_ratio": float(result["sharpe_ratio"]),
        "max_drawdown_pct": float(result["max_drawdown"] * 100.0),
    }


def build_comparison_table() -> pd.DataFrame:
    rows = []

    for path in sorted(RESULTS_DIR.glob("*.pkl")):
        ticker, model, impact = parse_filename(path)
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

    model_order = {
        "non_robust": 0,
        "robust_p1N2": 1,
        "robust_p1": 2,
    }
    wide["model_order"] = wide["model"].map(model_order).fillna(99)
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
