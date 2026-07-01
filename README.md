# Reproducing Robust Reinforcement Learning for Trading Strategies

This repository contains the source code and pre-trained models to reproduce the experiments presented in our paper. The code implements various reinforcement learning approaches for trading strategies, including standard RL, robust RL, and ball-constrained robust RL methods.

The main directory for all experiments is `momentum/`, which covers single-asset trading on META, MSFT, and SPY. Multi-asset trading is out of scope for this project.

---

## Environment Setup

```bash
cd momentum
pip install -r requirements.txt
```

To use your own data, you will need to extract it through the `extract_wrds_data.py` script, which requires a WRDS subscription.

The script pulls:
- TAQ daily trades tables into raw per-symbol daily files.
- CRSP daily prices/dividends into processed daily files.
- 1-minute intraday bars with the engineered columns expected by the repo.

It keeps the file layout aligned with the WRDS loader in `momentum/data.py`.

---

## Momentum — Robust RL Trading Pipeline

All commands are run from the `momentum/` directory.

### Step 1 — Train all RL models

```bash
python train.py
```

Trains four model variants for each asset (META, MSFT, SPY) and automatically runs the RL backtests upon completion:

| Variant | Uncertainty Set | Output Directory |
|---|---|---|
| Vanilla PPO | None | `models/` |
| Robust PPO (Elliptic, p1N2) | Elliptic, N=2, p=1 | `robust_models/` |
| Robust PPO (Ball, p1) | Ball, p=1 | `ball_models/` |
| Robust PPO (Dynamic Radius) | Elliptic, N=2, p=1 | `dynamic_radius_models/` |

Backtest results are saved as `.pkl` files in `backtest_rl_results/`.

> To train individual variants only, use `train_robust_rl.py` (elliptic), `train_ball_rl.py` (ball) or `train_dynamic_radius.py` (dynamic radius) instead.

---

### Step 2 — Run momentum benchmark

```bash
python backtest_momentum.py
```

Backtests the classical momentum strategy (breakout + volatility-targeted position sizing) with and without market impact. Saves results to `results/{ticker}_momentum_stats_{no,with}_impact.csv`.

---

### Step 3 — Calibration diagnostics

```bash
python calibration_diagnostics.py
```

Rolls out the nominal policy on the 20% calibration holdout and collects elliptic L1 norms of the market-impact residuals. Outputs:

| Output | Description |
|---|---|
| `calibration_results/calibration_summary.csv` | Per-asset summary statistics and calibrated β at each coverage quantile |
| `calibration_results/{ticker}_residual_norm_histogram.png` | Distribution of residual norms over the calibration fold |
| `calibration_results/{ticker}_residual_norm_timeseries.png` | Time evolution of residual norms over the calibration fold |

---

### Step 4 — Present results

**Full strategy comparison (plots + CSV statistics):**
```bash
python comparison_presentation.py
```

Outputs per-asset comparison plots and a unified statistics CSV:

| Output | Description |
|---|---|
| `results/{ticker}_strategy_comparison.png` | Cumulative return plot for all strategies with market impact |
| `results/strategy_statistics.csv` | Total return, Sharpe ratio, max drawdown, annualised volatility for every strategy |

**Side-by-side comparison (META and MSFT):**
```bash
python comparison_combined.py
```

Produces `results/META_MSFT_combined_comparison.png` — a single figure with both assets and a shared legend, suitable for inclusion in the paper.

**AUM scaling experiment:**
```bash
python run_aum_scaling.py
```

Re-evaluates Vanilla PPO and Dynamic Radius across AUM levels from \$100K to \$500M without retraining, and plots Sharpe and return degradation (no-MI minus with-MI) as a function of AUM. Outputs saved to `aum_scaling_results/`.

---

## Pre-trained Models

Pre-trained models are provided for immediate evaluation:

- Vanilla PPO: `momentum/models/{ticker}_best_model_no_robust.pth`
- Robust PPO (Elliptic): `momentum/robust_models/{ticker}_best_model_robust_p1N2_beta0.0001.pth`
- Robust PPO (Ball): `momentum/ball_models/{ticker}_best_model_robust_p1_beta0.0001.pth`
- Robust PPO (Dynamic Radius): `momentum/dynamic_radius_models/{ticker}_best_model_robust_dynamic_radius.pth`

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dimension | 256 |
| Learning rate | 3e-4 |
| Discount factor γ | 0.99 |
| PPO clip ε | 0.2 |
| PPO epochs | 10 |
| Batch size | 64 |
| Number of episodes | 100 |
| Training period | 2021-05-09 to 2022-05-09 |
| Backtest period | 2022-06-09 to 2022-12-09 |

**Robust PPO (p1N2) — uncertainty set parameters:**

| Parameter | Value |
|---|---|
| Beta (uncertainty size) | 1e-4 |
| Epsilon (price discretisation step) | 1e-3 |
| u dimension | 3 |
| Focus buy (u1, u2) | [-1.5e-5, 0, 1.5e-5], [-4.5e-5, 0, 4.5e-5] |
| Focus sell (u1, u2) | [1.5e-5, 0, -1.5e-5], [4.5e-5, 0, -4.5e-5] |

Foci satisfy Theorem 3.5(b): `‖u1 − u2‖₁ = 6×10⁻⁵ < β = 10⁻⁴`.
