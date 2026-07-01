# Reproducing Robust Reinforcement Learning for Trading Strategies

This repository contains the source code and pre-trained models to reproduce the experiments presented in our paper. The code implements various reinforcement learning approaches for trading strategies, including standard RL, robust RL, and ball-constrained robust RL methods.

## Repository Structure

- `momentum/` — Robust RL trading agent and classical momentum benchmark (single-asset)
- `portfolio-rebalancing/` — Portfolio rebalancing using RL techniques

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

It keeps the file layout aligned with the optional WRDS loader introduced in `momentum/data.py` and `portfolio-rebalancing/data.py`.

---

## Momentum — Robust RL Trading Pipeline

All commands are run from the `momentum/` directory.

### Step 1 — Train all RL models

```bash
python train.py
```

Trains four model variants for each asset (META, MSFT, SPY):

| Variant | Uncertainty Set | Output Directory |
|---|---|---|
| Vanilla PPO | None | `models/` |
| Robust PPO (Elliptic, p1N2) | Elliptic, N=2, p=1 | `robust_models/` |
| Robust PPO (Ball, p1) | Ball, p=1 | `ball_models/` |
| Robust PPO (Dynamic Radius) | Elliptic, N=2, p=1 | `dynamic_radius_models/` |

After each training run, backtesting is performed automatically over the period **2022-06-09 to 2022-12-09**. Results are saved as `.pkl` files in `backtest_rl_results/`.

> To train individual variants only, use `train_robust_rl.py` (elliptic), `train_ball_rl.py` (ball) or `train_dynamic_radius.py` (dynamic radius) instead.

---

### Step 2 — Run momentum benchmark

```bash
python backtest_momentum.py
```

Backtests the classical momentum strategy (breakout + volatility-targeted position sizing) with and without market impact. Saves results to `results/{ticker}_momentum_stats_{no,with}_impact.csv`.

---

### Step 3 — Present results

**Full strategy comparison (plots + CSV statistics):**
```bash
python comparison_presentation.py
```

Outputs per-asset comparison plots and a unified statistics CSV:

| Output | Description |
|---|---|
| `results/{ticker}_strategy_comparison.png` | Cumulative return plot: all strategies, with/without market impact |
| `results/strategy_statistics.csv` | Total return, Sharpe ratio, max drawdown, annualised volatility for every strategy × impact mode |

**Quick RL-only summary table:**
```bash
python backtest_rl_results/read_results.py
```

Aggregates all RL backtest pkl files into a wide-format CSV at `backtest_rl_results/rl_model_comparison.csv`.

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
