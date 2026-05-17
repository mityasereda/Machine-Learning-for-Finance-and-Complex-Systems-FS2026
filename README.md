# Reproducing Robust Reinforcement Learning for Trading Strategies

This repository contains the source code and pre-trained models to reproduce the experiments presented in our paper. The code implements various reinforcement learning approaches for trading strategies, including standard RL, robust RL, and ball-constrained robust RL methods.

## Repository Structure

- `momentum/` — Robust RL trading agent and classical momentum benchmark (single-asset)
- `portfolio-rebalancing/` — Portfolio rebalancing using RL techniques
- `lobster/` — Order book data and market impact analysis tools

---

## Environment Setup

```bash
cd momentum
pip install -r requirements.txt
```

Configure API keys in `momentum/config.yaml`:
- Replace `<YOUR_API_KEY>` with your Polygon.io API key (only needed if using the Polygon data provider)
- Replace `<YOUR_WANDB_ENTITY>` with your Weights & Biases username if you want to track experiments

---

## Momentum — Robust RL Trading Pipeline

All commands are run from the `momentum/` directory.

### Step 1 — Train all RL models

```bash
python train.py
```

Trains three model variants for each asset (META, MSFT, SPY):

| Variant | Uncertainty Set | Output Directory |
|---|---|---|
| Vanilla PPO | None | `models/` |
| Robust PPO (Elliptic, p1N2) | Elliptic, N=2, p=1 | `robust_models/` |
| Robust PPO (Ball, p1) | Ball, p=1 | `ball_models/` |

After each training run, backtesting is performed automatically over the period **2022-06-09 to 2022-12-09**. Results are saved as `.pkl` files in `backtest_rl_results/`.

> To train individual variants only, use `train_robust_rl.py` (elliptic) or `train_ball_rl.py` (ball) instead.

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

### Optional — Beta grid search (Robust Elliptic only)

```bash
python gridsearch/beta_gridsearch.py
```

Trains the p1N2 model across `beta ∈ {1e-4, 1e-3, 1e-2}` with fixed foci and selects the best beta using a composite score (`0.7 × mean Sharpe + 0.3 × min Sharpe` across assets). Outputs plots and statistics to `gridsearch/results/`.

---

## Pre-trained Models

Pre-trained models are provided for immediate evaluation:

- Vanilla PPO: `momentum/models/{ticker}_best_model_no_robust.pth`
- Robust PPO (Elliptic): `momentum/robust_models/{ticker}_best_model_robust_p1N2_beta0.0001.pth`
- Robust PPO (Ball): `momentum/ball_models/{ticker}_best_model_robust_p1_beta0.0001.pth`

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

---

## Market Impact Analysis

For analysing market impact using order book data:

```bash
cd lobster
python vis_impact_v3.py
```

The `lobster/` directory contains LOBSTER sample files for order book analysis. For portfolio optimisation experiments, see the `portfolio-rebalancing/` directory.
