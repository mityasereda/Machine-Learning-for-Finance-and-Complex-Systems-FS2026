# Reproducing Robust Reinforcement Learning for Trading Strategies

This repository contains the source code and pre-trained models to reproduce the experiments presented in our paper. The code implements various reinforcement learning approaches for trading strategies, including standard RL, robust RL, and ball-constrained robust RL methods.

## Repository Structure

- `momentum/` - Implementation of the reinforcement learning trading agent and the baseline momentum strategy
- `lobster/` - Order book data and market impact analysis tools
- `portfolio-rebalancing/` - Implementation of portfolio rebalancing using RL techniques

## Environment Setup

1. We have provided the environment setup in the `momentum` folder. To install the necessary dependencies:

```bash
cd momentum
pip install -r requirements.txt
```

2. Configure your API keys in the `momentum/config.yaml` file:
   - Replace `<YOUR_API_KEY>` with your Polygon.io API key
   - Replace `<YOUR_WANDB_ENTITY>` with your Weights & Biases username/entity if you want to track experiments

## Training Models

The project offers three types of reinforcement learning models:

### 1. Standard RL Model

To train the standard PPO reinforcement learning model:

```bash
cd momentum
python train_rl.py
```

### 2. Robust RL Model

To train a robust reinforcement learning model that handles uncertainty and perturbations:

```bash
cd momentum
python train_robust_rl.py
```

This model uses elliptic uncertainty sets with N=2 and p=1.

### 3. Ball-Constrained Robust RL Model

To train a ball-constrained robust reinforcement learning model:

```bash
cd momentum
python train_ball_rl.py
```

This model uses ball-constrained uncertainty sets.

## Pre-trained Models

We provide pre-trained models for immediate evaluation:

- Standard RL models: `momentum/models/SPY_best_model_no_robust.pth`, `momentum/models/MSFT_best_model_no_robust.pth`, and `momentum/models/META_best_model_no_robust.pth`
- Robust RL models: Available in the `momentum/robust_models/` directory
- Ball-constrained RL models: Available in the `momentum/ball_models/` directory

## Backtesting

To evaluate and compare all the trained models:

```bash
cd momentum
python compare_all.py
```

This will generate comparison results in the `momentum/models/comparison_results/` directory.

For individual backtesting, you can use:

```bash
cd momentum
python backtest_rl.py --model_path models/SPY_best_model_no_robust.pth --ticker SPY
```

## Hyperparameters

The key hyperparameters for different training methods are documented in `momentum/README.md` and can be configured in `momentum/config.yaml`.

| Parameter               | Standard RL | Robust RL (Elliptic) | Robust RL (Ball) |
| ----------------------- | ----------- | -------------------- | ---------------- |
| Robust Type             | None        | P1N2                 | P1               |
| Beta                    | N/A         | 0.0001               | 0.0001           |
| Hidden Dimension        | 256         | 256                  | 256              |
| Learning Rate           | 0.0003      | 0.0003               | 0.0003           |
| Gamma (Discount Factor) | 0.99        | 0.99                 | 0.99             |
| PPO Epsilon             | 0.2         | 0.2                  | 0.2              |

## Market Impact Analysis

For analyzing market impact using order book data:

```bash
cd lobster
python vis_impact_v3.py
```

## Additional Resources

- The `lobster/` directory contains LOBSTER sample files for order book analysis
- For portfolio optimization experiments, see the `portfolio-rebalancing/` directory
