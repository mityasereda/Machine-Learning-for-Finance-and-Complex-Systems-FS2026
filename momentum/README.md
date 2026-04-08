# Reproducing Instruction

This repository contains code for reproducing experiments in

## Install Dependencies

Run the following command to install the dependencies.

```bash
pip install -r requirements.txt
```

## Training RL Models

The project offers three types of RL models that can be trained:

Train standard PPO reinforcement learning models:

```bash
python train_rl.py
```

Train robust reinforcement learning models that handle uncertainty and perturbations:

```bash
python train_robust_rl.py
```

This model uses the ellipstic uncertainty sets with $N=2$ and $p=1$.

Train ball-constrained robust reinforcement learning models:

```bash
python train_ball_rl.py
```

This model uses the ball-constrained uncertainty sets.

## Backtesting Models

After completing training of all models, we provide the backtesting scrip

```bash
python compare_all.py
```

We also provide the pre-trained models and the becktesting results.

## Training Hyperparameters

The following table shows the current hyperparameter settings for each training script:

| Parameter               | train_rl.py | train_robust_rl.py | train_ball_rl.py |
| ----------------------- | ----------- | ------------------ | ---------------- |
| Robust Type             | None        | P1N2               | P1               |
| Beta                    | N/A         | 0.0001             | 0.0001           |
| P2 Coefficient          | N/A         | 1.0                | 1.0              |
| U Dimension             | N/A         | 3                  | 3                |
| Epsilon                 | N/A         | 0.001              | 0.001            |
| Hidden Dimension        | 256         | 256                | 256              |
| Learning Rate           | 0.0003      | 0.0003             | 0.0003           |
| Gamma (Discount Factor) | 0.99        | 0.99               | 0.99             |
| PPO Epsilon             | 0.2         | 0.2                | 0.2              |
| Training Epochs         | 10          | 10                 | 10               |
| Batch Size              | 64          | 64                 | 64               |
| Number of Episodes      | 15-50*      | 15                 | 15-50*           |

\* Number of episodes varies by ticker:

- SPY: 50 episodes
- MSFT, META: 15 episodes
