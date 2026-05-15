# RL Architecture: PPO with Robust Extension

## Overview

The RL system trains a PPO agent to trade a single asset. Files:
- `rl_model.py` — `ActorCritic` network + `PPOTrainer` (PPO update, robust bonus)
- `rl_environment.py` — `TradingEnvironment` (Gymnasium env, state/reward/market impact)
- `train_rl.py` / `train_robust_rl.py` — training loops

---

## ActorCritic Network

```
Input (state_dim = 4 * lookback_period)
  └─ Shared Feature Extractor: Linear → ReLU (×3), hidden_dim=256
       ├─ Actor Head: Linear → ReLU → Linear → Tanh  → action_mean ∈ (-1, 1)
       └─ Critic Head: Linear → ReLU → Linear        → V(s) scalar
  └─ Learnable log_std parameter (per action dim)
```

Weight init: orthogonal (gain=√2), biases zero.

---

## State Space

Shape: `(4 * lookback_period,)` — concatenation of z-scored:
1. `prices_norm` — closing prices
2. `volumes_norm` — volume
3. `returns_norm` — price returns
4. `volatilities_norm` — rolling 20-step std

---

## Policy Distribution

`_build_policy(action_mean)` constructs a **Tanh-squashed Gaussian**:
1. `latent_mean = atanh(clamp(action_mean, ±(1-ε)))` — invert the actor's Tanh
2. `Normal(latent_mean, exp(log_std))` — Gaussian in latent space
3. Wrap with `TanhTransform` → samples in `(-1, 1)`

This is the same design as SAC. Two modes:
- `sample_action` — stochastic (exploration during rollout)
- `select_action` — deterministic (evaluation/backtest)

---

## PPO Update (`update()`)

1. **Robust bonus** (if enabled): compute `u*` per step → add `γ · vᵀu*` to rewards
2. **Returns**: MC backward pass bootstrapped from `V(s_last)` — intentional to preserve the full adversarial signal from the robust correction without bootstrapping bias
3. **Advantages**: `A = R - V(s)`, normalized (mean 0, std 1)
4. **K epochs of minibatch updates**:
   - Ratio `r = exp(log π(a|s) - log π_old(a|s))`
   - Clipped surrogate: `L_actor = -min(r·A, clip(r, 1±ε)·A).mean()`
   - Critic loss: `L_critic = MSE(V(s), R)`
   - Total: `L = L_actor + 0.5 · L_critic`
   - Gradient clip: 0.5
5. **LR scheduler**: `ReduceLROnPlateau` stepped on `-mean_loss`

---

## Robust PPO Extension (Theorem 3.5)

Computes a correction term `u*` that represents the worst-case price perturbation, then adds it as an extra reward signal.

### Discretized Value Vector

Evaluates critic at 3 perturbed next states (price ± ε and unchanged):
```
v = [V(s'_{p+ε}), V(s'_p), V(s'_{p-ε})]   shape: [batch, 3]
```

### `p1N2` — Elliptic Uncertainty (Theorem 3.5b)

Foci `u1` depend on action sign (buy/sell, from config). `u2 = 0`.
```
midpoint = (u1 + u2) / 2
scale    = (β - ||u1-u2||₁) / 2
μ*       = -(v_max + v_min) / 2
λ*       = -(v_max - v_min) / 4
u*       = midpoint - scale · sign(v + μ*) · 𝟙[|v+μ*| ≥ 2|λ*|]
```

### `p1` — Ball Uncertainty (Theorem 3.5a)

```
u* = β · sign(v_{j*}) · e_{j*}   where j* = argmax_j |v_j + μ*|
```

### How it enters training

```python
robust_bonus = γ · (v · u*)           # per step, shape [batch]
rewards_for_returns = rewards + robust_bonus
# returns computed from rewards_for_returns
```

The environment also stochastically perturbs `effective_price` using a distribution shifted by `u*` (loaded from `u_star_{robust_type}.pkl`).

---

## Training Loop

```
for episode in range(num_episodes):
    rollout:  sample_action → env.step → collect (s, a, r, s', done, log_prob)
    update:   PPOTrainer.update(full episode buffer)
    save:     best model by episode reward
```

---

## Known Problems

### 1. `np.load` vs `pickle.dump` mismatch
**Location**: `rl_model.py:232` saves with `pickle.dump`; `rl_environment.py:264` loads with `np.load`

`np.load` will fail on a non-array pickle object unless `allow_pickle=True` is passed. In practice `u*` is a tensor, not a numpy array, so this silently falls to the `except` fallback every time — meaning the environment never uses the computed `u*`.

**Fix**: Load with `pickle.load` in `rl_environment.py`, and save the tensor as `.numpy()` if using `np.load`.

---

### 2. Perturbing normalized price with raw-space ε
**Location**: `rl_model.py:108–113`

`price_idx = state_dim // 4 - 1` correctly identifies the last element of `prices_norm`. But the perturbation adds `±ε` (e.g., `1e-3`) to the **normalized** price, whereas the environment applies the same `ε` to the **raw** price. The scales differ by the normalization factor (~σ of price window), so the critic is being evaluated at inconsistent perturbation magnitudes.

**Fix**: Either perturb the raw price and re-normalize before feeding the critic, or document that ε is intentionally defined in normalized space and use consistent values.

