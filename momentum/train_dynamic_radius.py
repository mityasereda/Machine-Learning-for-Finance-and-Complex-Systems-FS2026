import numpy as np
import pandas as pd
import yaml
from data import get_data
from rl_environment import TradingEnvironment
from rl_model import PPOTrainer
import torch
from datetime import datetime
import os
from tqdm import tqdm
import wandb
import json
from backtest_rl import final_backtest_rl
from seed_utils import set_seed

DEFAULT_COVERAGE_Q = 0.9
DEFAULT_CALIBRATION_FRACTION = 0.2

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data(ticker, from_date, until_date):
    # Parameters
    # # check if data is already in csv
    # if os.path.exists('data/intraday_data.csv') and os.path.exists('data/daily_data.csv'):
    #     df_intra = pd.read_csv('data/intraday_data.csv')
    #     df_daily = pd.read_csv('data/daily_data.csv')
    # else:
    df_intra, df_daily = get_data(ticker, from_date, until_date)  
    # save data to csv
    df_intra.to_csv('data/intraday_data.csv', index=False)
    df_daily.to_csv('data/daily_data.csv', index=False)
    return df_intra, df_daily

def split_train_calibration(df_intra, df_daily, calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
                            lookback_period=30):
    days = np.array(sorted(df_intra['day'].unique()))
    split_idx = int(len(days) * (1 - calibration_fraction))
    split_idx = min(max(split_idx, 1), len(days) - 1)
    train_days = set(days[:split_idx])
    calibration_days = set(days[split_idx:])

    train_intra = df_intra[df_intra['day'].isin(train_days)].copy()
    calibration_intra = df_intra[df_intra['day'].isin(calibration_days)].copy()

    daily = df_daily.copy()
    daily['day'] = pd.to_datetime(daily['caldt']).dt.date
    train_daily = daily[daily['day'].isin(train_days)].copy()
    calibration_daily = daily[daily['day'].isin(calibration_days)].copy()

    # Context buffer: last lookback_period trading days of the training split,
    # prepended to the calibration data so the environment warm-up uses real history.
    ctx_days = set(days[max(0, split_idx - lookback_period):split_idx])
    ctx_intra = df_intra[df_intra['day'].isin(ctx_days)].copy()
    ctx_daily = daily[daily['day'].isin(ctx_days)].copy()

    return train_intra, train_daily, calibration_intra, calibration_daily, ctx_intra, ctx_daily

def elliptic_l1_norm(residual, side, robust_params):
    residual_vector = np.array([-residual, 0.0, residual])
    if side == 'buy':
        focus_1 = np.array(robust_params["focus_buy"])
        focus_2 = np.array(robust_params["focus_buy_2"])
    else:
        focus_1 = np.array(robust_params["focus_sell"])
        focus_2 = np.array(robust_params["focus_sell_2"])
    return np.sum(np.abs(residual_vector - focus_1)) + np.sum(np.abs(residual_vector - focus_2))

def calibrate_beta(config, df_intra, df_daily, ticker, nominal_model_path, robust_params,
                   coverage_q=DEFAULT_COVERAGE_Q, ctx_intra=None, ctx_daily=None):
    if not 0 < coverage_q < 1:
        raise ValueError("coverage_q must be in (0, 1)")

    granularity = config['backtesting'].get('granularity', 'day')

    if ctx_intra is not None and not ctx_intra.empty:
        env_intra = pd.concat([ctx_intra, df_intra], ignore_index=True)
        env_daily = pd.concat([ctx_daily, df_daily], ignore_index=True)
    else:
        env_intra, env_daily = df_intra, df_daily

    nominal_env = TradingEnvironment(
        env_intra, env_daily, config, consider_market_impact=False,
        ticker=ticker, robust_params=None, granularity=granularity,
    )
    realised_env = TradingEnvironment(
        env_intra, env_daily, config, consider_market_impact=True,
        ticker=ticker, robust_params=None, granularity=granularity,
    )
    trainer = PPOTrainer(
        state_dim=nominal_env.observation_space.shape[0],
        action_dim=nominal_env.action_space.shape[0],
        hidden_dim=config['rl']['hidden_dim']
    )
    trainer.load(nominal_model_path)

    state = nominal_env.reset()
    realised_env.reset()
    residual_norms = []
    done = False
    while not done:
        action = trainer.select_action(state)
        next_state, _, done, nominal_info = nominal_env.step(action)
        _, _, realised_done, realised_info = realised_env.step(action)
        if realised_info['price_impact'] != 0:
            residual = (realised_info['effective_price'] - nominal_info['effective_price']) / nominal_info['effective_price']
            side = 'buy' if realised_info['price_impact'] > 0 else 'sell'
            residual_norms.append(elliptic_l1_norm(residual, side, robust_params))
        state = next_state
        done = done or realised_done

    if not residual_norms:
        return 0.0
    return float(np.quantile(residual_norms, coverage_q))

def train(config, df_intra, df_daily, ticker, robust_params=None):
    # Initialize wandb if enabled
    if config['wandb']['enabled']:
        # Create a meaningful run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}" if robust_params else "no_robust"
        run_name = f"{ticker}_PPO_{robust_str}_lr{config['rl']['learning_rate']}_gamma{config['rl']['gamma']}_ep{config['rl']['num_episodes']}_{timestamp}"
        
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            config={
                **config['rl'],
                'seed': config.get('seed', 42),
                'robust_params': robust_params,
                'ticker': ticker
            }
        )
    
    granularity = config['backtesting'].get('granularity', 'day')

    # Create environment
    env = TradingEnvironment(
        df_intra,
        df_daily,
        config,
        consider_market_impact=False,
        ticker=ticker,
        robust_params=None,
        granularity=granularity,
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['rl']['hidden_dim'],
        lr=config['rl']['learning_rate'],
        gamma=config['rl']['gamma'],
        epsilon=config['rl']['epsilon'],
        epochs=config['rl']['epochs'],
        batch_size=config['rl']['batch_size'],
        robust_params=robust_params
    )
    
    # Training parameters
    num_episodes = config['rl']['num_episodes']
    max_steps = config['rl']['max_steps']
    save_interval = config['rl']['save_interval']
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    os.makedirs('robust_models', exist_ok=True)
    
    # Training loop
    best_reward = float('-inf')
    
    # Create progress bar for episodes
    episode_pbar = tqdm(range(num_episodes), desc="Training Progress", unit="episode")
    
    for episode in episode_pbar:
        state = env.reset()
        episode_reward = 0
        
        # Lists to store episode data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        old_log_probs = []
        
        # Create progress bar for steps within episode
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode + 1}", leave=False)
        
        for _ in step_pbar:
            # Select action
            action, log_prob = trainer.sample_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_log_probs.append(log_prob)
            
            state = next_state
            episode_reward += reward
            
            # Update step progress bar
            step_pbar.set_postfix({'reward': f'{episode_reward:.2f}'})
            
            if done:
                break
        
        # Close step progress bar
        step_pbar.close()
        
        # Update policy
        loss = trainer.update(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(old_log_probs)
        )
        
        # Log metrics to wandb if enabled
        if config['wandb']['enabled']:
            wandb.log({
                'episode': episode,
                'episode_reward': episode_reward,
                'loss': loss,
                'best_reward': best_reward,
                'ticker': ticker
            })
        
        # Update episode progress bar
        episode_pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'loss': f'{loss:.4f}',
            'best_reward': f'{best_reward:.2f}'
        })
        
        # Generate model name with robust params
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}" if robust_params else "no_robust"
        
        # Save model if it's the best so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_name = f"{ticker}_best_model_{robust_str}.pth"
            trainer.save(f"robust_models/{model_name}") 
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_name = f"{ticker}_model_episode_{episode + 1}_{robust_str}.pth"
            trainer.save(f"robust_models/{model_name}") 
    
    # Close episode progress bar
    episode_pbar.close()
    model_path = f"robust_models/{ticker}_best_model_{robust_str}.pth"
    # Finish wandb run
    if config['wandb']['enabled']:
        wandb.finish()
    return model_path

def main():
    # Load configuration
    config = load_config()
    set_seed(config.get('seed', 42))
    robust_params = {
        "robust_type": "p1N2",
        "beta": None,
        "epsilon": 1e-3,
        "u_dim": 3,
        "focus_buy":   [-1.5e-5, 0,  1.5e-5],
        "focus_buy_2": [-4.5e-5, 0,  4.5e-5],
        "focus_sell":  [ 1.5e-5, 0, -1.5e-5],
        "focus_sell_2":[ 4.5e-5, 0, -4.5e-5]

    }

    assets = [
        "META", "MSFT", 'SPY'
    ]

    for ticker in assets:
        print(f"\nTraining model for {ticker}")
        from_date = '2021-05-09'
        until_date = '2022-05-09' 
        df_intra, df_daily = load_data(ticker, from_date, until_date) 
        train_intra, train_daily, calibration_intra, calibration_daily, ctx_intra, ctx_daily = split_train_calibration(df_intra, df_daily)
        nominal_model_path = train(config, train_intra, train_daily, ticker, robust_params=None)
        ticker_robust_params = robust_params.copy()
        ticker_robust_params["beta"] = calibrate_beta(
            config, calibration_intra, calibration_daily, ticker, nominal_model_path,
            ticker_robust_params, coverage_q=config.get('dynamic_radius', {}).get('coverage_q', DEFAULT_COVERAGE_Q),
            ctx_intra=ctx_intra, ctx_daily=ctx_daily,
        )
        print(f"Calibrated beta for {ticker}: {ticker_robust_params['beta']}")
        model_path = train(config, train_intra, train_daily, ticker, robust_params=ticker_robust_params)
        comparison_results = final_backtest_rl(ticker, model_path)
        print(comparison_results)

if __name__ == "__main__":
    main() 
