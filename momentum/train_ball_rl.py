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

def train(config, df_intra, df_daily, ticker, robust_params=None):
    # Initialize wandb if enabled
    if config['wandb']['enabled']:
        # Create a meaningful run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}_p2{robust_params['p2_coef']}" if robust_params else "no_robust"
        run_name = f"{ticker}_PPO_{robust_str}_lr{config['rl']['learning_rate']}_gamma{config['rl']['gamma']}_ep{config['rl']['num_episodes']}_{timestamp}"
        
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            config={
                **config['rl'],
                'robust_params': robust_params,
                'ticker': ticker
            }
        )
    
    # Create environment
    env = TradingEnvironment(df_intra, df_daily, config, robust_params=None)
    
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
        
        # Create progress bar for steps within episode
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode + 1}", leave=False)
        
        for step in step_pbar:
            # Select action
            action = trainer.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
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
            np.array(dones)
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
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}_p2{robust_params['p2_coef']}" if robust_params else "no_robust"
        
        # Save model if it's the best so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_name = f"{ticker}_best_model_{robust_str}.pth"
            trainer.save(f"ball_models/{model_name}") 
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_name = f"{ticker}_model_episode_{episode + 1}_{robust_str}.pth"
            trainer.save(f"ball_models/{model_name}") 
    
    # Close episode progress bar
    episode_pbar.close()
    model_path = f"ball_models/{ticker}_best_model_{robust_str}.pth"
    # Finish wandb run
    if config['wandb']['enabled']:
        wandb.finish()
    return model_path

def main():
    # Load configuration
    config = load_config()
    robust_params = {
        "robust_type": "p1",
        "beta": 0.0001,
        "p2_coef": 1.0,
        "u_dim": 3,
        "epsilon": 0.001
    }  
    assets = [
        "META", "MSFT",  'SPY' 
    ]

    for ticker in assets:
        print(f"\nTraining model for {ticker}")
        from_date = '2021-05-09'
        until_date = '2022-05-09'
        if ticker in ['MSFT', 'QQQ', 'SPY']:
            config['rl']['num_episodes'] = 50 
        elif ticker in ['AAPL', 'META', 'XOM']:
            config['rl']['num_episodes'] = 15  
        # Load data
        df_intra, df_daily = load_data(ticker, from_date, until_date)
        
        # Train model
        model_path = train(config, df_intra, df_daily, ticker, robust_params=robust_params)
        comparison_results = final_backtest_rl(ticker, model_path)
        print(comparison_results)

if __name__ == "__main__":
    main() 