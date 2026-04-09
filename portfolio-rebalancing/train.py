import numpy as np
import pandas as pd
import yaml
from data import get_data
from rl_environment import TradingEnvironment, load_data
from rl_trainer import PPOTrainer
import torch
from datetime import datetime
import os
from tqdm import tqdm
import wandb
import json 

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train(config, tickers, from_date, until_date, robust_params=None):
    if config['wandb']['enabled']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}" if robust_params else "no_robust"
        tickers_str = "_".join(tickers)
        run_name = f"{tickers_str}_PPO_{robust_str}_lr{config['rl']['learning_rate']}_gamma{config['rl']['gamma']}_ep{config['rl']['num_episodes']}_{timestamp}"
        
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=run_name,
            config={
                **config['rl'],
                'robust_params': robust_params,
                'tickers': tickers
            }
        )
    
    env = TradingEnvironment(
        config=config, 
        initial_cash=config['backtesting'].get('initial_cash', 100000),
        consider_market_impact=config['backtesting']['market_impact'].get('enabled', True),
        tickers=tickers,
        from_date=from_date,
        until_date=until_date,
        robust_params=robust_params
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    
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
    
    num_episodes = config['rl']['num_episodes']
    max_steps = config['rl']['max_steps']
    save_interval = config['rl']['save_interval']
    
    os.makedirs('models', exist_ok=True)
    
    best_reward = float('-inf')
    
    episode_pbar = tqdm(range(num_episodes), desc="Training Progress", unit="episode")
    
    for episode in episode_pbar:
        state = env.reset()
        episode_reward = 0
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode + 1}", leave=False)
        
        for step in step_pbar:
            action = trainer.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            step_pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'portfolio': f'{info["portfolio_value"]:.2f}'
            })
            
            if done:
                break
        
        step_pbar.close()
        
        loss = trainer.update(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        
        if config['wandb']['enabled']:
            log_data = {
                'episode': episode,
                'episode_reward': episode_reward,
                'loss': loss,
                'best_reward': best_reward,
                'portfolio_value': info["portfolio_value"],
                'cash': info["cash"]
            }
            
            for ticker, position in info['positions'].items():
                log_data[f'position_{ticker}'] = position
                
            wandb.log(log_data)
        
        episode_pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'loss': f'{loss:.4f}',
            'best_reward': f'{best_reward:.2f}',
            'portfolio': f'{info["portfolio_value"]:.2f}'
        })
        
        tickers_str = "_".join(tickers)
        robust_str = f"robust_{robust_params['robust_type']}_beta{robust_params['beta']}" if robust_params else "no_robust"
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_name = f"{tickers_str}_best_model_{robust_str}.pth"
            trainer.save(f"models/{model_name}") 
        
        if (episode + 1) % save_interval == 0:
            model_name = f"{tickers_str}_model_episode_{episode + 1}_{robust_str}.pth"
            trainer.save(f"models/{model_name}") 
    
    episode_pbar.close()
    model_path = f"models/{tickers_str}_best_model_{robust_str}.pth"
    
    if config['wandb']['enabled']:
        wandb.finish()
    
    return model_path

def main():
    config = load_config()
    
    robust_params = None
    
    tickers = ["SPY", "TLT", "GLD", "EFA", "VNQ"]
    from_date = "2021-05-09"
    until_date = "2022-05-09"
    
    model_path = train(config, tickers, from_date, until_date, robust_params)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 