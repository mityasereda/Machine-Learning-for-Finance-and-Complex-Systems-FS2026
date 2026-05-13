import pickle
import numpy as np
import pandas as pd
import yaml
from data import get_data
from rl_environment import TradingEnvironment, load_data
from rl_trainer import PPOTrainer
import torch
from datetime import datetime
import os
import matplotlib.pyplot as plt
from market_impact import MarketImpactCalculator
from seed_utils import set_seed

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def backtest_rl(config, tickers, from_date, until_date, model_path, consider_market_impact=True):
    cash = config['backtesting']['initial_aum']
    print(f"Initial cash: {cash}")
    env = TradingEnvironment(
        config=config, 
        initial_cash=cash,
        consider_market_impact=consider_market_impact,
        tickers=tickers,
        from_date=from_date,
        until_date=until_date,
        robust_params=None
    )
    
    trainer = PPOTrainer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=config['rl']['hidden_dim']
    )
    trainer.load(model_path)
    
    portfolio_values = []
    positions_by_ticker = {ticker: [] for ticker in tickers}
    trades = []
    daily_returns = []
    cash_values = []
    
    state = env.reset()
    done = False
    
    while not done:
        action = trainer.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        for ticker in tickers:
            positions_by_ticker[ticker].append(info['positions'][ticker])
        daily_returns.append(info['returns'])
        cash_values.append(info['cash'])
        
        state = next_state
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)
    initial_portfolio_value = cash
    cumulative_returns = (portfolio_values / initial_portfolio_value) - 1
    
    sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)

    portfolio_path = np.concatenate(([initial_portfolio_value], portfolio_values))
    peak = np.maximum.accumulate(portfolio_path)
    drawdown = (portfolio_path - peak) / peak
    max_drawdown = np.min(drawdown) 
    
    return {
        'initial_portfolio_value': initial_portfolio_value,
        'portfolio_values': portfolio_values,
        'positions_by_ticker': positions_by_ticker,
        'daily_returns': daily_returns,
        'cumulative_returns': cumulative_returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cash_values': cash_values
    }

def final_backtest_rl(tickers, model_path, 
                      from_date='2022-06-09', until_date='2022-12-09', volume=100000):
    model_name = os.path.basename(model_path)
    model_name = model_name.split('.')[0]
    date_tag = f"{from_date}_to_{until_date}_aligned_start_adverse_impact"
    
    config = load_config()
    print(f"Setting volume in final_backtest_rl to: {volume}")
    config['backtesting']['initial_aum'] = volume
    config['rl']['initial_balance'] = volume
    
    os.makedirs('backtest_rl_results', exist_ok=True)
    
    file1 = f'backtest_rl_results/{model_name}_no_impact_{date_tag}_{volume}.pkl' 
    file2 = f'backtest_rl_results/{model_name}_with_impact_{date_tag}_{volume}.pkl'
    if not (os.path.exists(file1) and os.path.exists(file2)):  
        print("Running backtest without market impact...")
        results_no_impact = backtest_rl(config, tickers, from_date, until_date, model_path, consider_market_impact=False)
        
        print("\nRunning backtest with market impact...")
        results_with_impact = backtest_rl(config, tickers, from_date, until_date, model_path, consider_market_impact=True)
        
        with open(file1, 'wb') as f:
            pickle.dump(results_no_impact, f)
        with open(file2, 'wb') as f:
            pickle.dump(results_with_impact, f)
    else:
        print(f"Loading cached results for volume: {volume}")
        with open(file1, 'rb') as f:
            results_no_impact = pickle.load(f)
        with open(file2, 'rb') as f:
            results_with_impact = pickle.load(f)
        print(f"Initial portfolio value loaded from cache: {results_no_impact['portfolio_values'][0]}")

    plt.figure(figsize=(12, 6))
    
    env = TradingEnvironment(config, initial_cash=volume, tickers=tickers, from_date=from_date, until_date=until_date)
    dates = env.days[env.start_day_idx:env.start_day_idx + len(results_no_impact['portfolio_values'])]
    
    plt.plot(dates, results_no_impact['portfolio_values'], label='Without Market Impact')
    plt.plot(dates, results_with_impact['portfolio_values'], label='With Market Impact')
    
    lookback_period = config['backtesting'].get('lookback_period', 30)
    initial_portfolio_value = results_no_impact['portfolio_values'][0]
    
    benchmark_ticker = 'SPY' if 'SPY' in tickers else tickers[0]
    
    benchmark_data = get_data(benchmark_ticker, from_date, until_date)[1]
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'backtest_rl_results/new_{model_name}_portfolio_comparison_{date_tag}_{volume}.png')
    plt.close()
    
    plt.figure(figsize=(14, 8))
    for ticker in tickers:
        plt.plot(dates, results_with_impact['positions_by_ticker'][ticker], label=f'{ticker} Position')
    
    plt.plot(dates, results_with_impact['cash_values'] / results_with_impact['portfolio_values'][0], label='Cash', linestyle='--')
    
    plt.title('Asset Allocation Over Time (With Market Impact)')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'backtest_rl_results/new_{model_name}_allocations_{date_tag}_{volume}.png')
    plt.close()
    
    comparison_results = {
        'final_portfolio_value': {
            'without_impact': results_no_impact['portfolio_values'][-1],
            'with_impact': results_with_impact['portfolio_values'][-1],
            'difference': results_with_impact['portfolio_values'][-1] - results_no_impact['portfolio_values'][-1]
        },
        'total_return': {
            'without_impact': results_no_impact['cumulative_returns'][-1] * 100,
            'with_impact': results_with_impact['cumulative_returns'][-1] * 100,
            'difference': (results_with_impact['cumulative_returns'][-1] - results_no_impact['cumulative_returns'][-1]) * 100
        },
        'sharpe_ratio': {
            'without_impact': results_no_impact['sharpe_ratio'],
            'with_impact': results_with_impact['sharpe_ratio'],
            'difference': results_with_impact['sharpe_ratio'] - results_no_impact['sharpe_ratio']
        },
        'max_drawdown': {
            'without_impact': results_no_impact['max_drawdown'] * 100,
            'with_impact': results_with_impact['max_drawdown'] * 100,
            'difference': (results_with_impact['max_drawdown'] - results_no_impact['max_drawdown']) * 100
        },
        'final_positions': {ticker: results_with_impact['positions_by_ticker'][ticker][-1] for ticker in tickers}
    }
    
    print("\nComparison Results:")
    print("=" * 50)
    print(f"{'Metric':<25} {'Without Impact':<15} {'With Impact':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'Final Portfolio Value':<25} ${comparison_results['final_portfolio_value']['without_impact']:,.2f} ${comparison_results['final_portfolio_value']['with_impact']:,.2f} ${comparison_results['final_portfolio_value']['difference']:,.2f}")
    print(f"{'Total Return':<25} {comparison_results['total_return']['without_impact']:.2f}% {comparison_results['total_return']['with_impact']:.2f}% {comparison_results['total_return']['difference']:.2f}%")
    print(f"{'Sharpe Ratio':<25} {comparison_results['sharpe_ratio']['without_impact']:.2f} {comparison_results['sharpe_ratio']['with_impact']:.2f} {comparison_results['sharpe_ratio']['difference']:.2f}")
    print(f"{'Maximum Drawdown':<25} {comparison_results['max_drawdown']['without_impact']:.2f}% {comparison_results['max_drawdown']['with_impact']:.2f}% {comparison_results['max_drawdown']['difference']:.2f}%")
    print("-" * 75)
    
    print("\nFinal Positions (With Market Impact):")
    for ticker, position in comparison_results['final_positions'].items():
        print(f"{ticker:<10}: {position:.2f} shares")
    print("=" * 50)
    
    return comparison_results

if __name__ == "__main__":
    os.makedirs('backtest_rl_results', exist_ok=True)
    config = load_config()
    set_seed(config.get('seed', 42))

    test_from_date = '2022-06-09'
    test_until_date = '2022-12-09'
    
    volumes =  [1_000_000] #[100000, 1000000, 2000000, 3000000, 4000000, 5000000]
    for volume in volumes:
        config = load_config()
        config['backtesting']['initial_aum'] = volume
        config['rl']['initial_balance'] = volume
        print(f"\n\n*** Testing with volume: {volume} ***\n")
            
        if not os.path.exists('models'):
            print("No models directory found.")
            exit(1)
        
        model_files = sorted(
            f for f in os.listdir('models')
            if f.endswith('.pth') and f.startswith('SPY_TLT_GLD_EFA_VNQ_best_model_')
        )
        
        if not model_files:
            print("No model files found in the models directory.")
            exit(1)
        
        for model_file in model_files:
            model_path = os.path.join('models', model_file)
            
            tickers = ['SPY', 'TLT', 'GLD', 'EFA', 'VNQ']
            
            print(f"\nTesting model: {model_file} with tickers: {tickers}")
            try:
                final_backtest_rl(tickers, model_path, test_from_date, test_until_date, volume=volume)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Error: Model was trained with a different number of tickers than specified.")
                    print(f"Please ensure you're using the correct tickers for this model.")
                    print(f"Error details: {str(e)}")
                else:
                    raise e 
