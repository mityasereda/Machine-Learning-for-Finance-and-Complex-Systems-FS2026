import pickle
import numpy as np
import pandas as pd
import yaml
from data import get_data
from rl_environment import TradingEnvironment
from rl_model import PPOTrainer
import torch
from datetime import datetime
import os
import matplotlib.pyplot as plt
from market_impact import MarketImpactCalculator
from seed_utils import set_seed

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data(ticker, from_date, until_date, override=False):
    # Parameters
    # check if data is already in csv
    if os.path.exists('data/intraday_data.csv') and os.path.exists('data/daily_data.csv') and not override:
        df_intra = pd.read_csv('data/intraday_data.csv')
        df_daily = pd.read_csv('data/daily_data.csv')
    else:
        df_intra, df_daily = get_data(ticker, from_date, until_date)  
        # save data to csv
        df_intra.to_csv('data/intraday_data.csv', index=False)
        df_daily.to_csv('data/daily_data.csv', index=False)
    return df_intra, df_daily

def get_context_buffer(ticker, before_date, lookback_period):
    """Load the last lookback_period trading days before before_date as a warm-up context."""
    ctx_intra, ctx_daily = get_data(ticker, '2022-04-01', before_date)
    if ctx_intra.empty:
        return ctx_intra, ctx_daily
    days = sorted(ctx_intra['day'].unique())
    ctx_days = set(days[-lookback_period:])
    ctx_daily = ctx_daily.copy()
    ctx_daily['day'] = pd.to_datetime(ctx_daily['caldt']).dt.date
    return (
        ctx_intra[ctx_intra['day'].isin(ctx_days)].copy(),
        ctx_daily[ctx_daily['day'].isin(ctx_days)].copy(),
    )


def backtest_rl(config, df_intra, df_daily, model_path, consider_market_impact=True,
                ctx_intra=None, ctx_daily=None):
    """Backtest the trained RL model"""
    granularity = config['backtesting'].get('granularity', 'day')

    if ctx_intra is not None and not ctx_intra.empty:
        env_intra = pd.concat([ctx_intra, df_intra], ignore_index=True)
        env_daily = pd.concat([ctx_daily, df_daily], ignore_index=True)
    else:
        env_intra, env_daily = df_intra, df_daily

    # Create environment
    env = TradingEnvironment(
        env_intra,
        env_daily,
        config,
        initial_cash=config['backtesting']['initial_aum'],
        consider_market_impact=consider_market_impact,
        granularity=granularity,
    )
    
    # Load trained model
    trainer = PPOTrainer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=config['rl']['hidden_dim']
    )
    trainer.load(model_path)
    
    # Initialize tracking variables
    portfolio_values = []
    positions = []
    trades = []
    daily_returns = []
    dates = []
    
    # Reset environment
    state = env.reset()
    done = False
    
    while not done:
        # Select action using trained model
        action = trainer.select_action(state)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Record results
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        daily_returns.append(info['returns'])
        if 'date' in info:
            dates.append(info['date'])
        
        # Update state
        state = next_state
    
    # Calculate performance metrics from the realized equity curve.
    portfolio_values = np.array(portfolio_values)
    initial_portfolio_value = env.initial_cash
    portfolio_path = np.concatenate(([initial_portfolio_value], portfolio_values))
    period_returns = portfolio_path[1:] / portfolio_path[:-1] - 1

    cumulative_returns = (portfolio_values / initial_portfolio_value) - 1
    sharpe_ratio = np.sqrt(252) * np.mean(period_returns) / (np.std(period_returns) + 1e-8)

    peak = np.maximum.accumulate(portfolio_path)
    drawdown = (portfolio_path - peak) / peak
    max_drawdown = np.min(drawdown)
    return {
        'initial_portfolio_value': initial_portfolio_value,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'daily_returns': period_returns,
        'cumulative_returns': cumulative_returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'dates': pd.to_datetime(dates) if dates else None,
    }

def final_backtest_rl(ticker, model_path):
    # Extract model name from path
    model_name = os.path.basename(model_path)
    model_name = model_name.split('.')[0]
    
    # Load configuration
    config = load_config()
    
    # Load data for backtesting period
    from_date = '2022-06-09'
    until_date = '2022-12-09'
    df_intra, df_daily = load_data(ticker, from_date, until_date, override=True)
    if df_intra.empty or df_daily.empty:
        print(f"Warning: API returned empty data for {ticker} from {from_date} to {until_date}")
        return

    # Run backtest without market impact
    # No context buffer: the first lookback_period days of the test window are
    # used as a warm-up buffer (features computed but no trades recorded),
    # ensuring clean separation from the training period.
    print("Running backtest without market impact...")
    results_no_impact = backtest_rl(config, df_intra, df_daily, model_path,
                                    consider_market_impact=False)

    # Run backtest with market impact
    print("\nRunning backtest with market impact...")
    results_with_impact = backtest_rl(config, df_intra, df_daily, model_path,
                                      consider_market_impact=True)
    
    # save results (dict) to pickle
    with open(f'backtest_rl_results/{model_name}_no_impact.pkl', 'wb') as f:
        pickle.dump(results_no_impact, f)
    with open(f'backtest_rl_results/{model_name}_with_impact.pkl', 'wb') as f:
        pickle.dump(results_with_impact, f)

    # Plot comparison
    plt.figure(figsize=(12, 6))
    x_values = range(len(results_no_impact['portfolio_values']))
    plt.plot(x_values, results_no_impact['portfolio_values'], label='Without Market Impact')
    plt.plot(x_values, results_with_impact['portfolio_values'], label='With Market Impact')
    
    # Calculate SPY performance using daily returns, starting from lookback period
    initial_portfolio_value = results_no_impact['portfolio_values'][0]
    spy_returns = df_daily['close'].pct_change().dropna()
    spy_returns = spy_returns[:len(results_no_impact['portfolio_values'])]
    spy_cumulative = initial_portfolio_value * (1 + spy_returns).cumprod()
    ticker_label = model_name.split('_', 1)[0]
    plt.plot(x_values, spy_cumulative.values, label=f'{ticker_label} Price', linestyle='--', alpha=0.7)
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'backtest_rl_results/new_{model_name}_comparison.png')
    plt.close()
    
    # Create comparison results dictionary
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
        }
    }
    
    # Print comparison for reference
    print("\nComparison Results:")
    print("=" * 50)
    print(f"{'Metric':<25} {'Without Impact':<15} {'With Impact':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'Final Portfolio Value':<25} ${comparison_results['final_portfolio_value']['without_impact']:,.2f} ${comparison_results['final_portfolio_value']['with_impact']:,.2f} ${comparison_results['final_portfolio_value']['difference']:,.2f}")
    print(f"{'Total Return':<25} {comparison_results['total_return']['without_impact']:.2f}% {comparison_results['total_return']['with_impact']:.2f}% {comparison_results['total_return']['difference']:.2f}%")
    print(f"{'Sharpe Ratio':<25} {comparison_results['sharpe_ratio']['without_impact']:.2f} {comparison_results['sharpe_ratio']['with_impact']:.2f} {comparison_results['sharpe_ratio']['difference']:.2f}")
    print(f"{'Maximum Drawdown':<25} {comparison_results['max_drawdown']['without_impact']:.2f}% {comparison_results['max_drawdown']['with_impact']:.2f}% {comparison_results['max_drawdown']['difference']:.2f}%")
    print("=" * 50)
    
    return comparison_results

if __name__ == "__main__":
    config = load_config()
    set_seed(config.get('seed', 42))
    # enumerate all files in models folder
    for file in os.listdir('models'):
        if file.endswith('.pth'):
            ticker = file.split('_')[0]
            model_path = os.path.join('models', file)
            final_backtest_rl(ticker, model_path) 
    # ticker = 'SPY'
    # model_path = os.path.join('models', 'best_model.pth')
    # final_backtest_rl(ticker, model_path) 
