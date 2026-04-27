import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from pathlib import Path
from market_impact import MarketImpactCalculator
from data import get_data

MODULE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = MODULE_DIR / 'data'

def load_data(ticker, from_date, until_date):
    df_intra, df_daily = get_data(ticker, from_date, until_date)  
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
    df_intra.to_csv(LOCAL_DATA_DIR / f'intraday_data_{ticker}.csv', index=False)
    df_daily.to_csv(LOCAL_DATA_DIR / f'daily_data_{ticker}.csv', index=False)
    return df_intra, df_daily

class TradingEnvironment(gym.Env):
    def __init__(self, config, initial_cash=100000, consider_market_impact=True, tickers=['SPY'], 
                 from_date='2024-01-01', until_date='2024-12-31', robust_params=None):
        super(TradingEnvironment, self).__init__()

        self.data = {}
        for ticker in tickers:
            df_intra, df_daily = load_data(ticker, from_date, until_date)
            self.data[ticker] = {
                'df_intra': df_intra,
                'df_daily': df_daily
            }
        self.num_tickers = len(tickers) 

        self.robust_params = robust_params
        self.config = config
        self.initial_cash = initial_cash
        self.ticker = ticker
        self.lookback_period = config['backtesting'].get('lookback_period', 30)
        
        self.days = sorted(df_intra['day'].unique())
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_tickers,), dtype=np.float32
        )

        num_features_per_asset = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_tickers * self.lookback_period * num_features_per_asset,),
            dtype=np.float32
        )
        
        self.config['backtesting']['market_impact']['enabled'] = consider_market_impact
        market_impact_config = config['backtesting'].get('market_impact', {})
        self.consider_market_impact = consider_market_impact
        
        if self.consider_market_impact:
            data_config = config.get('data', {})
            self.market_impact = MarketImpactCalculator(
                api_key=data_config.get('api_key'),
                market_impact_window=market_impact_config.get('window', 15),
                impact_threshold=market_impact_config.get('impact_threshold', 0.01),
                max_impact=market_impact_config.get('max_impact', 0.10),
                fallback_model=market_impact_config.get('fallback_model', True),
                api_retry_limit=market_impact_config.get('api_retry_limit', 3),
                cache_results=market_impact_config.get('cache_results', True),
                provider=data_config.get('provider', 'polygon'),
                wrds_root_dir=data_config.get('wrds_root_dir'),
                wrds_trades_subdir=data_config.get('wrds_trades_subdir', 'raw/taq_trades'),
                wrds_file_format=data_config.get('wrds_file_format', 'parquet')
            )
        
        self.reset()
    
    def reset(self):
        self.current_day_idx = self.lookback_period
        self.cash = self.initial_cash
        self.positions = {ticker: 0 for ticker in self.data.keys()}
        self.entry_prices = {ticker: 0 for ticker in self.data.keys()}
        self.portfolio_value = [self.initial_cash]
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self):
        current_day = self.days[self.current_day_idx]
        prev_day = self.days[self.current_day_idx - 1]
        
        all_features = []
        
        for ticker in self.data.keys():
            df_intra = self.data[ticker]['df_intra']
            
            lookback_data = df_intra[
                (df_intra['day'] >= self.days[self.current_day_idx - self.lookback_period]) &
                (df_intra['day'] < current_day)
            ]
            
            prices = lookback_data['close'].values
            volumes = lookback_data['volume'].values
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            volatilities = pd.Series(prices).rolling(window=20).std().values
            volatilities = np.nan_to_num(volatilities, 0)
            
            prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            volumes_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
            returns_norm = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            volatilities_norm = (volatilities - np.mean(volatilities)) / (np.std(volatilities) + 1e-8)
            
            ticker_features = np.concatenate([
                prices_norm[-self.lookback_period:],
                volumes_norm[-self.lookback_period:],
                returns_norm[-self.lookback_period:],
                volatilities_norm[-self.lookback_period:]
            ])
            
            all_features.append(ticker_features)
        
        obs = np.concatenate(all_features)
        
        return obs
    
    def step(self, action):
        current_day = self.days[self.current_day_idx]
        
        prev_portfolio_value = self.cash
        for ticker, position in self.positions.items():
            current_data = self.data[ticker]['df_intra'][self.data[ticker]['df_intra']['day'] == current_day]
            prev_price = current_data['close'].iloc[0]
            prev_portfolio_value += position * prev_price
        
        info = {'portfolio_value': 0, 'cash': self.cash, 'positions': {}, 'returns': 0, 'volatility': 0}
        
        for i, ticker in enumerate(self.data.keys()):
            current_data = self.data[ticker]['df_intra'][self.data[ticker]['df_intra']['day'] == current_day]
            
            current_price = current_data['close'].iloc[-1]
            prev_price = current_data['close'].iloc[0]
            
            max_shares = int(self.initial_cash / current_price)
            target_shares = int(action[i] * max_shares)
            position_change = target_shares - self.positions[ticker]
            
            effective_price = current_price
            if self.consider_market_impact and abs(position_change) > 0:
                volume_change = abs(position_change)
                effective_price = self.market_impact.calculate_market_impact(
                    ticker, 
                    current_price, 
                    volume_change, 
                    'buy' if position_change > 0 else 'sell',
                    current_data.index[-1]
                )
            
            if self.robust_params is not None:
                robust_type = self.robust_params["robust_type"]
                distribution = np.array([self.robust_params["beta"], 1-2*self.robust_params["beta"], self.robust_params["beta"]]) 
                try:
                    distribution_shift = np.load(f'u_star_{robust_type}.pkl')
                except:
                    distribution_shift = - self.robust_params["epsilon"] * np.ones(self.robust_params["u_dim"])
                    distribution_shift[0] +=  self.robust_params["epsilon"] * self.robust_params["u_dim"]
                if robust_type == "p1N2" or robust_type == "p1":
                    try: 
                        sample = np.random.choice([0, 1, 2], p=distribution+distribution_shift)
                        if sample == 0:
                            effective_price = effective_price * (1 + self.robust_params["epsilon"]) 
                        elif sample == 1:
                            pass 
                        else:
                            effective_price = effective_price * (1 - self.robust_params["epsilon"]) 
                    except:   
                        if position_change > 0:
                            effective_price = effective_price * (1 + self.robust_params["epsilon"])
                        else:
                            effective_price = effective_price * (1 - self.robust_params["epsilon"]) 
                else:
                    raise Exception(f"Unsupported robust type: {robust_type}")

            if abs(position_change) > 0:
                if position_change > 0:
                    self.cash -= position_change * effective_price
                    self.positions[ticker] += position_change
                    self.entry_prices[ticker] = effective_price
                else:
                    self.cash += abs(position_change) * effective_price
                    self.positions[ticker] -= abs(position_change)
                
                self.trades.append({
                    'day': current_day,
                    'ticker': ticker,
                    'action': position_change,
                    'shares': abs(position_change),
                    'price': effective_price,
                    'market_price': current_price,
                    'price_impact': effective_price - current_price
                })
            
            info['positions'][ticker] = self.positions[ticker]
            info[f'{ticker}_effective_price'] = effective_price
            info[f'{ticker}_market_price'] = current_price
            info[f'{ticker}_price_impact'] = effective_price - current_price if abs(position_change) > 0 else 0
        
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            current_data = self.data[ticker]['df_intra'][self.data[ticker]['df_intra']['day'] == current_day]
            current_price = current_data['close'].iloc[-1]
            portfolio_value += position * current_price
        
        self.portfolio_value.append(portfolio_value)
        
        returns = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        volatility = np.std([self.portfolio_value[-1] / p - 1 for p in self.portfolio_value[-20:]])
        reward = returns / (volatility + 1e-8)
        
        total_position_change_ratio = 0
        for i, ticker in enumerate(self.data.keys()):
            current_data = self.data[ticker]['df_intra'][self.data[ticker]['df_intra']['day'] == current_day]
            current_price = current_data['close'].iloc[-1]
            max_shares = int(self.initial_cash / current_price)
            target_shares = int(action[i] * max_shares)
            position_change = target_shares - self.positions[ticker]
            total_position_change_ratio += abs(position_change) / max_shares
        
        reward -= 0.001 * total_position_change_ratio
        
        self.current_day_idx += 1
        
        done = self.current_day_idx >= len(self.days) - 1
        
        obs = self._get_observation()
        
        info['portfolio_value'] = portfolio_value
        info['cash'] = self.cash
        info['returns'] = returns
        info['volatility'] = volatility
        
        return obs, reward, done, info 
