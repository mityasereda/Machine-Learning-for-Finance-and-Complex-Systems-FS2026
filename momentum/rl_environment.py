import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from market_impact import MarketImpactCalculator

class TradingEnvironment(gym.Env):
    """
    Custom Environment for RL trading that works with the existing framework
    """
    def __init__(self, df_intra, df_daily, config, initial_cash=100000, consider_market_impact=True, ticker='SPY', robust_params=None, granularity='day'):
        super(TradingEnvironment, self).__init__()

        self.robust_params = robust_params
        self.df_intra = df_intra
        self.df_daily = df_daily
        self.config = config
        self.initial_cash = initial_cash
        self.ticker = ticker
        self.granularity = granularity.lower()  # 'day' or 'minute'
        
        if self.granularity not in ['day', 'minute']:
            raise ValueError("granularity must be either 'day' or 'minute'")
            
        # Calculate lookback period from config
        self.lookback_period = config['backtesting'].get('lookback_period', 30)
        
        # Get unique days for episode steps
        self.days = sorted(df_intra['day'].unique())
        
        # Index the intraday data for faster access (needed for minute-level)
        if self.granularity == 'minute':
            self.df_intra_indexed = df_intra.rename_axis('index').reset_index()
            self.index_position_map = {
                idx: pos for pos, idx in enumerate(self.df_intra_indexed['index'].tolist())
            }
            self.total_steps = len(self.df_intra_indexed)
            self._create_day_indices_map()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32  # Single asset for now
        )
        
        # Enhanced state space: price, volume, returns, volatility
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(4 * self.lookback_period,),  # 4 features * lookback period
            dtype=np.float32
        )
        
        # Initialize market impact calculator if enabled
        self.config['backtesting']['market_impact']['enabled'] = consider_market_impact
        market_impact_config = config['backtesting'].get('market_impact', {})
        self.consider_market_impact = consider_market_impact
        
        if self.consider_market_impact:
            data_config = config.get('data', {})
            self.market_impact = MarketImpactCalculator(
                api_key=data_config.get('api_key'),
                market_impact_window=market_impact_config.get('window', 15),
                eta=market_impact_config.get('eta', 0.1),
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
    
    def _create_day_indices_map(self):
        """Create a mapping from days to indices in the dataframe for minute-level granularity"""
        self.day_indices = {}
        for day in self.days:
            day_data = self.df_intra[self.df_intra['day'] == day]
            day_indices = [self.index_position_map[idx] for idx in day_data.index]
            self.day_indices[day] = day_indices
    
    def reset(self):
        """Reset the environment"""
        if self.granularity == 'day':
            # Day-level reset
            self.current_day_idx = self.lookback_period
        else:
            # Minute-level reset
            # Check if we've gone through all days
            if not hasattr(self, 'current_day_idx') or self.current_day_idx >= len(self.days):
                self.current_day_idx = self.lookback_period  # Restart from beginning

            # Get the current day and its first data point index
            current_day = self.days[self.current_day_idx]
            self.current_step_idx = self.day_indices[current_day][0]  # First minute of the day
            self.current_episode_day = current_day
        
        # Reset portfolio state (common for both granularities)
        self.cash = self.initial_cash
        self.position = 0  # 初始化为0股
        self.entry_price = 0
        self.portfolio_value = [self.initial_cash]
        self.trades = []
        
        return self._get_observation()
    
    def _find_lookback_start_idx(self):
        """Find appropriate starting index that has enough lookback data for minute granularity"""
        current_day = self.days[self.current_day_idx]
        # Get the indices for the lookback period
        lookback_indices = []
        for i in range(self.lookback_period):
            if self.current_day_idx - i - 1 >= 0:
                prev_day = self.days[self.current_day_idx - i - 1]
                lookback_indices.extend(self.day_indices[prev_day])
        
        # Add current day indices
        lookback_indices.extend(self.day_indices[current_day])
        
        # Sort them to ensure chronological order
        lookback_indices.sort()
        
        return lookback_indices
    
    def _get_observation(self):
        """Get current observation with enhanced features"""
        if self.granularity == 'day':
            # Day-level observation
            current_day = self.days[self.current_day_idx]
            prev_day = self.days[self.current_day_idx - 1]
            
            # Get data for current and previous day
            current_data = self.df_intra[self.df_intra['day'] == current_day]
            prev_data = self.df_intra[self.df_intra['day'] == prev_day]
            
            # Get lookback window data
            lookback_data = self.df_intra[
                (self.df_intra['day'] >= self.days[self.current_day_idx - self.lookback_period]) &
                (self.df_intra['day'] < current_day)
            ]
            
            # Calculate features
            prices = lookback_data['close'].values
            volumes = lookback_data['volume'].values
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Pad with 0
            
            # Calculate volatility
            volatilities = pd.Series(prices).rolling(window=20).std().values
            volatilities = np.nan_to_num(volatilities, 0)
            
            # Normalize features
            prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            volumes_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
            returns_norm = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            volatilities_norm = (volatilities - np.mean(volatilities)) / (np.std(volatilities) + 1e-8)
            
            # Combine features
            obs = np.concatenate([
                prices_norm[-self.lookback_period:],
                volumes_norm[-self.lookback_period:],
                returns_norm[-self.lookback_period:],
                volatilities_norm[-self.lookback_period:]
            ])
        else:
            # Minute-level observation
            # Get all lookback data across previous days and current day up to current step
            lookback_indices = self._find_lookback_start_idx()
            
            # Filter indices up to current step
            valid_indices = [idx for idx in lookback_indices if idx <= self.current_step_idx]
            
            # Get the last lookback_period data points or all available if fewer
            if len(valid_indices) > self.lookback_period:
                lookback_indices = valid_indices[-self.lookback_period:]
            else:
                lookback_indices = valid_indices
            
            # Get the actual data from the indices
            lookback_data_indices = [self.df_intra_indexed.iloc[idx]['index'] for idx in lookback_indices]
            lookback_data = self.df_intra.loc[lookback_data_indices]
            
            # Calculate features
            prices = lookback_data['close'].values
            volumes = lookback_data['volume'].values
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Pad with 0
            
            # Calculate volatility
            volatilities = pd.Series(prices).rolling(window=min(20, len(prices))).std().values
            volatilities = np.nan_to_num(volatilities, 0)
            
            # Normalize features
            prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            volumes_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
            returns_norm = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            volatilities_norm = (volatilities - np.mean(volatilities)) / (np.std(volatilities) + 1e-8)
            
            # Ensure we have exactly lookback_period elements for each feature
            pad_length = self.lookback_period - len(prices_norm)
            if pad_length > 0:
                prices_norm = np.pad(prices_norm, (pad_length, 0), 'constant')
                volumes_norm = np.pad(volumes_norm, (pad_length, 0), 'constant')
                returns_norm = np.pad(returns_norm, (pad_length, 0), 'constant')
                volatilities_norm = np.pad(volatilities_norm, (pad_length, 0), 'constant')
            elif pad_length < 0:
                prices_norm = prices_norm[-self.lookback_period:]
                volumes_norm = volumes_norm[-self.lookback_period:]
                returns_norm = returns_norm[-self.lookback_period:]
                volatilities_norm = volatilities_norm[-self.lookback_period:]
            
            # Combine features
            obs = np.concatenate([
                prices_norm,
                volumes_norm,
                returns_norm,
                volatilities_norm
            ])
        
        return obs
    
    def step(self, action):
        """
        Execute one step in the environment
        action: float between -1 and 1, representing position size
        """
        if self.granularity == 'day':
            # Day-level step
            current_day = self.days[self.current_day_idx]
            current_data = self.df_intra[self.df_intra['day'] == current_day]
            
            # Get current price and calculate returns
            current_price = current_data['close'].iloc[-1]
            prev_price = current_data['close'].iloc[0]
            price_return = (current_price - prev_price) / prev_price
            
            # Calculate portfolio value before action
            prev_portfolio_value = self.cash + self.position * prev_price
            
            # 将动作值转换为目标股数
            max_shares = int(self.initial_cash / current_price)  # 最大可买股数
            target_shares = int(action[0] * max_shares)  # 将比例转换为股数
            position_change = target_shares - self.position  # 需要调整的股数
            
            # Calculate effective price with market impact BEFORE executing the trade
            effective_price = current_price
            if self.consider_market_impact and abs(position_change) > 0:
                volume_change = abs(position_change) 
                effective_price = self.market_impact.calculate_market_impact(
                    self.ticker, 
                    current_price, 
                    volume_change, 
                    'buy' if position_change > 0 else 'sell',
                    current_data.index[-1]
                )
            
            # Apply robust parameters logic if provided
            if self.robust_params is not None:
                robust_type = self.robust_params["robust_type"]
                distribution = np.array([self.robust_params["beta"], 1-2*self.robust_params["beta"], self.robust_params["beta"]]) 
                try:
                    distribution_shift = np.load(f'u_star_{robust_type}.npy')
                except:
                    distribution_shift = - self.robust_params["epsilon"] * np.ones(self.robust_params["u_dim"])
                    distribution_shift[0] +=  self.robust_params["epsilon"] * self.robust_params["u_dim"]
                
                if robust_type == "p1N2":
                    try: 
                        sample = np.random.choice([0, 1, 2], p=distribution+distribution_shift) # Force N=1
                        if sample == 0:
                            effective_price = effective_price + self.robust_params["epsilon"] 
                        elif sample == 1:
                            pass 
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                    except:   
                        if position_change > 0:
                            effective_price = effective_price + self.robust_params["epsilon"]
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                elif robust_type == "p1":
                    try: 
                        sample = np.random.choice([0, 1, 2], p=distribution+distribution_shift) # Force N=1
                        if sample == 0:
                            effective_price = effective_price + self.robust_params["epsilon"] 
                        elif sample == 1:
                            pass 
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                    except:   
                        if np.random.rand() < 0.5:
                            effective_price = effective_price + self.robust_params["epsilon"]
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"]
                else:
                    raise Exception(f"Unsupported robust type: {robust_type}")
            
            if abs(position_change) > 0:  # 只要有变化就执行交易
                # Execute trade using the effective price (with market impact)
                if position_change > 0:  # Buy
                    self.cash -= position_change * effective_price
                    self.position += position_change
                    self.entry_price = effective_price
                else:  # Sell
                    self.cash += abs(position_change) * effective_price
                    self.position -= abs(position_change)
                
                self.trades.append({
                    'day': current_day,
                    'action': position_change,
                    'shares': abs(position_change),
                    'price': effective_price,
                    'market_price': current_price,
                    'price_impact': effective_price - current_price
                })
            
            # Calculate portfolio value after action
            portfolio_value = self.cash + self.position * current_price
            self.portfolio_value.append(portfolio_value)
            
            # Calculate reward (Sharpe-like ratio)
            returns = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            volatility = np.std([self.portfolio_value[-1] / p - 1 for p in self.portfolio_value[-20:]])
            reward = returns / (volatility + 1e-8)
            
            # Penalize excessive trading
            reward -= 0.001 * abs(position_change) / max_shares  # 根据最大股数归一化的交易惩罚
            
            # Update state
            self.current_day_idx += 1
            
            # Check if episode is done
            done = self.current_day_idx >= len(self.days) - 1
            
            # Get new observation
            obs = self._get_observation()
            
            # Additional info
            info = {
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position': self.position,
                'returns': returns,
                'volatility': volatility,
                'effective_price': effective_price,
                'market_price': current_price,
                'price_impact': effective_price - current_price if abs(position_change) > 0 else 0,
                'date': current_day,
            }
        else:
            # Minute-level step
            # Get current and previous minute data
            current_idx = self.df_intra_indexed.iloc[self.current_step_idx]['index']
            current_data = self.df_intra.loc[current_idx:current_idx]
            
            # If we have a previous step, get that data, otherwise use current data
            if self.current_step_idx > 0:
                prev_idx = self.df_intra_indexed.iloc[self.current_step_idx - 1]['index']
                prev_data = self.df_intra.loc[prev_idx:prev_idx]
                prev_price = prev_data['close'].iloc[0]
            else:
                prev_price = current_data['close'].iloc[0]
            
            # Get current price and calculate returns
            current_price = current_data['close'].iloc[0]
            price_return = (current_price - prev_price) / prev_price
            
            # Calculate portfolio value before action
            prev_portfolio_value = self.cash + self.position * prev_price
            
            # 将动作值转换为目标股数
            max_shares = int(self.initial_cash / current_price)  # 最大可买股数
            target_shares = int(action[0] * max_shares)  # 将比例转换为股数
            position_change = target_shares - self.position  # 需要调整的股数
            
            # Calculate effective price with market impact BEFORE executing the trade
            effective_price = current_price
            if self.consider_market_impact and abs(position_change) > 0:
                volume_change = abs(position_change) 
                effective_price = self.market_impact.calculate_market_impact(
                    self.ticker, 
                    current_price, 
                    volume_change, 
                    'buy' if position_change > 0 else 'sell',
                    current_data.index[0]
                )
            
            # Apply robust parameters logic if provided
            if self.robust_params is not None:
                robust_type = self.robust_params["robust_type"]
                distribution = np.array([0.25, 0.5, 0.25]) 
                try:
                    distribution_shift = np.load(f'u_star_{robust_type}.npy')
                except:
                    distribution_shift = - self.robust_params["beta"] * np.ones(self.robust_params["u_dim"])
                    distribution_shift[0] +=  self.robust_params["beta"] * self.robust_params["u_dim"]
                
                if robust_type == "p1N2":
                    try: 
                        sample = np.random.choice([0, 1, 2], p=distribution+distribution_shift) # Force N=1
                        if sample == 0:
                            effective_price = effective_price + self.robust_params["epsilon"] 
                        elif sample == 1:
                            pass 
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                    except:   
                        if position_change > 0:
                            effective_price = effective_price + self.robust_params["epsilon"]
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                elif robust_type == "p1":
                    try: 
                        sample = np.random.choice([0, 1, 2], p=distribution+distribution_shift) # Force N=1
                        if sample == 0:
                            effective_price = effective_price + self.robust_params["epsilon"] 
                        elif sample == 1:
                            pass 
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"] 
                    except:   
                        if np.random.rand() < 0.5:
                            effective_price = effective_price + self.robust_params["epsilon"]
                        else:
                            effective_price = effective_price - self.robust_params["epsilon"]
                else:
                    raise Exception(f"Unsupported robust type: {robust_type}")
            else:
                if np.random.rand() < 0.25:
                    effective_price += 0.001 
                elif np.random.rand() > 0.75:
                    effective_price -= 0.001 
                else:
                    effective_price = effective_price 
            
            if abs(position_change) > 0:  # 只要有变化就执行交易
                # Execute trade using the effective price (with market impact)
                if position_change > 0:  # Buy
                    self.cash -= position_change * effective_price
                    self.position += position_change
                    self.entry_price = effective_price
                else:  # Sell
                    self.cash += abs(position_change) * effective_price
                    self.position -= abs(position_change)
                
                current_day = current_data['day'].iloc[0]
                self.trades.append({
                    'day': current_day,
                    'timestamp': current_data.index[0],
                    'action': position_change,
                    'shares': abs(position_change),
                    'price': effective_price,
                    'market_price': current_price,
                    'price_impact': effective_price - current_price
                })
            
            # Calculate portfolio value after action
            portfolio_value = self.cash + self.position * current_price
            self.portfolio_value.append(portfolio_value)
            
            # Calculate reward (Sharpe-like ratio)
            returns = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            volatility = np.std([self.portfolio_value[-1] / p - 1 for p in self.portfolio_value[-min(20, len(self.portfolio_value)):]])
            reward = returns / (volatility + 1e-8)
            
            # Penalize excessive trading
            reward -= 0.001 * abs(position_change) / max_shares  # 根据最大股数归一化的交易惩罚
            
            # Update state - move to next minute
            self.current_step_idx += 1
            
            # Get the day of the next step (if available)
            done = False
            if self.current_step_idx < self.total_steps:
                next_idx = self.df_intra_indexed.iloc[self.current_step_idx]['index']
                next_data = self.df_intra.loc[next_idx:next_idx]
                next_day = next_data['day'].iloc[0]
                
                # End episode if we've moved to a new day or reached the end of data
                done = (next_day != self.current_episode_day)
            else:
                # We've reached the end of data
                done = True
            
            # If episode is done, move to the next day for the next episode
            if done:
                self.current_day_idx += 1
            
            # Get new observation
            obs = self._get_observation()
            
            # Additional info
            info = {
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position': self.position,
                'returns': returns,
                'volatility': volatility,
                'effective_price': effective_price,
                'market_price': current_price,
                'price_impact': effective_price - current_price if abs(position_change) > 0 else 0,
                'timestamp': current_data.index[0],
                'day': current_data['day'].iloc[0]
            }
        
        return obs, reward, done, info 
