import pandas as pd
import numpy as np
from strategy import MomentumStrategy
import statsmodels.api as sm
from market_impact import MarketImpactCalculator

class BacktestEngine:
    def __init__(self, config, strategy_class=MomentumStrategy):
        self.config = config
        self.strategy = strategy_class(config)
        self.initial_aum = config['backtesting']['initial_aum']
        self.commission = config['backtesting']['commission']
        self.min_comm_per_order = config['backtesting']['min_commission_per_order']
        
        # Initialize market impact calculator if enabled
        market_impact_config = config['backtesting'].get('market_impact', {})
        self.consider_market_impact = market_impact_config.get('enabled', True)
        
        if self.consider_market_impact:
            self.market_impact = MarketImpactCalculator(
                api_key=config['data']['api_key'],
                market_impact_window=market_impact_config.get('window', 15),
                impact_threshold=market_impact_config.get('impact_threshold', 0.01),
                max_impact=market_impact_config.get('max_impact', 0.10),
                fallback_model=market_impact_config.get('fallback_model', True),
                api_retry_limit=market_impact_config.get('api_retry_limit', 3),
                cache_results=market_impact_config.get('cache_results', True)
            )

    def calculate_effective_price(self, ticker, price, volume, side, timestamp):
        """Calculate effective price including market impact"""
        if self.consider_market_impact:
            # Convert timestamp to datetime if it's not already
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)
            return self.market_impact.calculate_market_impact(ticker, price, volume, side, timestamp)
        return price

    def run(self, ticker, df_intra, df_daily):
        """Run the backtest"""
        # Initialize strategy DataFrame
        all_days = df_intra['day'].unique()
        strat = pd.DataFrame(index=all_days)
        strat['ret'] = np.nan
        strat['AUM'] = self.initial_aum
        strat['ret_spy'] = np.nan

        # Set index of df_daily to be the date
        df_daily['day'] = pd.to_datetime(df_daily['caldt']).dt.date
        df_daily.set_index('day', inplace=True)

        # Calculate daily returns for SPY if not already present
        if 'ret' not in df_daily.columns:
            df_daily['ret'] = df_daily['close'].pct_change()

        # Group data by day for faster access
        daily_groups = df_intra.groupby('day')

        # Loop through all days
        for d in range(1, len(all_days)):
            current_day = all_days[d]
            prev_day = all_days[d-1]
            
            if prev_day in daily_groups.groups and current_day in daily_groups.groups:
                prev_day_data = daily_groups.get_group(prev_day)
                current_day_data = daily_groups.get_group(current_day)

                if 'sigma_open' in current_day_data.columns and current_day_data['sigma_open'].isna().all():
                    continue

                # Calculate adjusted previous close
                prev_close_adjusted = prev_day_data['close'].iloc[-1] - df_intra.loc[current_day_data.index, 'dividend'].iloc[-1]

                # Get strategy signals
                signals = self.strategy.calculate_signals(current_day_data, prev_day_data, prev_close_adjusted)

                # Calculate position size
                previous_aum = strat.loc[prev_day, 'AUM']
                open_price = current_day_data['open'].iloc[0]
                spx_vol = current_day_data['spy_dvol'].iloc[0]
                shares = self.strategy.calculate_position_size(previous_aum, open_price, spx_vol)

                # Apply trading frequency
                exposure = self.strategy.apply_trading_frequency(signals, current_day_data['min_from_open'])

                # Calculate trades and PnL with market impact
                trades_count = np.sum(np.abs(np.diff(np.append(exposure, 0))))
                change_1m = current_day_data['close'].diff()

                # Calculate effective prices with market impact
                if trades_count > 0:
                    # For buys (positive exposure changes)
                    buy_mask = np.diff(np.append(exposure, 0)) > 0
                    if np.any(buy_mask):
                        buy_prices = current_day_data['close'].values[buy_mask]
                        buy_volumes = shares * np.abs(np.diff(np.append(exposure, 0))[buy_mask])
                        effective_buy_prices = [
                            self.calculate_effective_price(ticker, p, v, 'buy', current_day_data.index[i])
                            for i, (p, v) in enumerate(zip(buy_prices, buy_volumes))
                        ]
                        # For buys: higher effective price means worse execution
                        change_1m[buy_mask] = buy_prices - np.array(effective_buy_prices)

                    # For sells (negative exposure changes)
                    sell_mask = np.diff(np.append(exposure, 0)) < 0
                    if np.any(sell_mask):
                        sell_prices = current_day_data['close'].values[sell_mask]
                        sell_volumes = shares * np.abs(np.diff(np.append(exposure, 0))[sell_mask])
                        effective_sell_prices = [
                            self.calculate_effective_price(ticker, p, v, 'sell', current_day_data.index[i])
                            for i, (p, v) in enumerate(zip(sell_prices, sell_volumes))
                        ]
                        # For sells: lower effective price means worse execution
                        change_1m[sell_mask] = np.array(effective_sell_prices) - sell_prices

                gross_pnl = np.sum(exposure * change_1m) * shares
                commission_paid = trades_count * max(self.min_comm_per_order, self.commission * shares)
                net_pnl = gross_pnl - commission_paid

                # Update strategy metrics
                strat.loc[current_day, 'AUM'] = previous_aum + net_pnl
                strat.loc[current_day, 'ret'] = net_pnl / previous_aum
                
                # Get SPY return for the current day
                if current_day in df_daily.index:
                    strat.loc[current_day, 'ret_spy'] = df_daily.loc[current_day, 'ret']
                else:
                    # If the day is not in df_daily, use the previous day's return
                    strat.loc[current_day, 'ret_spy'] = strat.loc[prev_day, 'ret_spy']

        # Calculate cumulative returns
        strat['AUM_SPX'] = self.initial_aum * (1 + strat['ret_spy']).cumprod(skipna=True)
        
        return strat

    def calculate_statistics(self, strat):
        """Calculate strategy statistics"""
        # Check if we have enough data
        if len(strat['ret'].dropna()) < 2:
            return {
                'Error': 'Not enough data to calculate statistics'
            }

        try:
            stats = {
                'Total Return (%)': round((np.prod(1 + strat['ret'].dropna()) - 1) * 100, 0),
                'Annualized Return (%)': round((np.prod(1 + strat['ret']) ** (252 / len(strat['ret'])) - 1) * 100, 1),
                'Annualized Volatility (%)': round(strat['ret'].dropna().std() * np.sqrt(252) * 100, 1),
                'Sharpe Ratio': round(strat['ret'].dropna().mean() / strat['ret'].dropna().std() * np.sqrt(252), 2),
                'Hit Ratio (%)': round((strat['ret'] > 0).sum() / (strat['ret'].abs() > 0).sum() * 100, 0),
                'Maximum Drawdown (%)': round(strat['AUM'].div(strat['AUM'].cummax()).sub(1).min() * -100, 0)
            }

            # Calculate alpha and beta only if we have enough data
            Y = strat['ret'].dropna()
            X = strat['ret_spy'].reindex(Y.index).dropna()
            if len(Y) > 0 and len(X) > 0:
                X = sm.add_constant(X)
                model = sm.OLS(Y, X).fit()
                stats['Alpha (%)'] = round(model.params.const * 100 * 252, 2)
                stats['Beta'] = round(model.params['ret_spy'], 2)
            else:
                stats['Alpha (%)'] = 'N/A'
                stats['Beta'] = 'N/A'

            return stats
        except Exception as e:
            return {
                'Error': f'Failed to calculate statistics: {str(e)}'
            } 