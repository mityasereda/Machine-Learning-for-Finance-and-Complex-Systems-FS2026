import numpy as np
import pandas as pd
import math

class MomentumStrategy:
    def __init__(self, config):
        self.config = config
        self.band_mult = config['backtesting']['band_multiplier']
        self.band_simplified = config['backtesting']['band_simplified']
        self.trade_freq = config['backtesting']['trade_frequency']
        self.sizing_type = config['backtesting']['position_sizing_type']
        self.target_vol = config['backtesting']['target_volatility']
        self.max_leverage = config['backtesting']['max_leverage']

    def calculate_signals(self, current_day_data, prev_day_data, prev_close_adjusted):
        """Calculate trading signals based on price bands and VWAP"""
        open_price = current_day_data['open'].iloc[0]
        current_close_prices = current_day_data['close']
        vwap = current_day_data['vwap']
        sigma_open = current_day_data['sigma_open']

        # Calculate price bands
        UB = max(open_price, prev_close_adjusted) * (1 + self.band_mult * sigma_open)
        LB = min(open_price, prev_close_adjusted) * (1 - self.band_mult * sigma_open)

        # Determine trading signals
        signals = np.zeros_like(current_close_prices)
        signals[(current_close_prices > UB) & (current_close_prices > vwap)] = 1
        signals[(current_close_prices < LB) & (current_close_prices < vwap)] = -1

        return signals

    def calculate_position_size(self, previous_aum, open_price, spx_vol):
        """Calculate position size based on volatility targeting or full notional"""
        if self.sizing_type == "vol_target":
            if math.isnan(spx_vol):
                shares = round(previous_aum / open_price * self.max_leverage)
            else:
                shares = round(previous_aum / open_price * min(self.target_vol / spx_vol, self.max_leverage))
        elif self.sizing_type == "full_notional":
            shares = round(previous_aum / open_price)
        return shares

    def apply_trading_frequency(self, signals, min_from_open):
        """Apply trading signals at specified frequency"""
        if self.trade_freq >= len(min_from_open):
            # Daily mode: signal fires once at market open (first bar of the day)
            trade_indices = np.where(min_from_open == min_from_open.min())[0]
        else:
            trade_indices = np.where(min_from_open % self.trade_freq == 0)[0]
        exposure = np.full(len(signals), np.nan)
        exposure[trade_indices] = signals[trade_indices]

        # Custom forward-fill that stops at zeros
        last_valid = np.nan
        filled_values = []
        for value in exposure:
            if not np.isnan(value):
                last_valid = value
            if last_valid == 0:
                last_valid = np.nan
            filled_values.append(last_valid)

        # Convert to numpy array and shift by 1, then fill NaN with 0
        filled_values = np.array(filled_values)
        filled_values = np.roll(filled_values, 1)
        filled_values[0] = 0  # First element becomes 0 after shift
        filled_values = np.nan_to_num(filled_values, 0)
        
        return filled_values 