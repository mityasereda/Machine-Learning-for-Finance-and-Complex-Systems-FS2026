import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent

class MarketImpactCalculator:
    def __init__(self, api_key, base_url="https://api.polygon.io/v3/trades", 
                 market_impact_window=15, impact_threshold=0.01, max_impact=0.10,
                 fallback_model=True, api_retry_limit=3, cache_results=True,
                 provider="polygon", wrds_root_dir=None,
                 wrds_trades_subdir="raw/taq_trades", wrds_file_format="parquet"):
        self.api_key = api_key
        self.base_url = base_url
        self.market_impact_window = market_impact_window
        self.impact_threshold = impact_threshold
        self.max_impact = max_impact
        self.fallback_model = fallback_model
        self.api_retry_limit = api_retry_limit
        self.cache_results = cache_results
        self._cache = {}
        self.provider = provider.lower()
        if wrds_root_dir:
            wrds_root = Path(wrds_root_dir).expanduser()
            if not wrds_root.is_absolute():
                wrds_root = (MODULE_DIR / wrds_root).resolve()
            self.wrds_root_dir = wrds_root
        else:
            self.wrds_root_dir = None
        self.wrds_trades_subdir = wrds_trades_subdir
        self.wrds_file_format = wrds_file_format.lower()

    def _load_wrds_trade_file(self, ticker, timestamp):
        if self.wrds_root_dir is None:
            return None

        trade_date = pd.to_datetime(timestamp).date().isoformat()
        path = self.wrds_root_dir / self.wrds_trades_subdir / ticker / f"{trade_date}.{self.wrds_file_format}"
        if not path.exists():
            return None

        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path, parse_dates=['timestamp'])
        else:
            raise ValueError(f"Unsupported WRDS trade file format: {path.suffix}")

        if 'timestamp' not in df.columns or 'price' not in df.columns or 'size' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'price', 'size'])
        df = df[(df['price'] > 0) & (df['size'] > 0)].copy()
        if df.empty:
            return None
        if 'value' not in df.columns:
            df['value'] = df['price'] * df['size']
        return df

    def fetch_trades(self, ticker, timestamp, limit=50000):
        cache_key = f"{ticker}_{timestamp}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        
        window_start = timestamp - pd.Timedelta(minutes=self.market_impact_window)
        window_end = timestamp

        if self.provider == 'wrds':
            trades_df = self._load_wrds_trade_file(ticker, timestamp)
            if trades_df is None or trades_df.empty:
                return None

            trades_df = trades_df[
                (trades_df['timestamp'] >= window_start) &
                (trades_df['timestamp'] <= window_end)
            ].copy()
            if trades_df.empty:
                return None

            if self.cache_results:
                self._cache[cache_key] = trades_df
            return trades_df

        formatted_start = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        formatted_end = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        url = f"{self.base_url}/{ticker}"
        params = {
            "timestamp.gte": formatted_start,
            "timestamp.lte": formatted_end,
            "limit": limit,
            "apiKey": self.api_key,
            "sort": "timestamp"
        }
        
        for attempt in range(self.api_retry_limit):
            try:
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    print(f"Error response: {response.text}")
                    if attempt < self.api_retry_limit - 1:
                        continue
                    return None
                    
                data = response.json()
                if "results" not in data or not data["results"]:
                    return None
                    
                trades = data["results"]
                df = pd.DataFrame(trades)
                
                if 'sip_timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
                elif 'participant_timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
                else:
                    return None
                
                if 'price' not in df.columns or 'size' not in df.columns:
                    return None
                
                df['value'] = df['price'] * df['size']
                
                if self.cache_results:
                    self._cache[cache_key] = df
                
                return df
                
            except Exception as e:
                print(f"Error fetching trades data (attempt {attempt + 1}/{self.api_retry_limit}): {str(e)}")
                if attempt < self.api_retry_limit - 1:
                    continue
                return None

    def calculate_volume_profile(self, trades_df, num_bins=20):
        if trades_df is None or trades_df.empty:
            return None, None

        trades_df = trades_df.copy()
        trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
        trades_df['size'] = pd.to_numeric(trades_df['size'], errors='coerce')
        trades_df = trades_df.dropna(subset=['price', 'size'])
        trades_df = trades_df[(trades_df['price'] > 0) & (trades_df['size'] > 0)]
        if trades_df.empty:
            return None, None

        price_min = float(trades_df['price'].min())
        price_max = float(trades_df['price'].max())
        if not np.isfinite(price_min) or not np.isfinite(price_max):
            return None, None
        if price_min == price_max:
            price_min *= 0.999
            price_max *= 1.001
        price_range = np.linspace(price_min * 0.99, price_max * 1.01, num_bins + 1)
        
        trades_df['price_bin'] = pd.cut(trades_df['price'], bins=price_range)
        volume_profile = trades_df.groupby('price_bin', observed=False)['size'].sum()
        volume_profile = pd.to_numeric(volume_profile, errors='coerce').dropna()
        if volume_profile.empty:
            return None, None
        
        return price_range, volume_profile

    def calculate_market_impact(self, ticker, price, volume, side, timestamp):
        if not isinstance(price, (int, float)) or np.isnan(price) or price <= 0:
            return price
            
        if not isinstance(volume, (int, float)) or volume <= 0:
            return price

        trades_df = self.fetch_trades(ticker, timestamp)
        if trades_df is None or trades_df.empty:
            if self.fallback_model:
                return self.calculate_fallback_impact(price, volume, side)
            return price
            
        try:
            trades_df = trades_df.copy()
            trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
            trades_df['size'] = pd.to_numeric(trades_df['size'], errors='coerce')
            trades_df = trades_df.dropna(subset=['price', 'size'])
            trades_df = trades_df[(trades_df['price'] > 0) & (trades_df['size'] > 0)]
            if trades_df.empty:
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side)
                return price
                
            avg_trade_size = float(trades_df['size'].mean())
            total_volume = float(trades_df['size'].sum())
            mean_price = float(trades_df['price'].mean())
            price_std = float(trades_df['price'].std())
            if not np.isfinite(avg_trade_size) or avg_trade_size <= 0 or not np.isfinite(total_volume) or total_volume <= 0:
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side)
                return price
            if not np.isfinite(mean_price) or mean_price <= 0 or not np.isfinite(price_std):
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side)
                return price
            price_volatility = price_std / mean_price
            
            price_range, volume_profile = self.calculate_volume_profile(trades_df)
            if price_range is None or volume_profile is None or volume_profile.empty:
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side)
                return price
                
            price_bin = pd.cut([price], bins=price_range)[0]
            volume_profile_mean = pd.to_numeric(pd.Series([volume_profile.mean()]), errors='coerce').iloc[0]
            if pd.isna(price_bin) or not np.isfinite(volume_profile_mean) or volume_profile_mean <= 0:
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side)
                return price

            volume_in_bin = volume_profile.get(price_bin, np.nan)
            if pd.isna(volume_in_bin):
                volume_in_bin = volume_profile_mean
            volume_in_bin = pd.to_numeric(pd.Series([volume_in_bin]), errors='coerce').iloc[0]
            if not np.isfinite(volume_in_bin) or volume_in_bin <= 0:
                volume_in_bin = volume_profile_mean

            volume_factor = volume_in_bin / volume_profile_mean
            
            volume_ratio = volume / total_volume
            base_impact = volume_ratio * price_volatility * 100
            
            size_premium = 0
            if volume > avg_trade_size:
                size_ratio = volume / avg_trade_size
                size_premium = np.log10(size_ratio) * 0.002
            
            # Keep impact adverse-only: high local liquidity can reduce slippage,
            # but it should never flip the sign and improve execution.
            liquidity_factor = max(0.0, 1 + (1 - volume_factor) * 0.8)
            impact = (base_impact + size_premium) * liquidity_factor
            
            if volume > total_volume * self.impact_threshold:
                volume_threshold = total_volume * self.impact_threshold
                excess_ratio = (volume - volume_threshold) / volume_threshold
                non_linear_impact = excess_ratio * 0.01
                impact += non_linear_impact
            
            impact = max(0.0, min(impact, self.max_impact))
            impacted_price = price * (1 + impact) if side == 'buy' else price * (1 - impact)
            
            return impacted_price if not np.isnan(impacted_price) and impacted_price > 0 else price
            
        except Exception as e:
            print(f"Error calculating market impact: {str(e)}")
            if self.fallback_model:
                return self.calculate_fallback_impact(price, volume, side)
            return price

    def calculate_fallback_impact(self, price, volume, side):
        try:
            if volume <= 1000:
                daily_volume = 1000000
                impact_factor = 0.0002
            elif volume <= 5000:
                daily_volume = 5000000
                impact_factor = 0.0004
            elif volume <= 10000:
                daily_volume = 10000000
                impact_factor = 0.001
            else:
                daily_volume = 20000000
                impact_factor = 0.002
                
            volume_ratio = volume / daily_volume
            base_impact = volume_ratio * impact_factor * 200
            
            size_premium = 0
            if volume > 10000:
                size_ratio = volume / 10000
                size_premium = np.log10(size_ratio) * 0.002
            
            impact = min(base_impact + size_premium, 0.10)
            impacted_price = price * (1 + impact) if side == 'buy' else price * (1 - impact)
            
            return impacted_price if not np.isnan(impacted_price) and impacted_price > 0 else price
            
        except Exception as e:
            print(f"Error in fallback impact calculation: {str(e)}")
            return price 
