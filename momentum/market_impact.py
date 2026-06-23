import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent

# Typical 2022 average daily volumes for the project tickers (shares)
_FALLBACK_ADV = {
    'SPY':  80_000_000,
    'MSFT': 25_000_000,
    'META': 20_000_000,
}
_FALLBACK_SIGMA = 0.002  # conservative intraday volatility for large-cap US equities


class MarketImpactCalculator:
    def __init__(self, api_key, base_url="https://api.polygon.io/v3/trades",
                 market_impact_window=15, eta=0.1, max_impact=0.10,
                 fallback_model=True, api_retry_limit=3, cache_results=True,
                 provider="polygon", wrds_root_dir=None,
                 wrds_trades_subdir="raw/taq_trades", wrds_file_format="parquet"):
        self.api_key = api_key
        self.base_url = base_url
        self.market_impact_window = market_impact_window
        self.eta = eta
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
        if 'value' not in df.columns:
            df['value'] = df['price'] * df['size']
        return df

    def fetch_trades(self, ticker, timestamp, limit=50000):
        """Fetch trades data from Polygon or WRDS for a specific timestamp with a window."""
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

    def calculate_market_impact(self, ticker, price, volume, side, timestamp):
        """Calculate market impact using the Almgren-Chriss (2005) square-root model."""
        if not isinstance(price, (int, float)) or np.isnan(price) or price <= 0:
            return price

        if not isinstance(volume, (int, float)) or volume <= 0:
            return price

        trades_df = self.fetch_trades(ticker, timestamp)
        if trades_df is None or trades_df.empty:
            if self.fallback_model:
                return self.calculate_fallback_impact(price, volume, side, ticker)
            return price

        try:
            if trades_df['price'].isna().any() or trades_df['size'].isna().any():
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side, ticker)
                return price

            total_volume = trades_df['size'].sum()
            if total_volume <= 0:
                if self.fallback_model:
                    return self.calculate_fallback_impact(price, volume, side, ticker)
                return price

            price_volatility = trades_df['price'].std() / trades_df['price'].mean()

            # Scale window volume to an estimated average daily volume
            adv_estimate = total_volume * (390.0 / self.market_impact_window)

            # Almgren-Chriss square-root model: impact = η × σ × √(V_trade / V_ADV)
            participation_rate = volume / adv_estimate
            impact = self.eta * price_volatility * np.sqrt(participation_rate)
            impact = min(impact, self.max_impact)

            impacted_price = price * (1 + impact) if side == 'buy' else price * (1 - impact)
            return impacted_price if not np.isnan(impacted_price) and impacted_price > 0 else price

        except Exception as e:
            print(f"Error calculating market impact: {str(e)}")
            if self.fallback_model:
                return self.calculate_fallback_impact(price, volume, side, ticker)
            return price

    def calculate_fallback_impact(self, price, volume, side, ticker=None):
        """Fallback square-root impact using typical ADV when trade data is unavailable."""
        try:
            adv = _FALLBACK_ADV.get(ticker, 20_000_000)
            participation_rate = volume / adv
            impact = self.eta * _FALLBACK_SIGMA * np.sqrt(participation_rate)
            impact = min(impact, self.max_impact)

            impacted_price = price * (1 + impact) if side == 'buy' else price * (1 - impact)
            return impacted_price if not np.isnan(impacted_price) and impacted_price > 0 else price

        except Exception as e:
            print(f"Error in fallback impact calculation: {str(e)}")
            return price
