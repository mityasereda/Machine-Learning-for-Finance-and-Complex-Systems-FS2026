import requests
import yaml
import pytz
import pandas as pd
import time
from datetime import datetime
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) 

API_KEY = config["data"]["api_key"]
BASE_URL = config["data"]["base_url"] 
ENFORCE_RATE_LIMIT = config["data"]["enforce_rate_limit"]


def fetch_polygon_data(ticker, start_date, end_date, period, enforce_rate_limit=ENFORCE_RATE_LIMIT):
    multiplier = '1'
    timespan = period
    limit = '50000'
    eastern = pytz.timezone('America/New_York')
    
    url = f'{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=false&sort=asc&limit={limit}&apiKey={API_KEY}'
    
    data_list = []
    request_count = 0
    first_request_time = None
    
    while True:
        if enforce_rate_limit and request_count == 5:
            elapsed_time = time.time() - first_request_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                print(f"API rate limit reached. Waiting {wait_time:.2f} seconds before next request.")
                time.sleep(wait_time)
            request_count = 0
            first_request_time = time.time()

        if first_request_time is None and enforce_rate_limit:
            first_request_time = time.time()

        response = requests.get(url)
        if response.status_code != 200:
            print(response)
            error_message = response.json().get('error', 'No specific error message provided')
            print(f"Error fetching data: {error_message}")
            break

        data = response.json()
        request_count += 1
        
        results_count = len(data.get('results', []))
        print(f"Fetched {results_count} entries from API.")
        
        if 'results' in data:
            for entry in data['results']:
                utc_time = datetime.fromtimestamp(entry['t'] / 1000, pytz.utc)
                eastern_time = utc_time.astimezone(eastern)
                
                data_entry = {
                    'volume': entry['v'],
                    'open': entry['o'],
                    'high': entry['h'],
                    'low': entry['l'],
                    'close': entry['c'],
                    'caldt': eastern_time.replace(tzinfo=None) 
                }
                
                if period == 'minute':
                    if eastern_time.time() >= datetime.strptime('09:30', '%H:%M').time() and eastern_time.time() <= datetime.strptime('15:59', '%H:%M').time():
                        data_list.append(data_entry)
                else:
                    data_list.append(data_entry)
        
        if 'next_url' in data and data['next_url']:
            url = data['next_url'] + '&apiKey=' + API_KEY
        else:
            break
    
    df = pd.DataFrame(data_list)
    print("Data fetching complete.")
    return df


def fetch_polygon_dividends(ticker):
    url = f'{BASE_URL}/v3/reference/dividends?ticker={ticker}&limit=1000&apiKey={API_KEY}'
    
    dividends_list = []
    while True:
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
            for entry in data['results']:
                dividends_list.append({
                    'caldt': datetime.strptime(entry['ex_dividend_date'], '%Y-%m-%d'),
                    'dividend': entry['cash_amount']
                })
        
        if 'next_url' in data and data['next_url']:
            url = data['next_url'] + '&apiKey=' + API_KEY
        else:
            break
    
    return pd.DataFrame(dividends_list)

def get_data(ticker, from_date, until_date):
    spy_intra_data = fetch_polygon_data(ticker, from_date, until_date, 'minute')
    spy_daily_data = fetch_polygon_data(ticker, from_date, until_date, 'day')
    dividends      = fetch_polygon_dividends(ticker)

    if len(spy_intra_data) == 0 or len(spy_daily_data) == 0:
        print(f"Warning: API returned empty data for {ticker} from {from_date} to {until_date}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(spy_intra_data)
    df['day'] = pd.to_datetime(df['caldt']).dt.date
    df.set_index('caldt', inplace=True)

    daily_groups = df.groupby('day')

    all_days = df['day'].unique()

    df['move_open'] = np.nan
    df['vwap'] = np.nan
    df['spy_dvol'] = np.nan

    spy_ret = pd.Series(index=all_days, dtype=float)

    for d in range(1, len(all_days)):
        current_day = all_days[d]
        prev_day = all_days[d - 1]

        current_day_data = daily_groups.get_group(current_day)
        prev_day_data = daily_groups.get_group(prev_day)

        hlc = (current_day_data['high'] + current_day_data['low'] + current_day_data['close']) / 3

        vol_x_hlc = current_day_data['volume'] * hlc
        cum_vol_x_hlc = vol_x_hlc.cumsum()
        cum_volume = current_day_data['volume'].cumsum()

        df.loc[current_day_data.index, 'vwap'] = cum_vol_x_hlc / cum_volume

        open_price = current_day_data['open'].iloc[0]
        df.loc[current_day_data.index, 'move_open'] = (current_day_data['close'] / open_price - 1).abs()

        spy_ret.loc[current_day] = current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1] - 1

        if d > 14:
            df.loc[current_day_data.index, 'spy_dvol'] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)

    df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
    df['minute_of_day'] = df['min_from_open'].round().astype(int)

    minute_groups = df.groupby('minute_of_day')

    df['move_open_rolling_mean'] = minute_groups['move_open'].transform(lambda x: x.rolling(window=14, min_periods=13).mean())
    df['sigma_open'] = minute_groups['move_open_rolling_mean'].transform(lambda x: x.shift(1))

    if not dividends.empty:
        dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
        df = df.merge(dividends[['day', 'dividend']], on='day', how='left')
    else:
        df['dividend'] = 0
    
    df['dividend'] = df['dividend'].fillna(0)

    df_intra = df
    df_daily = spy_daily_data
    return df_intra, df_daily

if __name__ == "__main__":
    ticker = 'SPY'
    from_date = '2022-05-09'
    until_date = '2024-04-22'

    data_intra, data_daily = get_data(ticker, from_date, until_date)
    print(data_intra.head())
    print(data_daily.head())

    data_intra.to_csv('data_intra.csv', index=False)
    data_daily.to_csv('data_daily.csv', index=False)
 
