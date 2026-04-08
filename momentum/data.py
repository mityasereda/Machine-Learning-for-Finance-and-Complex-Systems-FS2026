import requests
import yaml
import pytz
import pandas as pd
import time
from datetime import datetime
import numpy as np
# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) 

API_KEY = config["data"]["api_key"]
BASE_URL = config["data"]["base_url"] 
ENFORCE_RATE_LIMIT = config["data"]["enforce_rate_limit"]


def fetch_polygon_data(ticker, start_date, end_date, period, enforce_rate_limit=ENFORCE_RATE_LIMIT):
    """Fetch stock data from Polygon.io based on the given period (minute or day).
       enforce_rate_limit: Set to True to enforce rate limits (suitable for free tiers), False for paid tiers with minimal or no rate limits.
    """
    multiplier = '1'
    timespan = period
    limit = '50000'  # Maximum entries per request
    eastern = pytz.timezone('America/New_York')  # Eastern Time Zone
    
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
            first_request_time = time.time()  # Reset the timer after the wait

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
    """ Fetches dividend data from Polygon.io for a specified stock ticker. """
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

    # Check if the API returned empty data
    if len(spy_intra_data) == 0 or len(spy_daily_data) == 0:
        print(f"Warning: API returned empty data for {ticker} from {from_date} to {until_date}")
        return pd.DataFrame(), pd.DataFrame()

    # Load the intraday data into a DataFrame and set the datetime column as the index.
    df = pd.DataFrame(spy_intra_data)
    df['day'] = pd.to_datetime(df['caldt']).dt.date  # Extract the date part from the datetime for daily analysis.
    df.set_index('caldt', inplace=True)  # Setting the datetime as the index for easier time series manipulation.

    # Group the DataFrame by the 'day' column to facilitate operations that need daily aggregation.
    daily_groups = df.groupby('day')

    # Extract unique days from the dataset to iterate through each day for processing.
    all_days = df['day'].unique()

    # Initialize new columns to store calculated metrics, starting with NaN for absence of initial values.
    df['move_open'] = np.nan  # To record the absolute daily change from the open price
    df['vwap'] = np.nan       # To calculate the Volume Weighted Average Price.
    df['spy_dvol'] = np.nan   # To record SPY's daily volatility.

    # Create a series to hold computed daily returns for SPY, initialized with NaN.
    spy_ret = pd.Series(index=all_days, dtype=float)

    # Iterate through each day to calculate metrics.
    for d in range(1, len(all_days)):
        current_day = all_days[d]
        prev_day = all_days[d - 1]

        # Access the data for the current and previous days using their groups.
        current_day_data = daily_groups.get_group(current_day)
        prev_day_data = daily_groups.get_group(prev_day)

        # Calculate the average of high, low, and close prices.
        hlc = (current_day_data['high'] + current_day_data['low'] + current_day_data['close']) / 3

        # Compute volume-weighted metrics for VWAP calculation.
        vol_x_hlc = current_day_data['volume'] * hlc
        cum_vol_x_hlc = vol_x_hlc.cumsum()  # Cumulative sum for VWAP calculation.
        cum_volume = current_day_data['volume'].cumsum()

        # Assign the calculated VWAP to the corresponding index in the DataFrame.
        df.loc[current_day_data.index, 'vwap'] = cum_vol_x_hlc / cum_volume

        # Calculate the absolute percentage change from the day's opening price.
        open_price = current_day_data['open'].iloc[0]
        df.loc[current_day_data.index, 'move_open'] = (current_day_data['close'] / open_price - 1).abs()

        # Compute the daily return for SPY using the closing prices from the current and previous day.
        spy_ret.loc[current_day] = current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1] - 1

        # Calculate the 15-day rolling volatility, starting calculation after accumulating 15 days of data.
        if d > 14:
            df.loc[current_day_data.index, 'spy_dvol'] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)

    # Calculate the minutes from market open and determine the minute of the day for each timestamp.
    df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
    df['minute_of_day'] = df['min_from_open'].round().astype(int)

    # Group data by 'minute_of_day' for minute-level calculations.
    minute_groups = df.groupby('minute_of_day')

    # Calculate rolling mean and delayed sigma for each minute of the trading day.
    df['move_open_rolling_mean'] = minute_groups['move_open'].transform(lambda x: x.rolling(window=14, min_periods=13).mean())
    df['sigma_open'] = minute_groups['move_open_rolling_mean'].transform(lambda x: x.shift(1))

    # Handle dividend data safely - check if dividends DataFrame is not empty
    if not dividends.empty:
        dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
        df = df.merge(dividends[['day', 'dividend']], on='day', how='left')
    else:
        # If no dividend data, just add a column of zeros
        df['dividend'] = 0
    
    df['dividend'] = df['dividend'].fillna(0)  # Fill missing dividend data with 0.

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

    # save data to csv
    data_intra.to_csv('data_intra.csv', index=False)
    data_daily.to_csv('data_daily.csv', index=False)
 
