import os
from datetime import datetime

import numpy as np
import yfinance as yf
import pandas as pd


# class DataPrepare:
#     def __init__(self):


def load_data(ticker: str, start_date: datetime, interval: str = '1d', download_fresh = False, end_date=None) -> pd.DataFrame:
    rel_path = f'../data/{ticker}_{start_date.strftime('%Y-%m-%d')}_{interval}.csv'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    print(f"Data path: {data_path}")
    if not download_fresh and os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, header=[0, 1])
    else:
        if end_date is None:
            end_date = datetime.now().date()
        data = yf.download(tickers=ticker, start=start_date, end=end_date, interval=interval)
        data.to_csv(data_path, index=True)
    return data

def log_smooth_prices(prices: np.ndarray, window=5) -> np.ndarray:
    log_prices = np.log(prices)
    smoothed = pd.Series(log_prices).ewm(span=window, adjust=False).mean().values
    return smoothed


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data.columns = data.columns.droplevel(1)
    except Exception as e:
        print(e)
        pass
    # data = data[['Close', 'Volume']]
    print(data.columns)
    data[['Close', 'Volume']] = data[
        ['Close', 'Volume']].astype(float)
    data.index = pd.to_datetime(data.index)
    # data['log_close'] = np.log(data['Close'])
    data['smooth_close'] = data['Close'].ewm(span=5, adjust=False).mean().values
    # data['norm_close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()
    data['log_close'] = np.log(data['Close'])
    data['smooth_log_close'] = log_smooth_prices(data['Close'], window=3)
    data['pct_change'] = data['Close'].pct_change()
    data['log_return'] = np.log1p(data["pct_change"])
    data = data.dropna(axis=0, how='any')
    return data
