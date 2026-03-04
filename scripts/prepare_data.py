import os
from datetime import datetime, timedelta
from fredapi import Fred
import numpy as np
import yfinance as yf
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
# from dotenv import load_dotenv

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
            end_date = datetime.now().date() - timedelta(days=1)
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

def prepare_jobs_data() -> pd.DataFrame:
    # load_dotenv()
    # fred_api_key = os.getenv('fred_api_key')
    fred_api_key = "cd54e1d9baa5154d79ca7ac54ab064e0"
    fred = Fred(api_key=fred_api_key)
    sectors = ["Total_jobs", "Finance", "Construction", "Education_Healthcare", "Leisure_Hospitality", "Manufacturing",
                "Information", 'Professional_Business_Services', 'Trade_Transportation_Utilities', 'Government', 
                "Real_Estate_Rental_Leasing", 'Mining_Logging', 'Other_Services']
    sector_codes = ['JTSJOL', 'JTU5200JOL', 'JTS2300JOL', 'JTS6000JOL', 'JTS7000JOL', 'JTS3000JOL',
                     'JTU5100JOL', 'JTS540099JOL', 'JTS4000JOL', 'JTS9000JOL',
                     'JTU5300JOL', 'JTU110099JOL', 'JTU8100JOL']
    jobs_data = pd.DataFrame()
    for i in range(len(sectors)):
        path_to_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f'{sectors[i]}_jolts_data.csv')
        if not os.path.exists(path_to_csv):
            print(f"Fetching {sectors[i]} JOLTS data...")
            sector_data = fred.get_series(sector_codes[i])
            sector_data.to_csv(path_to_csv, header=False)
        else:
            print(f"{sectors[i]} JOLTS data already exists. Skipping download.")
            sector_data = pd.read_csv(path_to_csv, index_col=0, parse_dates=True, header=None)
        sector_data = sector_data.dropna()
        if sector_codes[i].startswith("JTU"):
            # sector_data.plot()
            sector_data = sector_data.asfreq("MS")
            # sector_data = sector_data.asfreq('m')
            decomposed_series = seasonal_decompose(sector_data, model='additive', period=12)
            sector_data = decomposed_series.trend

        jobs_data = pd.concat([jobs_data, sector_data], axis=1)
    jobs_data.columns = sectors
    jobs_data.index.name = 'Date'
    jobs_data = jobs_data.dropna()
    # print(jobs_data.head())
    # sum1 = np.sum(jobs_data.iloc[0]) - 2*jobs_data['Total_jobs'].iloc[0]
    # print(sum1)
    return jobs_data

def prepare_employment_data() -> pd.DataFrame:
    # load_dotenv()
    # fred_api_key = os.getenv('fred_api_key')
    fred_api_key = "cd54e1d9baa5154d79ca7ac54ab064e0"
    fred = Fred(api_key=fred_api_key)
    sectors = ['TotalNonfarm', 'healthcare', 'architectural_engineering', 'construction', 'PROFESSIONAL_SCIENCE_TECHNICAL_SERVICES',
               'ComputerElectronicProductManufacturing', 'ComputerSystemsDesign', 'FinanceInsurance', 'LeisureHospitality']
    series_codes = ['PAYEMS', 'CES6562000101', 'CES6054130001', 'USCONS', 'CES6054000001',
                    'CES3133400001', 'CES6054150001', 'CES5552000001', 'USLAH']
    employment_data = pd.DataFrame()
    for i in range(len(sectors)):
        path_to_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f'employment_{sectors[i]}.csv')
        if not os.path.exists(path_to_csv):
            print(f"Fetching {sectors[i]} Employment data...")
            sector_data = fred.get_series(series_codes[i])
            sector_data.to_csv(path_to_csv, header=False)
        else:
            print(f"{sectors[i]} Employment data already exists. Skipping download.")
            sector_data = pd.read_csv(path_to_csv, index_col=0, parse_dates=True, header=None)
        sector_data = sector_data.dropna()

        employment_data = pd.concat([employment_data, sector_data], axis=1)
    employment_data.columns = sectors
    employment_data.index.name = 'Date'
    employment_data = employment_data.dropna()[:-1]
    # print(employment_data.head())
    return employment_data