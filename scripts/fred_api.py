import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import os
from prepare_data import load_data, prepare_data

fred_api_key = "cd54e1d9baa5154d79ca7ac54ab064e0"
fred = Fred(api_key=fred_api_key)

path_to_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'jolts_data.csv')
print("Fetching JOLTS data...")
# jolts_data = fred.get_series('JTSJOL')
jolts_data = pd.read_csv(path_to_csv, index_col=0, parse_dates=True, header=None)
jolts_data = jolts_data.dropna() 
# jolts_data.to_csv(path_to_csv)

# Print the last 5 months to verify
print("\nRecent Job Openings (in thousands):")
print(jolts_data.tail())

fig, ax = plt.subplots(figsize=(11, 7), nrows=2, ncols=1)
ax[0].plot(jolts_data.index, jolts_data.values, label='JOLTS Total Job Openings', color='blue')
ax[0].set_title('US Job Openings (2000 - Present)')
ax[0].set_ylabel('Job Openings (in thousands)')
ax[0].set_xlabel('Date')
ax[0].grid(True)
ax[0].legend()

ticker = '^GSPC'
start_date = datetime.datetime(2000, 12, 1)
end_date = datetime.datetime(2025, 12, 1)
stock_data = load_data(ticker=ticker, start_date=start_date, end_date=end_date, interval='1mo')
stock_data = prepare_data(stock_data)
ax[1].plot(stock_data.index, stock_data['Close'], label='S&P 500 Close Price', color='green')
ax[1].set_title(f'SP500 Close Price (2000 - Present)')
ax[1].set_ylabel('S&P 500 Close Price (USD)')
ax[1].set_xlabel('Date')
ax[1].grid(True)
ax[1].legend()
plt.tight_layout()
plt.show()