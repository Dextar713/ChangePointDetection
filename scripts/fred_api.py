import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import os
from prepare_data import load_data, prepare_data

def load_data():
    fred_api_key = "cd54e1d9baa5154d79ca7ac54ab064e0"
    fred = Fred(api_key=fred_api_key)

    sectors = ['TotalNonfarm', 'healthcare', 'architectural_engineering', 'construction', 'PROFESSIONAL_SCIENCE_TECHNICAL_SERVICES',
               'ComputerElectronicProductManufacturing', 'ComputerSystemsDesign', 'FinanceInsurance', 'LeisureHospitality']
    series_codes = ['PAYEMS', 'CES6562000101', 'CES6054130001', 'USCONS', 'CES6054000001',
                    'CES3133400001', 'CES6054150001', 'CES5552000001', 'USLAH']
    for sector, series_code in zip(sectors, series_codes):
        file_name = f'employment_{sector}.csv'
        path_to_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', file_name)
        if os.path.exists(path_to_csv):
            print(f"{file_name} already exists. Skipping download.")
            data = pd.read_csv(path_to_csv, index_col=0, parse_dates=True, header=None)
            continue
        print("Fetching JOLTS data...")
        data = fred.get_series(series_code)
        # jolts_data = pd.read_csv(path_to_csv, index_col=0, parse_dates=True, header=None)
        data = data.dropna() 
        data.to_csv(path_to_csv)


if __name__ == "__main__":
    load_data()