import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series | np.ndarray):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")