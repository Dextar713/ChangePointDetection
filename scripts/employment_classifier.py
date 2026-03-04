from prepare_data import prepare_employment_data, prepare_jobs_data
from online_detector import FastOnlineDetector
from opt_seg import OptSegmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def classify_employment(data: pd.DataFrame, method:str='offline'):
    min_dist = 12
    horizon_size = 150
    sectors = data.columns
    trends = []
    slopes = []
    for sector in sectors:
        if method == 'offline':
            detector = OptSegmentation(model='linear', min_dist=min_dist)
            cps = detector.fit_predict(data[sector])
        else:
            cps = []
            detector = FastOnlineDetector(cost_type='linear', min_dist=min_dist, horizon_size=horizon_size)
            for val in data[sector].values:
                is_detected = detector.update(val)
                if is_detected:
                    cps.append(detector.last_change_point+detector.offset)
                    # detection_times.append(detector.n_samples-1)
        n = len(data[sector])
        last_cp = cps[-1]
        cp_offset = 1
        last_segment = data[sector].values[last_cp+cp_offset:n] 
        # last_segment -= last_segment.mean()
        print(last_segment[-3:])
        x_vals = np.arange(n - last_cp - cp_offset)
        # x_vals = np.arange(last_cp+cp_offset, n)
        # slope = np.polynomial.polynomial.Polynomial.fit(x_vals, last_segment, 1).coef[-1]
        slope, intercept = np.polyfit(x_vals, last_segment, 1)
        relative_change = 100*slope/intercept
        slopes.append(relative_change)
        # plt.plot(x_vals, last_segment)
        # plt.plot(x_vals, slope*x_vals+intercept)
        # plt.show()
        up_threshold = 0.1
        down_threshold = -0.1
        if relative_change < down_threshold:
            trends.append('down')
        elif relative_change > up_threshold:
            trends.append('up')
        else:
            trends.append('flat')
    return slopes, trends

def test_employment_classifier(method:str = 'offline'):
    # data = prepare_employment_data()
    data = prepare_jobs_data()
    slopes, trends = classify_employment(data, method=method)
    sectors = data.columns
    for i in range(len(sectors)):
        print(f'Sector: {sectors[i]}, trend: {trends[i]}, slope: {slopes[i]}')

if __name__ == '__main__':
    test_employment_classifier(method='online')


