import datetime
import os

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from bin_seg import BinarySegmentation
from opt_seg import OptSegmentation
from online_detector import NaiveOnlineDetector, FastOnlineDetector
from prepare_data import load_data, prepare_data
from stats_tests import check_stationarity
import warnings
warnings.filterwarnings("ignore")

def plot_elbow_gains(gains: list[float]) -> None:
    plt.title("ELBOW GAINS")
    plt.plot(sorted(gains)[::-1])
    plt.savefig("../visuals/elbow_gains.png")
    # plt.show()

def generate_series(cost_type: str = 'normal', n_segments=3, noise_std=1.0) -> np.ndarray:
    np.random.seed(101)
    if cost_type == 'normal':
        r1 = np.random.normal(0, 0.5, 200)
        r2 = np.random.normal(-0.2, 2.0, 100)
        r3 = np.random.normal(0.05, 1.0, 150)
        time_series = np.concatenate([r1, r2, r3])
        return time_series
    if cost_type == 'linear':
        slopes = [0.7, -0.8, 1.3, 0.01, -1.5, 0.4, -0.33]
        # slopes = np.random.uniform(low=-7.7, high=7.5, size=n_segments)
        last_val = 0
        time_series = []
        for slope in slopes:
            cur_len = np.random.randint(low=20, high=50)
            x = np.arange(cur_len)
            trend = last_val + slope * x
            cur_segment = trend + np.random.normal(0, noise_std, size=cur_len)
            last_val = cur_segment[-1]
            time_series.append(cur_segment)

        return np.concatenate(time_series)
    if cost_type == 'l2':
        means = np.random.randint(low=-20, high=25, size=n_segments)
        time_series = []
        for mu in means:
            cur_len = np.random.randint(low=20, high=50)
            cur_seg = np.random.normal(mu, noise_std, size=cur_len)
            time_series.append(cur_seg)
        return np.concatenate(time_series)
    if cost_type == 'mean_var':
        regimes = [
            (0, 1.0, 50),
            (10, 1.0, 30),
            (10, 3.7, 40),
            (-5, 0.7, 30)
        ]
        time_series = []
        for (mu, sigma, sz) in regimes:
            cur_seg = np.random.normal(mu, sigma, size=sz)
            time_series.append(cur_seg)
        return np.concatenate(time_series)

    raise ValueError("Invalid model")


def visualize_detections(time_series: np.ndarray, time_axes, change_points: np.ndarray, show_std: bool = False) -> None:
    is_x_date = isinstance(time_axes, pd.DatetimeIndex) or isinstance(time_axes[0], (datetime.datetime, pd.Timestamp))
    if is_x_date:
        print(f"Detected {len(change_points)} Points:    {time_axes[change_points]}")
    else:
        print(f"Detected {len(change_points)} Points:    {change_points}")

    plt.figure(figsize=(12, 6))
    plt.plot(time_axes, time_series, color='gray', alpha=0.6, label='Time series')
    plt.title("Detected Change Points ")

    for point in change_points:
        if is_x_date:
            x_val = time_axes[point-1]
        else:
            x_val = point
        plt.axvline(x=x_val, color='red', linestyle='--', linewidth=1, label='Change Point')

    if show_std:
        start = 0
        for point in change_points + [len(time_series)]:
            segment = time_series[start:point]
            std_dev = np.std(segment)
            mu = np.mean(segment)
            if is_x_date:
                plt.fill_between([time_axes[start], time_axes[point-1]], mu-std_dev * 2, mu+std_dev * 2, color='red', alpha=0.1)
            else:
                plt.fill_between([start, point], mu-std_dev * 2, mu+std_dev * 2, color='red', alpha=0.1)
            start = point

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    save_path = os.path.join(os.path.dirname(__file__), "../visuals", "detected_change_points.png")
    plt.savefig(save_path)
    plt.show()


def test_detection(time_series: np.ndarray | None = None, time_axes = None, cost_type:str='normal', 
                   model_type:str='opt', method:str = 'offline') -> None:
    if time_series is None:
        time_series = generate_series(cost_type, n_segments=7)
    if time_axes is None:
        time_axes = np.arange(0, time_series.shape[0])
    
    if method == 'offline':
        if model_type == 'opt':
            model = OptSegmentation(model=cost_type, min_dist=5)
        elif model_type == 'binseg':
            model = BinarySegmentation(model=cost_type, min_dist=5)
            # plot_elbow_gains(gains)
        else:
            raise ValueError("Invalid model type")
        change_points = model.fit_predict(time_series)
    else:
        change_points = []
        model = NaiveOnlineDetector(cost_type, model_type, min_dist=10)
        # model = FastOnlineDetector(cost_type=cost_type, min_dist=10, horizon_size=90)
        n = len(time_series)
        for i in range(n):
            is_detected = model.update(time_series[i])
            if is_detected:
                # print(f'Changepoint {model.last_change_point+model.offset} detected at point {i}')
                change_points.append(model.last_change_point + model.offset)
                # change_points.append(i)
                # print(model.last_change_point)
                # print(i-model.offset)
               
                # change_points.append(i)

    show_std = cost_type == 'normal'
    visualize_detections(time_series, time_axes, change_points, show_std)
    

def run_tests():
    data = load_data(ticker='AMD', start_date=datetime.datetime(2020, 1, 1), interval='1d')
    data = prepare_data(data)
    # target_col = 'Close'
    target_col = 'Close'
    # cost_type = 'mean_var'
    cost_type = 'linear'
    data = data[-200:]
    # check_stationarity(np.diff(data['smooth_log_close']))
    test_detection(time_series=data[target_col].values, time_axes=data.index, cost_type=cost_type, model_type='opt', method='online')
    # test_detection(None, None, 'linear', 'opt', method='online')

if __name__ == '__main__':
    run_tests()