import datetime
import pandas as pd
import numpy as np
from prepare_data import load_data, prepare_data
from online_detector import NaiveOnlineDetector

# Load the same data used in your test
data = load_data(ticker='AMD', start_date=datetime.datetime(2020, 1, 1), interval='1d')
data = prepare_data(data)
target_col = 'smooth_log_close'
cost_type = 'linear'
data = data[-200:]
time_series = data[target_col].values
actual_prices = np.exp(time_series)  # Convert log prices back to actual prices

print(f"Actual price range: [${actual_prices.min():.2f}, ${actual_prices.max():.2f}]")
print(f"Log price range: [{time_series.min():.6f}, {time_series.max():.6f}]\n")

# Run the detector
model = NaiveOnlineDetector(cost_type, 'opt', min_dist=10)
n = len(time_series)

for i in range(n):
    is_detected = model.update(time_series[i])
    if is_detected:
        start = model.last_change_point
        end = i - model.offset
        length = end - start
        
        # Get slope on log scale
        slope_log, _ = model.model.cost_computer.get_slope_intercept(start=start, end=end)
        angle_log = np.degrees(np.arctan(slope_log))
        
        # Get prices
        price_start = actual_prices[start]
        price_end = actual_prices[end-1]
        log_price_start = time_series[start]
        
        # Convert to actual price scale
        # Simple approximation: slope_actual ≈ slope_log * price_start
        slope_actual_simple = slope_log * price_start
        angle_actual_simple = np.degrees(np.arctan(slope_actual_simple))
        
        # Exact formula: slope_actual = price_start * (e^(slope_log * length) - 1) / length
        slope_actual_exact = price_start * (np.exp(slope_log * length) - 1) / length
        angle_actual_exact = np.degrees(np.arctan(slope_actual_exact))
        
        # Also calculate directly from actual prices
        price_change = price_end - price_start
        slope_direct = price_change / length
        angle_direct = np.degrees(np.arctan(slope_direct))
        
        print(f"Segment [{start:3d}, {end:3d}] - Length: {length} points")
        print(f"  Log-scale slope: {slope_log:.6f}, angle: {angle_log:.2f}°")
        print(f"  Actual-scale slope (simple approx):  {slope_actual_simple:.6f}, angle: {angle_actual_simple:.2f}°")
        print(f"  Actual-scale slope (exact formula):  {slope_actual_exact:.6f}, angle: {angle_actual_exact:.2f}°")
        print(f"  Direct from prices (${price_start:.2f} → ${price_end:.2f}): {slope_direct:.6f}, angle: {angle_direct:.2f}°")
        print()
