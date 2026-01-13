import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

N = 1000
polynom2 = np.array([0.1, -1, 3000])
t = np.arange(N)
trend_component = np.column_stack([t**2, t, np.ones_like(t)]) @ polynom2
f1, f2 = 7/N, 30/N
A1, A2 = 2*N, 5*N
seasonal_component = A1*np.sin(2 * np.pi * f1 * t) + A2*np.cos(2 * np.pi * f2 * t)
mu, sigma = 1500, N//2
noise_component = np.random.normal(mu, sigma, size=N)
time_series = trend_component + seasonal_component + noise_component

ts_centered = time_series - np.mean(time_series)
raw_autocorr = np.correlate(ts_centered, ts_centered, mode='full')
pearson_autocorr = acf(time_series, nlags=N)
top3_lags = np.argsort(pearson_autocorr)[-3:]
print('Top 3 lags: ', top3_lags)

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
# ax[0,0].plot(t, trend_component)
# ax[0,0].set_title('Trend')
# ax[0,1].plot(t, seasonal_component)
# ax[0,1].set_title('Seasonal')
# ax[1,0].plot(t, noise_component)
# ax[1,0].set_title('Noise')
# ax[1,1].plot(t, time_series)
# ax[1,1].set_title('Time Series')
p_s = np.arange(10, 100, 10)
train_percentages = np.arange(0.2, 1, 0.1)
best_params = {
    "p": 10,
    "train_percentage": 1,
    "rmse": np.sum(time_series)
}

def train_and_predict(t: np.ndarray, p, percentage) -> np.ndarray:
    train_end = int(N * 0.8)
    train_start = train_end - int(train_percentage * train_end)
    train_len = train_end - train_start
    test_len = N - train_end
    # print(train_start, train_end, train_len, test_len)

    y_train = time_series[train_start:train_end]
    y_test = time_series[train_end:]
    Y_train = np.zeros((train_len - p, p))
    y_target_train = y_train[p:]
    for i in range(train_len - p):
        Y_train[i, :] = y_train[i: i + p][::-1]
    ar_coefficients = np.linalg.lstsq(Y_train, y_target_train)[0]

    Y_test = np.zeros((test_len, p))
    test_history = np.concatenate((y_train[-p:], y_test))
    for i in range(test_len):
        Y_test[i, :] = test_history[i: i + p][::-1]

    y_pred = Y_test @ ar_coefficients
    return y_pred

for p in p_s:
    for train_percentage in train_percentages:
        y_pred = train_and_predict(time_series, p, train_percentage)
        y_test = time_series[int(N * 0.8):]
        rmse = np.sqrt(1/len(y_pred)*np.sum((y_test-y_pred)**2))
        # print(rmse)
        if rmse < best_params["rmse"]:
            best_params["p"] = p
            best_params["train_percentage"] = train_percentage
            best_params["rmse"] = rmse
print("Best params:", best_params)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
lags = np.arange(len(pearson_autocorr))
ax[0].plot(lags, pearson_autocorr)
ax[0].set_title('Autocorrelation Pearson')
train_end = int(N * 0.8)
y_test = time_series[train_end:]
y_pred = train_and_predict(time_series, best_params["p"], best_params["train_percentage"])
test_time_range = np.arange(train_end, N, 1)
ax[1].set_title('True series vs prediction')
ax[1].plot(test_time_range, y_test, label='y_test')
ax[1].plot(test_time_range, y_pred, label='y_pred')
ax[1].legend()
plt.tight_layout()
# plt.savefig('time_series.pdf')
# plt.savefig('AR_predictions.pdf')
plt.savefig('best_predictions.pdf')
plt.show()

# best params: p = 50, 90% of train data, rmse between 550 and 600