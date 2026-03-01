import numpy as np

def get_gain_threshold(signal: np.ndarray, cost_type: str, horizon_size:int|None = None) -> float:
    n = len(signal)
    if n == 0:
        return 0
    # if horizon_size is not None:
    #     n = max(horizon_size, n)
    if cost_type == 'l2':
        beta = 1.0
        signal_var = np.var(signal)
        return beta * np.log(n) * signal_var
    if cost_type == 'normal':
        beta = 3.0
        return beta * np.log(n)
    if cost_type == 'linear':
        beta = 2.0
        diff_var = np.var(np.diff(signal))
        return (2 * beta) * np.log(n) * diff_var / 2
    if cost_type == 'mean_var':
        beta = 1.0
        return 2 * beta * np.log(n)
    raise ValueError('Unsupported cost type')

class CostComputer:
    def __init__(self, signal: np.ndarray, cost_type: str):
        self.n_samples = len(signal)
        self.cost_type = cost_type
        self.cum_sum = np.concatenate([[0], np.cumsum(signal)])
        self.cum_sum_square = np.concatenate([[0], np.cumsum(np.power(signal, 2))])
        self.min_epsilon = 1e-11

    def _get_stats(self, start: int | np.ndarray, end: int | np.ndarray) -> tuple[float|np.ndarray, float|np.ndarray]:
        cur_len = end - start
        mean = (self.cum_sum[end] - self.cum_sum[start]) / cur_len
        variance = (self.cum_sum_square[end] - self.cum_sum_square[start]) / cur_len - mean * mean
        return mean, variance

    def cost(self, start: int | np.ndarray, end: int | np.ndarray) -> float:
        cur_len = end - start
        mean, variance = self._get_stats(start, end)
        if self.cost_type == 'l2':
            return cur_len * variance
        if self.cost_type == 'normal':
            return cur_len * np.log(variance)
        if self.cost_type == 'mean_var':
            sxx = self.cum_sum_square[end] - self.cum_sum_square[start]
            sx = self.cum_sum[end] - self.cum_sum[start]
            temp = sxx + cur_len * mean**2 - 2*mean*sx
            return cur_len * np.log(variance) + 1/variance * temp
        raise ValueError('Unsupported cost type')

class LinearCostComputer:
    def __init__(self, signal: np.ndarray):
        self.n_samples = len(signal)
        x = np.arange(self.n_samples)
        self.sum_x = np.concatenate([[0], np.cumsum(x)])
        self.sum_y = np.concatenate([[0], np.cumsum(signal)])
        self.sum_xx = np.concatenate([[0], np.cumsum(np.power(x, 2))])
        self.sum_yy = np.concatenate([[0], np.cumsum(np.power(signal, 2))])
        self.sum_xy = np.concatenate([[0], np.cumsum(x * signal)])

    def cost(self, start: int | np.ndarray, end: int | np.ndarray) -> float | np.ndarray:
        n = end - start
        sxx = self.sum_xx[end] - self.sum_xx[start]
        # sxx = self.sum_xx[n]
        syy = self.sum_yy[end] - self.sum_yy[start]
        sx = self.sum_x[end] - self.sum_x[start]
        # sx = self.sum_x[n]
        sy = self.sum_y[end] - self.sum_y[start]
        sxy = self.sum_xy[end] - self.sum_xy[start]
        # sxy = (self.sum_xy[end] - self.sum_xy[start]) - start * sy
        S_xx = sxx - sx**2/n
        S_yy = syy - sy**2/n
        S_xy = sxy - (sx * sy)/n
        RSS = S_yy - (S_xy**2)/S_xx
        return RSS