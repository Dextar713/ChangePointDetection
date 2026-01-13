import numpy as np

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
        syy = self.sum_yy[end] - self.sum_yy[start]
        sx = self.sum_x[end] - self.sum_x[start]
        sy = self.sum_y[end] - self.sum_y[start]
        sxy = self.sum_xy[end] - self.sum_xy[start]
        S_xx = sxx - sx**2/n
        S_yy = syy - sy**2/n
        S_xy = sxy - (sx * sy)/n
        RSS = S_yy - (S_xy**2)/S_xx
        return RSS


def get_gain_threshold(signal: np.ndarray, cost_type: str) -> float:
    n = len(signal)
    if n == 0:
        return 0
    if cost_type == 'l2':
        beta = 1.0
        signal_var = np.var(signal)
        return beta * np.log(n) * signal_var
    if cost_type == 'normal':
        beta = 1.7
        return beta * np.log(n)
    if cost_type == 'linear':
        beta = 2.0
        diff_var = np.var(np.diff(signal))
        return (2 * beta) * np.log(n) * diff_var / 2
    if cost_type == 'mean_var':
        beta = 2.0
        return 2 * beta * np.log(n)
    raise ValueError('Unsupported cost type')

class BinarySegmentation:
    def __init__(self, model = 'normal', min_dist: int = 5):
        self.model = model
        self.min_dist = min_dist
        self.cost_computer = None

    def _best_single_point(self, start: int, end: int) -> tuple[int, float]:
        max_gain = -np.inf
        best_point = -1
        segment_cost = self.cost_computer.cost(start, end)
        for k in range(start + self.min_dist, end - self.min_dist):
            left_cost = self.cost_computer.cost(start, k)
            right_cost = self.cost_computer.cost(k, end)
            gain = segment_cost - left_cost - right_cost
            if gain > max_gain:
                max_gain = gain
                best_point = k
        return best_point, max_gain


    def fit_predict(self, signal: np.ndarray) -> tuple[list[int], list[float]]:
        min_gain = get_gain_threshold(signal, self.model)
        if self.model == 'linear':
            self.cost_computer = LinearCostComputer(signal)
        else:
            self.cost_computer = CostComputer(signal, self.model)
        segments = [(0, len(signal))]
        gains = []
        while True:
            best_gain = -np.inf
            best_point = -1
            best_segment_id = -1
            for i, (start, end) in enumerate(segments):
                if end - start < self.min_dist * 2:
                    continue
                cur_point, gain = self._best_single_point(start, end)
                if gain > best_gain:
                    best_gain = gain
                    best_point = cur_point
                    best_segment_id = i
            if best_point == -1 or best_gain < min_gain:
                break
            gains.append(best_gain)
            start, end = segments.pop(best_segment_id)
            segments.append((start, best_point))
            segments.append((best_point, end))
        change_points = [s[1] for s in sorted(segments)[:-1]]
        return change_points, gains

class OptSegmentation:
    def __init__(self, model = 'normal', min_dist: int = 5):
        self.model = model
        self.min_dist = min_dist
        self.cost_computer = None

    def fit_predict(self, signal: np.ndarray) -> list[int]:
        min_gain = get_gain_threshold(signal, self.model)
        if self.model == 'linear':
            self.cost_computer = LinearCostComputer(signal)
        else:
            self.cost_computer = CostComputer(signal, self.model)
        n = len(signal)
        C = np.full(shape=n+1, fill_value=np.inf)
        C[0] = -min_gain
        path = np.zeros(n+1, dtype=int)
        for t in range(self.min_dist, n+1):
            prev_points = np.arange(t-self.min_dist+1)
            prev_costs = C[prev_points]
            cur_costs = self.cost_computer.cost(prev_points, t)
            total_cost = prev_costs + cur_costs + min_gain
            best_idx = np.argmin(total_cost)
            min_cost = total_cost[best_idx]
            best_point = prev_points[best_idx]
            C[t] = min_cost
            path[t] = best_point
        change_points = []
        cur_point = n
        while cur_point > 0:
            cur_point = path[cur_point]
            change_points.append(cur_point)
        change_points.pop()
        change_points.reverse()
        #noinspection PyTypeChecker
        return change_points

