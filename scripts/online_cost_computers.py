import numpy as np

class OnlineCostComputer:
    def __init__(self, cost_type: str):
        self.cost_type = cost_type
        self.cum_sum = np.array([0])
        self.cum_sum_sq = np.array([0])
        self.n_samples = 0

    def double_size(self):
        self.cum_sum = np.concatenate([self.cum_sum, np.zeros(self.n_samples+1)])
        self.cum_sum_sq = np.concatenate([self.cum_sum_sq, np.zeros(self.n_samples+1)])


    def update(self, val):
        cur_size = self.cum_sum.shape[0]
        self.n_samples += 1
        if cur_size <= self.n_samples:
            self.double_size()
        self.cum_sum[self.n_samples] = self.cum_sum[self.n_samples-1] + val
        self.cum_sum_sq[self.n_samples] = self.cum_sum_sq[self.n_samples-1] + val**2

    def _get_stats(self, start: int | np.ndarray, end: int | np.ndarray) -> tuple[float|np.ndarray, float|np.ndarray]:
        cur_len = end - start
        mean = (self.cum_sum[end] - self.cum_sum[start]) / cur_len
        variance = (self.cum_sum_sq[end] - self.cum_sum_sq[start]) / cur_len - mean * mean
        return mean, variance
    
    def cost(self, start: int | np.ndarray, end: int | np.ndarray) -> float:
        cur_len = end - start
        mean, variance = self._get_stats(start, end)
        if self.cost_type == 'l2':
            return cur_len * variance
        if self.cost_type == 'normal':
            return cur_len * np.log(variance)
        if self.cost_type == 'mean_var':
            sxx = self.cum_sum_sq[end] - self.cum_sum_sq[start]
            sx = self.cum_sum[end] - self.cum_sum[start]
            temp = sxx + cur_len * mean**2 - 2*mean*sx
            return cur_len * np.log(variance) + 1/variance * temp
        raise ValueError('Unsupported cost type')
    
    def get_penalty(self, start:int, end:int):
        n = end - start
        if n == 0:
            return 0
        if self.cost_type == 'normal':
            beta = 1.7
            return beta * np.log(n)
        if self.cost_type == 'mean_var':
            beta = 2.0
            return 2 * beta * np.log(n)
        _, variance = self._get_stats(start, end)
        if self.cost_type == 'l2':
            beta = 1.0
            return beta * np.log(n) * variance
        
        raise ValueError('Unsupported cost type')