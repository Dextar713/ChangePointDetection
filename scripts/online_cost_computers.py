import numpy as np

class OnlineCostComputer:
    def __init__(self, cost_type: str, horizon_size:int = 100, signal_var: float|None = None):
        self.horizon_size = horizon_size
        self.cost_type = cost_type
        self.signal_var = signal_var
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
        n = self.horizon_size
        if self.cost_type == 'normal':
            beta = 5.0
            return beta * np.log(n)
        if self.cost_type == 'mean_var':
            beta = 2.0
            return 2 * beta * np.log(n)
        if self.signal_var is not None:
            variance = self.signal_var
        else:
            _, variance = self._get_stats(start, end)
        if self.cost_type == 'l2':
            beta = 4.0
            return beta * np.log(n) * variance
        
        raise ValueError('Unsupported cost type')
    

class LinearOnlineCostComputer:
    def __init__(self, horizon_size:int = 100):
        self.horizon_size = horizon_size
        self.sum_x = np.array([0])
        self.sum_y = np.array([0])
        self.sum_xx = np.array([0])
        self.sum_yy = np.array([0])
        self.sum_xy = np.array([0])
        
        self.sum_diff_var = np.array([0])
        self.n_samples = 0

    def double_size(self):
        self.sum_x = np.concatenate([self.sum_x, np.zeros(self.n_samples+1)])
        self.sum_y = np.concatenate([self.sum_y, np.zeros(self.n_samples+1)])
        self.sum_xx = np.concatenate([self.sum_xx, np.zeros(self.n_samples+1)])
        self.sum_yy = np.concatenate([self.sum_yy, np.zeros(self.n_samples+1)])
        self.sum_xy = np.concatenate([self.sum_xy, np.zeros(self.n_samples+1)])
        self.sum_diff_var = np.concatenate([self.sum_diff_var, np.zeros(self.n_samples+1)])
        # print(self.sum_diff_var.shape)


    def update(self, val):
        cur_size = self.sum_x.shape[0]
        self.n_samples += 1
        # print(self.n_samples)
        if cur_size <= self.n_samples:
            self.double_size()
        self.sum_x[self.n_samples] = self.sum_x[self.n_samples-1] + self.n_samples
        self.sum_y[self.n_samples] = self.sum_y[self.n_samples-1] + val
        self.sum_xx[self.n_samples] = self.sum_xx[self.n_samples-1] + self.n_samples**2
        self.sum_yy[self.n_samples] = self.sum_yy[self.n_samples-1] + val**2
        self.sum_xy[self.n_samples] = self.sum_xy[self.n_samples-1] + self.n_samples*val 
        if self.n_samples > 1:
            self.sum_diff_var[self.n_samples-1] = self.sum_diff_var[self.n_samples-2] + (val - self.sum_y[self.n_samples-1] + self.sum_y[self.n_samples-2])**2

    
    def cost(self, start: int | np.ndarray, end: int | np.ndarray) -> float:
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
        
    
    def get_penalty(self, start:int, end:int):
        n = end - start
        if n == 0:
            return 0
        beta = 1.2
        diff_var = (self.sum_diff_var[end-1] - self.sum_diff_var[start])/(n-1)
        # print(diff_var)
        return beta * np.log(self.horizon_size) * diff_var
        
        