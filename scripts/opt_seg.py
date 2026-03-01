from cost_computers import CostComputer, LinearCostComputer, get_gain_threshold
from online_cost_computers import OnlineCostComputer, LinearOnlineCostComputer
import numpy as np

class OptSegmentation:
    def __init__(self, model = 'normal', min_dist: int = 5):
        self.model = model
        self.min_dist = min_dist
        self.cost_computer = None

    def fit_predict(self, signal: np.ndarray, horizon_size:int|None=None) -> list[int]:
        n = len(signal)
        if n < 2 * self.min_dist:
            return []
        min_gain = get_gain_threshold(signal, self.model, horizon_size=horizon_size)
        if self.model == 'linear':
            self.cost_computer = LinearCostComputer(signal)
        else:
            self.cost_computer = CostComputer(signal, self.model)
        C = np.full(shape=n+1, fill_value=np.inf)
        C[0] = -min_gain
        path = np.zeros(n+1, dtype=int)
        candidates_set = set()
        for t in range(self.min_dist, n+1):
            candidates_set.add(t-self.min_dist)
            if t == n:
                for i in range(self.min_dist//2):
                    candidates_set.add(t-self.min_dist+1+i)
            # prev_points = np.arange(t-self.min_dist+1)
            prev_points = np.fromiter(candidates_set, dtype=int, count=len(candidates_set))
            prev_costs = C[prev_points]
            cur_costs = self.cost_computer.cost(prev_points, t)
            total_cost = prev_costs + cur_costs + min_gain
            best_idx = np.argmin(total_cost)
            min_cost = total_cost[best_idx]
            best_point = prev_points[best_idx]
            C[t] = min_cost
            path[t] = best_point
            for p in prev_points:
                if C[p] + self.cost_computer.cost(p, t) + min_gain > C[t]:
                    candidates_set.discard(p)

        if horizon_size is not None:
            return [path[n]]
        change_points = []
        cur_point = n
        while cur_point > 0:
            cur_point = path[cur_point]
            change_points.append(cur_point)
        change_points.pop()
        change_points.reverse()
        #noinspection PyTypeChecker
        return change_points
    

class OnlineOptSegmentation:
    def __init__(self, cost_type = 'normal', min_dist: int = 5, horizon_size: int = 60):
        self.cost_type = cost_type
        self.min_dist = min_dist
        if cost_type == 'linear':
            self.cost_computer = LinearOnlineCostComputer(horizon_size=horizon_size)
        else:
            self.cost_computer = OnlineCostComputer(cost_type, horizon_size=horizon_size)
        self.horizon_size = horizon_size
        self.n_samples = 0 
        self.C = np.full(2*self.min_dist+1, np.inf)
        self.path = np.zeros(2*self.min_dist+1, dtype=int)
        self.candidates_set = set()

    def double_size(self):
        self.C = np.concatenate([self.C, np.full(self.n_samples+1, np.inf)])
        self.path = np.concatenate([self.path, np.zeros(self.n_samples+1, dtype=int)], dtype=int)


    def update(self, value) -> list[int]:
        self.n_samples += 1 
        self.cost_computer.update(value)
        if self.n_samples < 2*self.min_dist:
            return []
        start, end = max(0, self.n_samples - self.horizon_size), self.n_samples
        penalty = self.cost_computer.get_penalty(start, end)
        if self.C[0] == np.inf:
            self.C[0] = -penalty
            start_idx = self.min_dist
        else:
            start_idx = self.n_samples
            if self.C.shape[0] <= self.n_samples:
                self.double_size()
        for t in range(start_idx, self.n_samples+1):
            self.candidates_set.add(t-self.min_dist)
            if t == self.n_samples:
                for i in range(self.min_dist//2):
                    self.candidates_set.add(t-self.min_dist+1+i)
            oldest_point = max(0, self.n_samples-self.horizon_size)
            candidates_to_delete = []
            for candidate in self.candidates_set:
                if candidate < oldest_point:
                    candidates_to_delete.append(candidate)
            for candidate in candidates_to_delete:
                self.candidates_set.discard(candidate)
            # prev_points = np.arange(oldest_point, t-self.min_dist+1, dtype=int)
            prev_points = np.fromiter(self.candidates_set, dtype=int, count=len(self.candidates_set))
            prev_costs = self.C[prev_points]
            cur_costs = self.cost_computer.cost(prev_points, t)
            total_costs = prev_costs + cur_costs + penalty
            best_idx = np.argmin(total_costs)
            min_cost = total_costs[best_idx]
            best_point = prev_points[best_idx]
            self.C[t] = min_cost
            self.path[t] = best_point
            for p in prev_points:
                if self.C[p] + self.cost_computer.cost(p, t) > self.C[t]:
                    self.candidates_set.discard(p)
        return self.backtrack_changepoints()

    def backtrack_changepoints(self) -> list[int]:
        change_points = []
        cur_point = self.n_samples
        oldest_point = max(0, self.n_samples - self.horizon_size)
        while cur_point > oldest_point:
            cur_point = self.path[cur_point]
            change_points.append(cur_point)
        # change_points.pop()
        if cur_point == 0:
            change_points.pop()
        change_points.reverse()
        return change_points
        
            
