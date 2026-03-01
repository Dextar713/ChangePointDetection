from cost_computers import CostComputer, LinearCostComputer, get_gain_threshold
import numpy as np

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


    def fit_predict(self, signal: np.ndarray, horizon_size:int|None = None) -> list[int]:
        n = len(signal)
        if n < 2 * self.min_dist:
            return []
        min_gain = get_gain_threshold(signal, self.model, horizon_size=horizon_size)
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
        return change_points