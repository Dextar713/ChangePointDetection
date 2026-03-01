import numpy as np
from bin_seg import BinarySegmentation
from opt_seg import OptSegmentation, OnlineOptSegmentation 

class NaiveOnlineDetector:
    def __init__(self, cost_type:str, model_type:str = 'opt', min_dist: int = 5, horizon_size:int=30):
        if model_type == 'opt':
            self.model = OptSegmentation(cost_type, min_dist)
        elif model_type == 'binseg':
            self.model = BinarySegmentation(cost_type, min_dist)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        self.signal_buffer = []
        self.min_dist = min_dist
        self.last_change_point = 0
        self.horizon_size = horizon_size
        self.n_samples = 0
        self.offset = 0


    def update(self, value) -> bool:
        self.n_samples += 1
        self.signal_buffer.append(value)
        if len(self.signal_buffer) < 2 * self.min_dist:
            return False 
        change_points = self.model.fit_predict(np.array(self.signal_buffer), horizon_size=self.horizon_size)
        if len(change_points) == 0:
            return False
        
        if np.abs(change_points[-1] - self.last_change_point) < self.min_dist or change_points[-1] < self.last_change_point:
            return False
        # cutoff_idx = max(0, self.last_change_point-2*self.min_dist)
        cutoff_idx = max(0, self.last_change_point - self.horizon_size)
        self.signal_buffer = self.signal_buffer[cutoff_idx:]
        self.offset += cutoff_idx
        self.last_change_point = change_points[-1] - cutoff_idx
        return True


class FastOnlineDetector:
    def __init__(self, cost_type:str, min_dist: int = 5, horizon_size:int=60):
        self.model = OnlineOptSegmentation(cost_type, min_dist, horizon_size=horizon_size)
        # self.signal_buffer = []
        self.min_dist = min_dist
        self.offset: int = 0
        self.last_change_point: int = 0
        self.horizon_size = horizon_size
        self.n_samples: int = 0

    def update(self, value) -> bool:
        # self.signal_buffer.append(value)
        self.n_samples += 1
        change_points = self.model.update(value)
        if len(change_points) == 0:
            return False 
        cur_last_point = change_points[-1] - self.offset
        if np.abs(cur_last_point - self.last_change_point) < self.min_dist or cur_last_point < self.last_change_point:
            return False
        
        # cutoff_idx = max(0, self.last_change_point-self.horizon_size)
        # self.signal_buffer = self.signal_buffer[cutoff_idx:]
        # self.offset += cutoff_idx
        self.last_change_point = change_points[-1]
        return True