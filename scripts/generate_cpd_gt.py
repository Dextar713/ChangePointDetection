import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_cp_series(num_tests: int = 10, num_points: int = 10, cost_type:str = 'linear'):
    methods = {"linear": generate_linear_cpd, 'normal': generate_variance_cpd, 
               'l2': generate_mean_cpd, 'mean_var': generate_variance_cpd}
    try:
        cur_method = methods[cost_type]
    except KeyError:
        raise ValueError(f"Cost type {cost_type} not supported")
    
    tests = []
    for _ in range(num_tests):
        series, cps = cur_method(num_points=num_points)
        tests.append((series, cps))
    return tests 
        


def generate_linear_cpd(num_points: int = 10):
    slope_range = (-1.7, 1.7)
    init_intercept_range = (20, 100)
    length_range = (20, 50)
    # length_range = (20, 150)
    prev_slope = np.random.uniform(low=slope_range[0], high=slope_range[1])
    init_intercept = np.random.uniform(low=init_intercept_range[0], high=init_intercept_range[1])
    init_length = np.random.randint(low=length_range[0], high=length_range[1]+1)
    time_series = [prev_slope*np.arange(init_length) + init_intercept + np.random.normal(0, 1, size=init_length)]
    change_points = []
    cur_point = init_length
    for _ in range(num_points):
        cur_slope = np.random.uniform(low=slope_range[0], high=slope_range[1])
        while np.abs(prev_slope-cur_slope)<0.4:
            cur_slope = np.random.uniform(low=slope_range[0], high=slope_range[1])
        cur_intercept = time_series[-1][-1]
        cur_length = np.random.randint(low=length_range[0], high=length_range[1]+1)
        x = np.arange(cur_length)
        noise_std = np.random.uniform(0.5, 2.5)
        # noise = np.random.normal(0, noise_std, size=cur_length)
        noise = scipy.stats.t.rvs(df=4,loc=0, scale=noise_std, size=cur_length)
        cur_segment = cur_slope * x + cur_intercept + noise 
        time_series.append(cur_segment)
        change_points.append(cur_point)
        cur_point += cur_length
        prev_slope = cur_slope
    return np.concatenate(time_series), change_points

def generate_variance_cpd(num_points: int = 10):
    std_list = [1.0]
    for i in range(num_points):
        std = np.random.uniform(0.3, 5.0)
        while std / std_list[-1] < 1.7 and std_list[-1] / std < 1.7:
            std = np.random.uniform(0.5, 7.0)
        std_list.append(std)
    # print(std_list)
    time_series = []
    cps = []
    last_point = 0
    for i in range(len(std_list)):
        cur_size = np.random.randint(low=50, high=80)
        segment = np.random.normal(0, std_list[i], size=cur_size)
        time_series.append(segment)
        last_point += cur_size
        cps.append(last_point)
    cps.pop()
    return np.concatenate(time_series), cps

def generate_mean_cpd(num_points: int = 10):
    base_mean = 10
    mean_list = [base_mean]
    std = base_mean * 0.2
    last_point = 25 
    drift_size = 8
    cps = []
    time_series = [np.random.normal(mean_list[0], std, size=last_point)]
    for i in range(num_points):
        factor_down = np.random.uniform(-0.5, -0.9)
        factor_up = np.random.uniform(0.5, 0.9)
        factor = np.random.choice([factor_up, factor_down])
        new_mean = mean_list[-1] + base_mean * factor
        # std = base_mean * 0.2
        if new_mean < mean_list[-1]:
            drift = np.linspace(new_mean, mean_list[-1], num=drift_size, endpoint=True)[::-1]
        else:
            drift = np.linspace(mean_list[-1], new_mean, num=drift_size, endpoint=True)
        time_series.append(drift + np.random.normal(std))
        cps.append(last_point + drift_size//2+1)
        new_size = np.random.randint(30, 60)
        mean_list.append(new_mean)
        segment = np.random.normal(new_mean, std, size=new_size)
        time_series.append(segment)
        last_point += drift_size + new_size 
    return np.concatenate(time_series), cps

    

def plot_cps(time_series: np.ndarray, change_points: list[int]):
    plt.figure(figsize=(12, 6))
    time_axes = np.arange(len(time_series))
    plt.plot(time_axes, time_series, color='gray', alpha=0.6, label='Time series')
    plt.title("True Change Points ")
    for point in change_points:
        plt.axvline(x=point, color='green', linestyle='--', linewidth=1, label='Change Point')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(time_series.min(), time_series.max())
    # save_path = os.path.join(os.path.dirname(__file__), "../visuals", "detected_change_points.png")
    # plt.savefig(save_path)
    plt.show()

def plot_cps_with_detections(time_series: np.ndarray, change_points: list[int], detections:list[int]):
    plt.figure(figsize=(12, 6))
    time_axes = np.arange(len(time_series))
    plt.plot(time_axes, time_series, color='gray', alpha=0.6, label='Time series')
    plt.title("Change Points True vs Detected")
    for point in change_points:
        plt.axvline(x=point, color='green', linestyle='--', linewidth=1, label='True Change Point')
    for point in detections:
        plt.axvline(x=point, color='red', linestyle='--', linewidth=1, label='Detected Change Point')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(time_series.min(), time_series.max())
    # save_path = os.path.join(os.path.dirname(__file__), "../visuals", "detected_change_points.png")
    # plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    tests = generate_cp_series(num_tests=1, num_points=10, cost_type='linear')
    plot_cps(tests[0][0], tests[0][1])

