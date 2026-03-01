from generate_cpd_gt import generate_cp_series, plot_cps_with_detections
from opt_seg import OptSegmentation
from bin_seg import BinarySegmentation
from online_detector import NaiveOnlineDetector, FastOnlineDetector
import numpy as np 


def test_cpd_offline(cost_type:str='linear', model:str = 'opt', min_dist:int = 15, plot:bool = False):
    tests = generate_cp_series(num_tests=30, num_points=100, cost_type=cost_type)
    if model == 'opt':
        detector = OptSegmentation(model=cost_type, min_dist=min_dist)
    elif model == 'binseg':
        detector = BinarySegmentation(model=cost_type, min_dist=min_dist)
    else:
        raise ValueError("Model not supported")
    precision_list = []
    recall_list = []
    for test in tests:
        detected_cps = detector.fit_predict(test[0])
        if plot:
            plot_cps_with_detections(test[0], test[1], detected_cps)
        # print('Num true cps: ', len(test[1]))
        # print('Num detected cps: ', len(detected_cps))
        precision, recall = calc_precision_recall(test[1], detected_cps, tolerance=min_dist//2+1)
        precision_list.append(precision)
        recall_list.append(recall)
        # print('Precision: ', precision)
        # print('Recall: ', recall)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    return avg_precision, avg_recall

def test_cpd_online(cost_type:str='linear', model:str = 'opt', min_dist:int = 15, plot:bool = False):
    tests = generate_cp_series(num_tests=50, num_points=50, cost_type=cost_type)
    detector = NaiveOnlineDetector(cost_type=cost_type, model_type=model, min_dist=min_dist, horizon_size=90)
    # detector = FastOnlineDetector(cost_type=cost_type, min_dist=min_dist, horizon_size=80)
    precision_list = []
    recall_list = []
    for test in tests:
        detected_cps = []
        for i in range(len(test[0])):
            is_detected = detector.update(test[0][i])
            if is_detected:
                detected_cps.append(i)
        if plot:
            plot_cps_with_detections(test[0], test[1], detected_cps)
        # print('Num true cps: ', len(test[1]))
        # print('Num detected cps: ', len(detected_cps))
        precision, recall = calc_precision_recall(test[1], detected_cps, tolerance=min_dist*2//3)
        precision_list.append(precision)
        recall_list.append(recall)
        # print('Precision: ', precision)
        # print('Recall: ', recall)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    return avg_precision, avg_recall

def calc_precision_recall(true_cps, detected_cps, tolerance=5):
    detected_set = set(detected_cps)
    true_positives = 0
    for true_cp in true_cps:
        # if any(abs(true_cp - detected_cp) <= tolerance for detected_cp in detected_cps):
        #     true_positives += 1
        cp_idx = np.searchsorted(detected_cps, true_cp)
        idx = cp_idx
        while idx < len(detected_cps) and abs(detected_cps[idx] - true_cp) <= tolerance:
            if detected_cps[idx] in detected_set:
                true_positives += 1
                detected_set.remove(detected_cps[idx])  
                break
            idx += 1
        if idx == len(detected_cps) or abs(detected_cps[idx] - true_cp) > tolerance:
            idx = cp_idx - 1
            while idx >= 0 and abs(detected_cps[idx] - true_cp) <= tolerance:
                if detected_cps[idx] in detected_set:
                    true_positives += 1
                    detected_set.remove(detected_cps[idx])  
                    break
                idx -= 1
        
    precision = true_positives / len(detected_cps) if detected_cps else 0
    recall = true_positives / len(true_cps) if true_cps else 0
    return precision, recall


if __name__ == '__main__':
    # test_cpd_offline(cost_type='linear', model='opt', min_dist=15)
    test_cpd_online(cost_type='linear', model='opt', min_dist=15)