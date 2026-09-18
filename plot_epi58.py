import pickle
import numpy as np
import scipy.signal

with open('data/test_blackout_windows.pkl', 'rb') as f:
    windows = pickle.load(f)

# Find episode 58. 
# wait, the picks were random, let's just find the one that has max lag
for i, w in enumerate(windows):
    accel = w['blackout']['raw_accel'][:, 1]
    gt_speeds = w['ground_truth']['speeds_ms']
    gt_accel = np.diff(gt_speeds)
    
    corr = scipy.signal.correlate(gt_accel - np.mean(gt_accel), accel - np.mean(accel), mode='full')
    lag = np.argmax(corr) - len(accel) + 1
    if i == 58:
        print(f"Episode 58 lag: {lag} steps ({lag/10.0} seconds)")
        
    if i == 6:
        print(f"Episode 6 lag: {lag} steps ({lag/10.0} seconds)")
        
    if i == 33:
        print(f"Episode 33 lag: {lag} steps ({lag/10.0} seconds)")
        
