import pickle
import numpy as np
import scipy.signal

with open('data/test_blackout_windows.pkl', 'rb') as f:
    windows = pickle.load(f)

lags = []
for idx, w in enumerate(windows):
    accel = w['blackout']['raw_accel'][:, 1] # Accel Y
    gt_speeds = w['ground_truth']['speeds_ms']
    gt_accel = np.diff(gt_speeds)
    gt_accel = np.insert(gt_accel, 0, gt_accel[0])
    
    # Cross correlate accel Y with gt_accel
    # Usually accel Y is negative when braking (decelerating)
    # Let's correlate
    corr = scipy.signal.correlate(gt_accel - np.mean(gt_accel), accel - np.mean(accel), mode='full')
    lags.append(np.argmax(corr) - len(accel) + 1)
    
print("Median lag:", np.median(lags))
print("Lag quantiles (0, 25, 50, 75, 100):", np.quantile(lags, [0, 0.25, 0.5, 0.75, 1]))
