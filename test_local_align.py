import pickle
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

with open('data/test_blackout_windows.pkl', 'rb') as f:
    windows = pickle.load(f)

aligned = 0
dropped = 0
for idx in [6, 33, 58, 77]:
    w = windows[idx]
    
    accel_y = w['blackout']['raw_accel'][:, 1]
    gt_speeds = w['ground_truth']['speeds_ms']
    gt_accel = np.diff(gt_speeds)
    gt_accel = np.insert(gt_accel, 0, gt_accel[0])
    
    # We don't have the extended IMU data in the window! 
    # The window only has 600 steps. We can't shift it if the lag is -19s, 
    # because we need IMU data from 19s BEFORE the window!
    pass
