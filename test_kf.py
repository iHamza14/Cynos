import pickle, numpy as np
windows = pickle.load(open('data/test_blackout_windows.pkl', 'rb'))
for i in range(3):
    w = windows[i]
    print(f"Window {i}")
    print("Mobile GPS:", w['ground_truth']['gps_eval']['mobile_speed_ms'][:5])
    print("GT Speed:", w['ground_truth']['speeds_ms'][:5])
