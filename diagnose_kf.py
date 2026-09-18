import pickle, torch, numpy as np
import matplotlib.pyplot as plt

def load(path):
    with open(path,'rb') as f: return pickle.load(f)

windows = load('data/test_blackout_windows.pkl')
scalers = load('data/scalers.pkl')

import sys
sys.path.append('.')
from train.diagnose_tcn_delta_v import TCN
model = TCN()
model.load_state_dict(torch.load('data/tcn_delta_v.pt', map_location='cpu'))
model.eval()

kf_maes = []
pure_maes = []
base_maes = []

for i, w in enumerate(windows):
    try:
        x = np.concatenate([scalers[k].transform(np.concatenate([w['context'][k], w['blackout'][k]], axis=0))
                            for k in ('raw_accel', 'raw_gyro', 'gravity')], axis=1).astype(np.float32)
        with torch.no_grad():
            dv_pred = model(torch.from_numpy(x.T).unsqueeze(0))[:, -600:].squeeze(0).numpy()
            
        gt_speed = np.asarray(w['ground_truth']['speeds_ms'])
        mobile_gps = np.asarray(w['ground_truth']['gps_eval']['mobile_speed_ms'])
        seed = float(w['context']['vr_seed_ms'])
        
        if not np.isfinite(gt_speed).all() or not np.isfinite(mobile_gps).all(): continue
        
        pure_v = seed + np.cumsum(dv_pred)
        base_v = np.full_like(pure_v, seed)
        
        x_kf = np.array([seed, 0.0])
        P = np.array([[1.0, 0.0], [0.0, 0.1]])
        F = np.array([[1.0, -1.0], [0.0, 1.0]])
        Q = np.array([[0.05, 0.0], [0.0, 0.0001]]) 
        H = np.array([[1.0, 0.0]])
        R = np.array([[10.0]]) # GPS is very laggy, trust it less (high R)
        
        kf_v = []
        for t in range(600):
            x_kf[0] = x_kf[0] + (dv_pred[t] - x_kf[1])
            P = F @ P @ F.T + Q
            
            update = True
            if t > 0 and mobile_gps[t] == mobile_gps[t-1]:
                update = False
                
            if update:
                y = np.array([mobile_gps[t]]) - (H @ x_kf)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x_kf = x_kf + K @ y
                P = (np.eye(2) - K @ H) @ P
                
            kf_v.append(x_kf[0])
            
        kf_v = np.array(kf_v)
        
        pure_maes.append(np.abs(pure_v[-1] - gt_speed[-1]))
        kf_maes.append(np.abs(kf_v[-1] - gt_speed[-1]))
        base_maes.append(np.abs(base_v[-1] - gt_speed[-1]))
        
        if i in [10, 20, 30]:
            plt.figure(figsize=(10,4))
            plt.plot(gt_speed, label='Vehicle GT')
            plt.plot(pure_v, label='Pure Integration')
            plt.plot(kf_v, label='Kalman Filter')
            plt.plot(mobile_gps, label='Mobile GPS')
            plt.legend()
            plt.title(f'Window {i}')
            plt.savefig(f'kf_window_{i}.png')
            plt.close()
    except Exception as e:
        print("Error:", e)

print(f"Mean MAE @ 60s:")
print(f"Baseline: {np.mean(base_maes):.4f}")
print(f"Pure TCN: {np.mean(pure_maes):.4f}")
print(f"KF TCN:   {np.mean(kf_maes):.4f}")
