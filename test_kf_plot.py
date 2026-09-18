import pickle, numpy as np, torch
import matplotlib.pyplot as plt

windows = pickle.load(open('data/test_blackout_windows.pkl', 'rb'))
model_path = 'data/tcn_delta_v.pt'

# We'll just load the first test window to see
w = windows[49] # just pick one at random, say index 4
w_idx = 4
for i, w in enumerate(windows):
    if i == 4:
        break

scalers = pickle.load(open('data/scalers.pkl', 'rb'))
x = np.concatenate([scalers[k].transform(np.concatenate([w['context'][k], w['blackout'][k]], axis=0))
                    for k in ('raw_accel', 'raw_gyro', 'gravity')], axis=1).astype(np.float32)

import sys
sys.path.append('.')
from train.diagnose_tcn_delta_v import TCN
model = TCN()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

with torch.no_grad():
    dv_pred = model(torch.from_numpy(x.T).unsqueeze(0))[:, -600:].squeeze(0).numpy()

dv_true = np.asarray(w['ground_truth']['delta_v_ms'], dtype=np.float32)
seed = float(w['context']['vr_seed_ms'])
gt_speed = np.asarray(w['ground_truth']['speeds_ms'])
mobile_gps = np.asarray(w['ground_truth']['gps_eval']['mobile_speed_ms'])

# Pure integration (what we have now)
pure_v = seed + np.cumsum(dv_pred)

# Kalman Filter
# State: [v, b]
x_kf = np.array([seed, 0.0])
P = np.array([[1.0, 0.0], [0.0, 0.1]])
F = np.array([[1.0, -1.0], [0.0, 1.0]])
Q = np.array([[0.05, 0.0], [0.0, 0.0001]]) # Tune these
H = np.array([[1.0, 0.0]])
R = np.array([[1.0]]) # GPS is somewhat noisy/laggy

kf_v = []
for t in range(600):
    # Predict
    x_kf[0] = x_kf[0] + (dv_pred[t] - x_kf[1])
    # x_kf[1] remains the same
    P = F @ P @ F.T + Q
    
    # Update (fuse with GPS)
    # To handle GPS 1Hz updates, we could only update when mobile_gps changes!
    # Let's see if updating every step works well, since it's interpolated
    z = np.array([mobile_gps[t]])
    
    # Let's only update when GPS actually changes value to simulate 1Hz?
    # Actually mobile_gps_speed in CSV is 1Hz zero-order hold usually.
    # We can just update every step with higher R, or only on change.
    update = True
    if t > 0 and mobile_gps[t] == mobile_gps[t-1]:
        # it's a held value, we can still update but with huge R, or skip
        update = False
        
    if update:
        y = z - (H @ x_kf)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_kf = x_kf + K @ y
        P = (np.eye(2) - K @ H) @ P
        
    kf_v.append(x_kf[0])

kf_v = np.array(kf_v)

print("Pure MAE:", np.mean(np.abs(pure_v - gt_speed)))
print("KF MAE:", np.mean(np.abs(kf_v - gt_speed)))

plt.figure(figsize=(10,4))
plt.plot(gt_speed, label='Vehicle GT (Target)')
plt.plot(pure_v, label='TCN Pure Integration')
plt.plot(kf_v, label='TCN + Kalman Filter')
plt.plot(mobile_gps, label='Mobile GPS (Raw)', alpha=0.5)
plt.legend()
plt.savefig('kf_test.png')
