import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('data/proof', exist_ok=True)

with open('data/test_blackout_windows.pkl', 'rb') as f:
    windows = pickle.load(f)

for idx in [6, 33, 58]:
    w = windows[idx]
    
    accel_y = w['blackout']['raw_accel'][:, 1]
    gt_speeds = w['ground_truth']['speeds_ms']
    
    t = np.arange(0, 600) / 10.0
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    # Plot GT speed on left axis (Blue)
    ax1.plot(t, gt_speeds, color='blue', linewidth=2, label='Vehicle GT Speed (OBD)')
    ax1.set_ylabel('Speed (m/s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot IMU Accel Y on right axis (Red)
    ax2.plot(t, accel_y, color='red', alpha=0.7, label='Raw IMU Accel Y (Phone)')
    ax2.set_ylabel('IMU Accel (m/s^2)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f'Test Episode {idx}: Raw Data Alignment Check')
    fig.tight_layout()
    plt.savefig(f'data/proof/episode_{idx}_raw_alignment.png')
    plt.close()

