import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

def main():
    with open('data/test_blackout_windows.pkl', 'rb') as f:
        test_windows = pickle.load(f)
    
    rng = np.random.default_rng(42)
    seeds_list = [w['context']['vr_seed_ms'] for w in test_windows]
    picks = np.sort(rng.choice(len(seeds_list), size=min(5, len(seeds_list)), replace=False))
    
    os.makedirs('data/debug_plots', exist_ok=True)
    
    for idx in picks:
        w = test_windows[idx]
        
        ctx_accel = w['context']['raw_accel']
        blk_accel = w['blackout']['raw_accel']
        accel = np.concatenate([ctx_accel, blk_accel], axis=0) # (700, 3)
        
        gt_speeds = w['ground_truth']['speeds_ms']
        
        t_accel = np.arange(-100, 600) / 10.0
        t_gt = np.arange(0, 600) / 10.0
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        
        ax1.plot(t_accel, accel[:, 0], label='Accel X', alpha=0.5)
        ax1.plot(t_accel, accel[:, 1], label='Accel Y', alpha=0.5)
        ax1.plot(t_accel, accel[:, 2], label='Accel Z', alpha=0.5)
        ax1.set_ylabel('IMU Accel')
        
        ax2.plot(t_gt, gt_speeds, color='black', linewidth=2, label='GT Speed')
        ax2.set_ylabel('Speed (m/s)')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title(f'Test episode {idx}')
        plt.savefig(f'data/debug_plots/episode_{idx}_raw.png')
        plt.close()

if __name__ == '__main__':
    main()
