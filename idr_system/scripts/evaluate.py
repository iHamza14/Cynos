import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.idr_model import IDRModel
from ekf.differentiable_ekf import DifferentiableEKF
from data.dataset import IDRDataset
from scripts.train import run_ekf_forward

def evaluate():
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Load best model
    model = IDRModel(config).to(device)
    model.load_state_dict(torch.load('data/best_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    ekf = DifferentiableEKF(config).to(device)
    
    val_dataset = IDRDataset('data/test_blackout_windows.pkl', scalers_path='data/scalers.pkl', augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    results = {15: [], 30: [], 60: []}
    
    os.makedirs('plots', exist_ok=True)
    
    print("Evaluating Test Windows...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            # 1. TCN + EKF Forward
            preds_speed, preds_traj = run_ekf_forward(model, ekf, batch, config)
            
            gt_traj = batch['gt_cum_disp']
            pred_end_pos = preds_traj[0, -1, :]
            gt_end_pos = gt_traj[0, -1, :]
            
            end_pos_error = torch.sqrt(torch.sum((pred_end_pos - gt_end_pos)**2)).item()
            dist_traveled = batch['total_dist'][0].item()
            
            drift_pct = (end_pos_error / (dist_traveled + 1e-8)) * 100.0
            dur = int(batch['blackout_dur'][0].item())
            
            if dur in results:
                results[dur].append(drift_pct)
                
            # 2. Raw integration (F8 baseline)
            # Simple double integration of rotated raw accel (using initial yaw)
            raw_a = batch['raw_a_10hz'][0] # (seq_len, 3)
            dt = config['ekf']['dt']
            initial_yaw = batch['initial_yaw'][0].item()
            vr_seed = batch['vr_seed'][0].item()
            
            v_raw = np.zeros((raw_a.shape[0], 2))
            p_raw = np.zeros((raw_a.shape[0], 2))
            
            v_raw[0, 0] = vr_seed * np.cos(initial_yaw)
            v_raw[0, 1] = vr_seed * np.sin(initial_yaw)
            
            # Simple 2D rotation for raw accel (assuming phone is perfectly aligned in pitch/roll)
            cos_y = np.cos(initial_yaw)
            sin_y = np.sin(initial_yaw)
            
            for t in range(1, raw_a.shape[0]):
                ax = raw_a[t, 0].item()
                ay = raw_a[t, 1].item()
                
                # Rotate to world
                ax_w = ax * cos_y - ay * sin_y
                ay_w = ax * sin_y + ay * cos_y
                
                v_raw[t, 0] = v_raw[t-1, 0] + ax_w * dt
                v_raw[t, 1] = v_raw[t-1, 1] + ay_w * dt
                
                p_raw[t, 0] = p_raw[t-1, 0] + v_raw[t-1, 0] * dt + 0.5 * ax_w * dt**2
                p_raw[t, 1] = p_raw[t-1, 1] + v_raw[t-1, 1] * dt + 0.5 * ay_w * dt**2
                
            # Plot
            plt.figure(figsize=(8, 6))
            
            gt_np = gt_traj[0].cpu().numpy()
            pred_np = preds_traj[0].cpu().numpy()
            
            plt.plot(gt_np[:, 0], gt_np[:, 1], label='Ground Truth (GPS)', linewidth=2)
            plt.plot(pred_np[:, 0], pred_np[:, 1], label='TCN + EKF', linewidth=2)
            plt.plot(p_raw[:, 0], p_raw[:, 1], label='Raw Integration', linestyle='dashed')
            
            plt.title(f'Window {i} ({dur}s) - Drift: {drift_pct:.2f}%')
            plt.xlabel('Easting (m)')
            plt.ylabel('Northing (m)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(f'plots/window_{i}_{dur}s.png')
            plt.close()

    print("\n--- Evaluation Report ---")
    print(f"{'Duration':<10} | {'Median Drift %':<15} | {'80th %ile Drift %':<20} | {'N windows':<10}")
    print("-" * 65)
    for dur in [15, 30, 60]:
        drifts = results[dur]
        if len(drifts) > 0:
            median = np.median(drifts)
            p80 = np.percentile(drifts, 80)
            print(f"{dur:<10} | {median:<15.2f} | {p80:<20.2f} | {len(drifts):<10}")
        else:
            print(f"{dur:<10} | {'N/A':<15} | {'N/A':<20} | {0:<10}")

if __name__ == "__main__":
    evaluate()
