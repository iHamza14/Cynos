import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
from inference import inference_step
from blackout import score_blackout_window
import os
import datetime
import pickle

def evaluate():
    test_pkl = 'data/test_blackout_windows.pkl'
    scaler_pkl = 'data/scalers.pkl'
    
    if not os.path.exists(test_pkl):
        print("Data not found. Please run blackout.py first.")
        return

    test_dataset = DeadReckoningDataset(test_pkl, scaler_pkl)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = DVSE()
    model_path = 'checkpoints/dvse_dead_reckoning.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded trained model: {model_path}")
    else:
        print("Model file not found! Please run train_stage1.py and train_stage2.py first.")
        return

    model.eval()
    
    all_pred_v = []
    all_target_v = []
    
    with open(test_pkl, 'rb') as f:
        windows = pickle.load(f)
        
    scores = []
    
    print(f"Evaluating {len(test_dataset)} test sequences...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            (acc_t, gyro_t, mtn_t, raw_t, grav_t, 
             vr_train, vr_seed, delta_v_seq, target_disp, 
             heading_target, raw_gyro_t, start_heading_t) = batch
            
            delta_v_pred, euler_pred = inference_step(model, acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_seed)
            
            # Ground truth velocity
            v_tgt = vr_train.squeeze(0)[:, 0].numpy() # (T,)
            
            # Predict velocity
            v_cum = vr_seed.expand(-1, v_tgt.shape[0]) # (1, T)
            v_cum = v_cum.squeeze(0).numpy() # (T,)
            
            all_pred_v.extend(v_cum)
            all_target_v.extend(v_tgt)
            
            # Relative yaw integration
            start_heading = start_heading_t[0].item()
            gyro_yaw_rate = raw_gyro_t[0, :, 2].numpy()
            
            yaw_pred_rad = np.zeros(len(v_cum))
            yaw_pred_rad[0] = start_heading
            for t in range(1, len(v_cum)):
                yaw_pred_rad[t] = yaw_pred_rad[t-1] + gyro_yaw_rate[t-1]
                
            # Compute cumulative displacement vectors
            dn = v_cum * np.cos(yaw_pred_rad)
            de = v_cum * np.sin(yaw_pred_rad)
            dr_positions_ne = np.stack([np.cumsum(dn), np.cumsum(de)], axis=-1)
            
            # Score against Hackathon Constraints
            score = score_blackout_window(windows[i], dr_positions_ne)
            score['duration'] = windows[i]['blackout_dur_sec']
            scores.append(score)

    all_pred_v = np.array(all_pred_v)
    all_target_v = np.array(all_target_v)
    
    # --- Velocity Metrics ---
    errors = np.abs(all_pred_v - all_target_v)
    mae = np.mean(errors)
    p80 = np.percentile(errors, 80)
    
    # --- Drift Metrics (60s windows only for standard) ---
    scores_60s = [s for s in scores if s['duration'] == 60]
    avg_max_error = np.mean([s['max_error_m'] for s in scores_60s])
    avg_max_pct_drift = np.mean([s['max_pct_drift'] for s in scores_60s])
    avg_total_dist = np.mean([s['total_distance_m'] for s in scores_60s])
    
    print("\n=== DVSE Hackathon Constraint Evaluation ===")
    print(f"Total 60-Second Windows Tested: {len(scores_60s)}")
    print(f"Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"Velocity P80 Error: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)")
    print("-" * 50)
    print("Drift Constraints (60s Windows):")
    print(f"  Average Distance Travelled:  {avg_total_dist:.1f} meters")
    print(f"  Average Maximum Drift:       {avg_max_error:.2f} meters")
    print(f"  Average Maximum % Drift:     {avg_max_pct_drift:.2f}% (Constraint: < 10%)")
    
    if avg_max_pct_drift < 10.0:
        print("  => STATUS: PASSED (Under 10% Drift Constraint)")
    else:
        print("  => STATUS: FAILED (Exceeds 10% Drift Constraint)")

if __name__ == '__main__':
    evaluate()
