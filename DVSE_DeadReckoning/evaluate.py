import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
from inference import inference_step
import os
import datetime

def evaluate():
    test_pkl = 'data/test_windows.pkl'
    scaler_pkl = 'data/scalers.pkl'
    
    if not os.path.exists(test_pkl):
        print("Data not found. Please run data_pipeline.py first.")
        return

    test_dataset = DeadReckoningDataset(test_pkl, scaler_pkl)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = DVSE()
    model_path = 'checkpoints/dvse_dead_reckoning.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded trained model: {model_path}")
    else:
        print("Model file not found! Please run train.py first.")
        return

    model.eval()
    
    all_pred_v = []
    all_target_v = []
    all_disp_errors = []
    
    print(f"Evaluating {len(test_dataset)} test sequences (60s windows)...")
    
    # For a clean plot, we will just save the first contiguous sequence
    plot_pred = []
    plot_tgt = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_train, vr_seed, delta_v_seq, target_disp, heading_target = batch
            
            delta_v_pred, euler_pred = inference_step(model, acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_seed)
            
            # Predict velocity
            v_cum = vr_seed + torch.cumsum(delta_v_pred.squeeze(-1), dim=1) # (1, T)
            v_cum = v_cum.squeeze(0).numpy() # (T,)
            
            # Ground truth velocity
            v_tgt = vr_train.squeeze(0)[:, 0].numpy() # (T,)
            
            all_pred_v.extend(v_cum)
            all_target_v.extend(v_tgt)
            
            # Calculate Displacement Error (60s window)
            heading_rad = euler_pred[..., 2]
            v_cum_t = torch.FloatTensor(v_cum).unsqueeze(0)
            
            delta_north = (v_cum_t * torch.cos(heading_rad)).sum(dim=1)
            delta_east = (v_cum_t * torch.sin(heading_rad)).sum(dim=1)
            pred_disp = torch.stack([delta_north, delta_east], dim=-1)
            
            disp_err = torch.norm(pred_disp - target_disp, dim=-1).item()
            all_disp_errors.append(disp_err)
            
            # Save the first 5 non-overlapping windows (15s step * 4 = 60s offset) for plotting
            if i % 4 == 0 and len(plot_pred) < 60 * 5:
                plot_pred.extend(v_cum)
                plot_tgt.extend(v_tgt)

    all_pred_v = np.array(all_pred_v)
    all_target_v = np.array(all_target_v)
    
    # --- Calculate Metrics ---
    errors = np.abs(all_pred_v - all_target_v)
    mae = np.mean(errors)
    p80 = np.percentile(errors, 80)
    avg_60s_disp = np.mean(all_disp_errors)
    
    print("\n=== DVSE Dead Reckoning Evaluation Results ===")
    print(f"Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"Velocity P80 Error: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)")
    print(f"Average 60s Haversine Displacement Error: {avg_60s_disp:.2f} meters")
        
    # --- Logging ---
    log_file = "evaluation_log.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Tested on: {test_pkl} (Vw04 Dead Reckoning)\n")
        f.write(f"  - Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)\n")
        f.write(f"  - Velocity P80: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)\n")
        f.write(f"  - Avg 60s Disp Error: {avg_60s_disp:.2f} m\n")
        f.write("-" * 50 + "\n")
    print(f"Results appended to {log_file}")
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(plot_tgt) * 3.6, label='Wheel Odometry (Ground Truth)', color='blue', linewidth=2)
    plt.plot(np.array(plot_pred) * 3.6, label='DVSE Predicted Speed (Autoregressive)', color='red', linestyle='dashed', linewidth=2)
    plt.title("DVSE Dead Reckoning: Autoregressive Speed Estimation (5 Mins)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Vehicle Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dvse_test_plot.png')
    print("Saved continuous plot to 'dvse_test_plot.png'")
    
if __name__ == '__main__':
    evaluate()
