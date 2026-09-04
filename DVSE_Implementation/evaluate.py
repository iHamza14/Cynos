import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import DVSE
from dataset import DVSEDataset
import os

def evaluate():
    test_path = 'data/test_windows.csv'
    if not os.path.exists(test_path):
        print("Data not found. Please run data_sync.py first.")
        return

    test_dataset = DVSEDataset(test_path, seq_len=10)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 for sequential evaluation
    
    model = DVSE()
    if os.path.exists('dvse_model.pth'):
        model.load_state_dict(torch.load('dvse_model.pth', weights_only=True))
        print("Loaded trained model: dvse_model.pth")
    else:
        print("Model file 'dvse_model.pth' not found! Please run train.py first.")
        return

    model.eval()
    
    all_pred_v = []
    all_target_v = []
    
    print(f"Evaluating {len(test_dataset)} test sequences...")
    with torch.no_grad():
        for batch in test_loader:
            acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_dv, target_v = batch
            
            pred_dv, euler = model(acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav)
            
            # Reconstruct velocity
            pred_v = torch.zeros_like(pred_dv)
            pred_v[:, 0] = target_v[:, 0] # Ground truth initial velocity
            for t in range(1, pred_dv.size(1)):
                pred_v[:, t] = pred_v[:, t-1] + pred_dv[:, t]
                
            # For graphing, we take the final predicted velocity of the 10-second sequence
            # (or we could flatten them, but taking the last step shows the accumulated drift)
            all_pred_v.append(pred_v[0, -1, 0].item())
            all_target_v.append(target_v[0, -1, 0].item())

    all_pred_v = np.array(all_pred_v)
    all_target_v = np.array(all_target_v)
    
    # --- Calculate Metrics (As requested by the paper) ---
    errors = np.abs(all_pred_v - all_target_v)
    mae = np.mean(errors)
    p80 = np.percentile(errors, 80)
    
    print("\n=== DVSE Evaluation Results (Test Split) ===")
    print(f"Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"Velocity P80 Error: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)")
    
    # Distance Error Calculation (Averaged over given time frames)
    # distance = sum(v * dt), dt = 1 sec
    def avg_distance_error(pred, target, frame_size):
        if len(pred) < frame_size: return None
        errors = []
        # Calculate non-overlapping windows of frame_size
        for i in range(0, len(pred) - frame_size + 1, frame_size):
            dist_pred = np.sum(pred[i:i+frame_size])
            dist_tgt = np.sum(target[i:i+frame_size])
            errors.append(np.abs(dist_pred - dist_tgt))
        return np.mean(errors)

    err_30s = avg_distance_error(all_pred_v, all_target_v, 30)
    err_60s = avg_distance_error(all_pred_v, all_target_v, 60)
    
    if err_30s is not None:
        print(f"Average Distance Error (30-sec timeframe): {err_30s:.2f} meters")
    if err_60s is not None:
        print(f"Average Distance Error (60-sec timeframe): {err_60s:.2f} meters")
        
    # --- Logging ---
    import datetime
    log_file = "evaluation_log.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Tested on: {test_path} (Vw04 Test Split)\n")
        f.write(f"  - Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)\n")
        f.write(f"  - Velocity P80: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)\n")
        f.write(f"  - Avg Distance Error (30s): {err_30s:.2f} m\n" if err_30s else "  - Avg Distance Error (30s): N/A\n")
        f.write(f"  - Avg Distance Error (60s): {err_60s:.2f} m\n" if err_60s else "  - Avg Distance Error (60s): N/A\n")
        f.write("-" * 50 + "\n")
    print(f"Results appended to {log_file}")
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(all_target_v * 3.6, label='GPS Ground Truth', color='blue', linewidth=2)
    plt.plot(all_pred_v * 3.6, label='DVSE Predicted Speed', color='red', linestyle='dashed', linewidth=2)
    plt.title("DVSE Smartphone-Only Speed Estimation vs GPS")
    plt.xlabel("Time (seconds into test split)")
    plt.ylabel("Vehicle Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dvse_test_plot.png')
    print("Saved plot to 'dvse_test_plot.png'")
    
if __name__ == '__main__':
    evaluate()
