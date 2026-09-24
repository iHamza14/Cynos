"""
Generates evaluation plots and metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.append("train")
from model.velocity.velocity_estimator import DVSEModel

def main():
    device = torch.device("cpu")
    
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    model = DVSEModel().to(device)
    model.load_state_dict(torch.load("train/dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, item in enumerate(test_items):
            if i >= len(axes):
                break
                
            raw_acc = item["blackout"]["raw_accel"]
            raw_gyro = item["blackout"]["raw_gyro"]
            speeds = item["ground_truth"]["speeds_ms"]
            vr_seed = item["context"]["vr_seed_ms"]
            
            scaled_acc = scalers["raw_accel"].transform(raw_acc)
            scaled_gyro = scalers["raw_gyro"].transform(raw_gyro)
            
            predicted_60s = []
            current_v_seed = vr_seed
            
            for k in range(12):
                acc_chunk = scaled_acc[k*50 : (k+1)*50]
                gyro_chunk = scaled_gyro[k*50 : (k+1)*50]
                
                acc_t = torch.tensor(acc_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                gyro_t = torch.tensor(gyro_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                
                v0_t = torch.tensor([[current_v_seed]], dtype=torch.float32).to(device)
                vr_t = torch.full((1, 5, 1), current_v_seed, dtype=torch.float32).to(device)
                
                _, v_pred, _, _ = model(acc_t, gyro_t, vr_t, v_0=v0_t)
                
                v_pred_np = v_pred.numpy()[0]
                predicted_60s.extend(v_pred_np)
                
                current_v_seed = v_pred_np[-1]
                
            predicted_60s = np.array(predicted_60s)
            gt_speeds_1hz = np.array([speeds[(j+1)*10 - 1] for j in range(60)])
            
            ax = axes[i]
            ax.plot(gt_speeds_1hz, label="Actual (GT)", color="green", linewidth=2)
            ax.plot(predicted_60s, label="Predicted", color="blue", linestyle="--", linewidth=2)
            
            session = item["metadata"].get("session", "unknown")
            mae = np.mean(np.abs(predicted_60s - gt_speeds_1hz))
            ax.set_title(f"Window {i+1} ({session})\nMAE: {mae:.2f} m/s", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
    
    for j in range(len(test_items), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig("velocity_predictions.png", dpi=150)
    print("Plot saved to velocity_predictions.png")

if __name__ == "__main__":
    main()
