"""
Generates evaluation plots and metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Ensure models can be imported
import os
if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))
from model.velocity.velocity_estimator import DVSEModel

def plot_velocity_for_window(window_index=12):
    """Plots the GPS (Ground Truth) vs Predicted Velocity for a specific window."""
    device = torch.device("cpu")
    
    # 1. Load test dataset
    print("Loading test data...")
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    # 2. Load DVSE Model
    print("Loading DVSE model...")
    model = DVSEModel(scalers).to(device)
    model.load_state_dict(torch.load("dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    # 3. Extract the specific blackout window
    item = test_items[window_index]
    raw_acc = item["blackout"]["raw_accel"]
    raw_gyro = item["blackout"]["raw_gyro"]
    gt_speeds = item["ground_truth"]["speeds_ms"]
    vr_seed = item["context"]["vr_seed_ms"]
    
    
    predicted_speeds = []
    current_v_seed = vr_seed
    
    # 4. Perform 5-second (50 sample) leaping inference
    print(f"Running inference for Window {window_index+1}...")
    with torch.no_grad():
        for k in range(6):
            acc_chunk = raw_acc[k*100 : (k+1)*100]
            gyro_chunk = raw_gyro[k*100 : (k+1)*100]
            
            acc_t = torch.tensor(acc_chunk, dtype=torch.float32).unsqueeze(0).to(device)
            gyro_t = torch.tensor(gyro_chunk, dtype=torch.float32).unsqueeze(0).to(device)
            
            v0_t = torch.tensor([[current_v_seed]], dtype=torch.float32).to(device)
            vr_t = torch.full((1, 10, 1), current_v_seed, dtype=torch.float32).to(device)
            
            _, v_pred, _, _ = model(acc_t, gyro_t, vr_t, v_0=v0_t)
            v_pred_np = v_pred.numpy()[0]
            predicted_speeds.extend(v_pred_np)
            
            # Leap forward: the last prediction becomes the new reference scalar
            current_v_seed = v_pred_np[-1]
            
    predicted_speeds = np.array(predicted_speeds)
    
    # Extract ground truth at 1Hz
    gt_speeds_1hz = np.array([gt_speeds[(j+1)*10 - 1] for j in range(60)])
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted_speeds - gt_speeds_1hz))
    
    # 5. Plot the comparison
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(gt_speeds_1hz, label="Actual GPS Velocity", color="green", linewidth=3)
    plt.plot(predicted_speeds, label="Predicted Velocity (DVSE)", color="blue", linestyle="--", linewidth=3)
    
    plt.title(f"Velocity Comparison for Window {window_index+1} (MAE: {mae:.2f} m/s)", fontsize=16)
    plt.xlabel("Time into Blackout (seconds)", fontsize=14)
    plt.ylabel("Speed (m/s)", fontsize=14)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"evaluation/plots/velocity_comparison_window_{window_index+1}.png", dpi=150)
    print(f"Success! Saved as evaluation/plots/velocity_comparison_window_{window_index+1}.png")

if __name__ == "__main__":
    plot_velocity_for_window(0)  # 12 is Window 13
