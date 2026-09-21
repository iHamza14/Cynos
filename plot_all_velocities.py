import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

# Ensure models can be imported
sys.path.append("train")
from models import DVSEModel

def plot_all_windows():
    """Plots the GPS vs Predicted Velocity for ALL windows and saves them in a folder."""
    device = torch.device("cpu")
    
    # 1. Load test dataset
    print("Loading test data...")
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    # 2. Load DVSE Model
    print("Loading DVSE model...")
    model = DVSEModel().to(device)
    model.load_state_dict(torch.load("train/dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    # Create output directory
    output_dir = "velocity_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for window_index, item in enumerate(test_items):
            print(f"Processing Window {window_index+1}/{len(test_items)}...")
            
            # 3. Extract the specific blackout window
            raw_acc = item["blackout"]["raw_accel"]
            raw_gyro = item["blackout"]["raw_gyro"]
            gt_speeds = item["ground_truth"]["speeds_ms"]
            vr_seed = item["context"]["vr_seed_ms"]
            
            scaled_acc = scalers["raw_accel"].transform(raw_acc)
            scaled_gyro = scalers["raw_gyro"].transform(raw_gyro)
            
            predicted_speeds = []
            current_v_seed = vr_seed
            
            # 4. Perform 5-second (50 sample) leaping inference
            for k in range(12):
                acc_chunk = scaled_acc[k*50 : (k+1)*50]
                gyro_chunk = scaled_gyro[k*50 : (k+1)*50]
                
                acc_t = torch.tensor(acc_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                gyro_t = torch.tensor(gyro_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                
                v0_t = torch.tensor([[current_v_seed]], dtype=torch.float32).to(device)
                vr_t = torch.full((1, 5, 1), current_v_seed, dtype=torch.float32).to(device)
                
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
            plt.figure(figsize=(10, 6))
            plt.plot(gt_speeds_1hz, label="Actual GPS Velocity", color="green", linewidth=3)
            plt.plot(predicted_speeds, label="Predicted Velocity (DVSE)", color="blue", linestyle="--", linewidth=3)
            
            session = item["metadata"].get("session", "unknown")
            plt.title(f"Velocity Comparison for Window {window_index+1} - {session} (MAE: {mae:.2f} m/s)", fontsize=16)
            plt.xlabel("Time into Blackout (seconds)", fontsize=14)
            plt.ylabel("Speed (m/s)", fontsize=14)
            plt.grid(True, alpha=0.4)
            plt.legend(fontsize=14)
            
            plt.tight_layout()
            file_path = os.path.join(output_dir, f"window_{window_index+1:02d}.png")
            plt.savefig(file_path, dpi=150)
            plt.close() # Free memory
            
    print(f"Success! All {len(test_items)} plots have been saved in the '{output_dir}/' folder.")

if __name__ == "__main__":
    plot_all_windows()
