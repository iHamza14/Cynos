import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("train")
from train_speed_model import DVSEDataset
from models import DVSEModel
from torch.utils.data import DataLoader

def main():
    device = torch.device("cpu")
    
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    dataset = DVSEDataset(test_items, scalers)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = DVSEModel().to(device)
    model.load_state_dict(torch.load("train/dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for batch in loader:
            acc = batch["acc_window"].to(device)
            gyro = batch["gyro_window"].to(device)
            vr = batch["vr_seq"].to(device)
            v0 = batch["v_0"].to(device)
            
            gt_speeds = batch["target_speeds"].numpy()[0]
            
            _, pred_v, _, _ = model(acc, gyro, vr, v_0=v0)
            pred_speeds = pred_v.numpy()[0]
            
            all_gt.extend(gt_speeds)
            all_pred.extend(pred_speeds)
            
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_gt - all_pred))
    corr = np.corrcoef(all_gt, all_pred)[0, 1]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(all_gt, all_pred, alpha=0.5, color='royalblue', edgecolors='none', s=20)
    
    # Line of perfect prediction (y = x)
    min_val = min(np.min(all_gt), np.min(all_pred))
    max_val = max(np.max(all_gt), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Perfect Prediction (y=x)")
    
    plt.title(f"Predicted vs True Velocity (Test Set)\nMAE: {mae:.2f} m/s | Correlation: {corr:.3f}", fontsize=14)
    plt.xlabel("True Velocity (m/s)", fontsize=12)
    plt.ylabel("Predicted Velocity (m/s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("velocity_scatter.png", dpi=150)
    print("Scatter plot saved to velocity_scatter.png")

if __name__ == "__main__":
    main()
