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
    abs_errors = np.abs(all_gt - all_pred)
    
    plt.figure(figsize=(10, 6))
    
    # 1. Scatter plot of all absolute errors
    plt.scatter(all_gt, abs_errors, alpha=0.3, color='royalblue', s=15, label='Absolute Error per sample')
    
    # 2. Binned MAE
    # Create bins every 5 m/s
    bins = np.arange(0, np.max(all_gt) + 5, 5)
    bin_centers = []
    bin_maes = []
    
    for i in range(len(bins)-1):
        mask = (all_gt >= bins[i]) & (all_gt < bins[i+1])
        if np.any(mask):
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_maes.append(np.mean(abs_errors[mask]))
            
    # Plot binned MAE trendline
    plt.plot(bin_centers, bin_maes, color='red', linewidth=3, marker='o', markersize=8, label='Mean Absolute Error (Binned)')
    
    plt.title("Absolute Error vs. GPS Speed", fontsize=14)
    plt.xlabel("True GPS Speed (m/s)", fontsize=12)
    plt.ylabel("Absolute Error (m/s)", fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("mae_vs_speed.png", dpi=150)
    print("Plot saved to mae_vs_speed.png")

if __name__ == "__main__":
    main()
