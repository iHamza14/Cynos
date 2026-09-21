import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Import the dataset and model
import sys
sys.path.append("train")
from train_speed_model import DVSEDataset, set_seed
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
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= len(axes):
                break
                
            acc = batch["acc_window"].to(device)
            gyro = batch["gyro_window"].to(device)
            vr = batch["vr_seq"].to(device)
            v0 = batch["v_0"].to(device)
            
            gt_speeds = batch["target_speeds"].numpy()[0]
            
            _, pred_v, _, _ = model(acc, gyro, vr, v_0=v0)
            pred_speeds = pred_v.numpy()[0]
            
            ax = axes[i]
            ax.plot(gt_speeds, label="Actual (GT)", color="green", linewidth=2)
            ax.plot(pred_speeds, label="Predicted", color="blue", linestyle="--", linewidth=2)
            
            # Metadata
            session = test_items[i]["metadata"].get("session", "unknown")
            mae = np.mean(np.abs(pred_speeds - gt_speeds))
            ax.set_title(f"Window {i+1} ({session})\nMAE: {mae:.2f} m/s", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
    
    # Hide unused subplots
    for j in range(len(test_items), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig("velocity_predictions.png", dpi=150)
    print("Plot saved to velocity_predictions.png")

if __name__ == "__main__":
    main()
