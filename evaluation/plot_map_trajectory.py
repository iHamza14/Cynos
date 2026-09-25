"""
Generates evaluation plots and metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import math

import os
if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))
from model.velocity.velocity_estimator import DVSEModel

EARTH_R = 6371000.0

def step_lat_lon(lat, lon, distance, heading_rad):
    """Move lat/lon by distance (meters) along heading (radians)"""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance / EARTH_R) +
        math.cos(lat_rad) * math.sin(distance / EARTH_R) * math.cos(heading_rad)
    )
    
    new_lon_rad = lon_rad + math.atan2(
        math.sin(heading_rad) * math.sin(distance / EARTH_R) * math.cos(lat_rad),
        math.cos(distance / EARTH_R) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def main():
    device = torch.device("cpu")
    
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    model = DVSEModel(scalers).to(device)
    model.load_state_dict(torch.load("dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    # Pick the first window
    item = test_items[12]
    
    raw_acc = item["blackout"]["raw_accel"]
    raw_gyro = item["blackout"]["raw_gyro"]
    gt_speeds = item["ground_truth"]["speeds_ms"]
    gt_headings = item["ground_truth"]["headings_rad"]
    
    start_lat = item["ground_truth"]["start_lat"]
    start_lon = item["ground_truth"]["start_lon"]
    vr_seed = item["context"]["vr_seed_ms"]
    
    
    predicted_60s = []
    current_v_seed = vr_seed
    
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
            predicted_60s.extend(v_pred_np)
            current_v_seed = v_pred_np[-1]
            
    # Calculate Lat/Lon paths (1Hz)
    gt_path = [(start_lat, start_lon)]
    pred_path = [(start_lat, start_lon)]
    
    curr_gt_lat, curr_gt_lon = start_lat, start_lon
    curr_pr_lat, curr_pr_lon = start_lat, start_lon
    
    # Plotting X/Y coords for matplotlib (relative meters)
    gt_xy = [(0, 0)]
    pr_xy = [(0, 0)]
    curr_gt_x, curr_gt_y = 0, 0
    curr_pr_x, curr_pr_y = 0, 0
    
    for t in range(60):
        # 1Hz sampling
        idx = (t+1)*10 - 1
        heading = gt_headings[idx] # Use GT heading to isolate speed model accuracy
        
        # Ground truth move
        v_gt = gt_speeds[idx]
        curr_gt_lat, curr_gt_lon = step_lat_lon(curr_gt_lat, curr_gt_lon, v_gt * 1.0, heading)
        gt_path.append((curr_gt_lat, curr_gt_lon))
        
        curr_gt_x += v_gt * math.sin(heading)
        curr_gt_y += v_gt * math.cos(heading)
        gt_xy.append((curr_gt_x, curr_gt_y))
        
        # Predicted move
        v_pr = predicted_60s[t]
        curr_pr_lat, curr_pr_lon = step_lat_lon(curr_pr_lat, curr_pr_lon, v_pr * 1.0, heading)
        pred_path.append((curr_pr_lat, curr_pr_lon))
        
        curr_pr_x += v_pr * math.sin(heading)
        curr_pr_y += v_pr * math.cos(heading)
        pr_xy.append((curr_pr_x, curr_pr_y))
        
    # --- 1. MATPLOTLIB 2D MAP ---
    gt_xy = np.array(gt_xy)
    pr_xy = np.array(pr_xy)
    
    plt.figure(figsize=(10, 8))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="Ground Truth Path", color="green", linewidth=3)
    plt.plot(pr_xy[:, 0], pr_xy[:, 1], label="Predicted DVSE Path", color="blue", linestyle="--", linewidth=3)
    
    plt.scatter([0], [0], color="black", s=100, label="Blackout Start", zorder=5)
    
    plt.title("Map Trajectory for Window 13 (Speed Model + True Heading)", fontsize=14)
    plt.xlabel("East Displacement (meters)", fontsize=12)
    plt.ylabel("North Displacement (meters)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axis("equal")
    
    plt.tight_layout()
    os.makedirs("evaluation/plots", exist_ok=True)
    plt.savefig("evaluation/plots/map_trajectory_2d.png", dpi=150)
    print("Saved 2D map plot to evaluation/plots/map_trajectory_2d.png")
    
    # --- 2. FOLIUM INTERACTIVE MAP ---
    try:
        import folium
        m = folium.Map(location=[start_lat, start_lon], zoom_start=18)
        
        folium.PolyLine(gt_path, color="green", weight=4, opacity=0.8, tooltip="Ground Truth").add_to(m)
        folium.PolyLine(pred_path, color="blue", weight=4, dash_array="10", opacity=0.8, tooltip="Predicted Path").add_to(m)
        
        folium.Marker([start_lat, start_lon], popup="GPS Lost Here", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        
        m.save("evaluation/plots/map_trajectory.html")
        print("Saved interactive map to evaluation/plots/map_trajectory.html")
    except ImportError:
        print("Folium not installed, skipping interactive HTML map.")

if __name__ == "__main__":
    main()
