"""
Utility script for evaluating model checkpoints and outputs.
"""

import torch
import numpy as np
import pickle
import sys
import math

sys.path.append("train")
from model.velocity.velocity_estimator import DVSEModel

EARTH_R = 6371000.0

def step_lat_lon(lat, lon, distance, heading_rad):
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

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def main():
    device = torch.device("cpu")
    
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    model = DVSEModel().to(device)
    model.load_state_dict(torch.load("train/dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    checkpoints = [15, 30, 60]
    drift_errors = {cp: [] for cp in checkpoints}
    baseline_drifts = {cp: [] for cp in checkpoints}
    
    with torch.no_grad():
        for item in test_items:
            raw_acc = item["blackout"]["raw_accel"]
            raw_gyro = item["blackout"]["raw_gyro"]
            gt_speeds = item["ground_truth"]["speeds_ms"]
            gt_headings = item["ground_truth"]["headings_rad"]
            
            start_lat = item["ground_truth"]["start_lat"]
            start_lon = item["ground_truth"]["start_lon"]
            vr_seed = item["context"]["vr_seed_ms"]
            
            scaled_acc = scalers["raw_accel"].transform(raw_acc)
            scaled_gyro = scalers["raw_gyro"].transform(raw_gyro)
            
            predicted_60s = []
            current_v_seed = vr_seed
            
            for k in range(12): # 12 leaps of 5 seconds
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
            
            # Now step through and calculate drift
            curr_gt_lat, curr_gt_lon = start_lat, start_lon
            curr_pr_lat, curr_pr_lon = start_lat, start_lon
            curr_bs_lat, curr_bs_lon = start_lat, start_lon
            
            for t in range(60):
                idx = (t+1)*10 - 1
                heading = gt_headings[idx]
                
                # Ground truth move
                curr_gt_lat, curr_gt_lon = step_lat_lon(curr_gt_lat, curr_gt_lon, gt_speeds[idx] * 1.0, heading)
                # Model predicted move
                curr_pr_lat, curr_pr_lon = step_lat_lon(curr_pr_lat, curr_pr_lon, predicted_60s[t] * 1.0, heading)
                # Baseline (Constant Velocity) move
                curr_bs_lat, curr_bs_lon = step_lat_lon(curr_bs_lat, curr_bs_lon, vr_seed * 1.0, heading)
                
                sec = t + 1
                if sec in checkpoints:
                    # Calculate Haversine Drift (Meters)
                    model_drift = haversine(curr_gt_lat, curr_gt_lon, curr_pr_lat, curr_pr_lon)
                    base_drift = haversine(curr_gt_lat, curr_gt_lon, curr_bs_lat, curr_bs_lon)
                    
                    drift_errors[sec].append(model_drift)
                    baseline_drifts[sec].append(base_drift)
                    
    print("\n--- SPATIAL DRIFT EVALUATION (MODEL VS BASELINE) ---")
    print(f"{'Time (s)':<10} | {'Model Drift':<15} | {'Baseline Drift':<15} | {'Improvement':<15}")
    print("-" * 65)
    for cp in checkpoints:
        m_drift = np.mean(drift_errors[cp])
        b_drift = np.mean(baseline_drifts[cp])
        imp = b_drift - m_drift
        print(f"{cp:<10} | {m_drift:<10.2f} m    | {b_drift:<10.2f} m    | {imp:+.2f} m")

if __name__ == "__main__":
    main()
