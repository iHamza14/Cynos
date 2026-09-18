import argparse
import pickle
import json
import torch
import numpy as np
from pathlib import Path
from collections import deque
import sys
import os
import math

sys.path.append(os.path.abspath("models"))
from models import DVSEModel, GyroTCN
from dvse_physics import euler_to_rotation_matrix, physics_velocity_update
from viterbi_snapper import ViterbiSnapper, SimpleRoadGraph, haversine

HZ = 10.0
EARTH_R = 6371000.0

def main():
    device = torch.device("cpu")
    test_path = Path("../data/test_blackout_windows.pkl")
    scalers_path = Path("../data/scalers.pkl")
    
    with test_path.open("rb") as f:
        test_items = pickle.load(f)
    with scalers_path.open("rb") as f:
        scalers = pickle.load(f)
        
    acc_scaler = scalers["raw_accel"]
    gyro_scaler = scalers["raw_gyro"]

    dvse_model = DVSEModel().to(device)
    dvse_model.load_state_dict(torch.load("dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    dvse_model.eval()

    gyro_model = GyroTCN().to(device)
    gyro_ckpt = torch.load("gyro_tcn_output/best_gyro_tcn_processed.pt", map_location=device, weights_only=False)
    gyro_model.load_state_dict(gyro_ckpt["model_state"])
    gyro_model.eval()
    
    y_mean = gyro_ckpt.get("target_mean", 0.0)
    y_std = gyro_ckpt.get("target_std", 1.0)

    all_simulations = []

    print(f"Simulating {len(test_items)} episodes concurrently...")

    with torch.no_grad():
        for ep_idx, item in enumerate(test_items[:15]):
            print(f"Processing Episode {ep_idx+1}/{len(test_items)}...", flush=True)
            
            ctx_acc = item["context"]["raw_accel"]
            ctx_gyr = item["context"]["raw_gyro"]
            
            imu_buffer_acc = deque(ctx_acc, maxlen=100)
            imu_buffer_gyr = deque(ctx_gyr, maxlen=100)

            vr_seed = float(item["context"]["vr_seed_ms"])
            start_lat = float(item["ground_truth"]["start_lat"])
            start_lon = float(item["ground_truth"]["start_lon"])
            init_heading_deg = float(item["ground_truth"]["headings_deg"][0])

            current_lat = start_lat
            current_lon = start_lon
            current_heading_deg = init_heading_deg
            current_velocity_ms = vr_seed

            blackout_acc = item["blackout"]["raw_accel"]
            blackout_gyr = item["blackout"]["raw_gyro"]
            gt_disp = item["ground_truth"]["cumulative_displacement_ne_m"]
            gt_headings = item["ground_truth"]["headings_deg"]

            gt_coords_full = [(start_lat, start_lon)]
            for i in range(len(gt_disp)):
                g_n = gt_disp[i, 0]
                g_e = gt_disp[i, 1]
                l_rad = np.radians(start_lat) + (g_n / EARTH_R)
                ln_rad = np.radians(start_lon) + (g_e / (EARTH_R * np.cos(np.radians(start_lat))))
                gt_coords_full.append((np.degrees(l_rad), np.degrees(ln_rad)))

            road_graph = SimpleRoadGraph([gt_coords_full])
            snapper = ViterbiSnapper(sigma_z=200.0, beta=20.0)
            
            dr_coords = [(start_lat, start_lon)]
            headings_log = []
            speeds_log = []

            total_steps = len(blackout_acc)
            
            for i in range(total_steps):
                time_sec = (i + 1) / HZ
                
                imu_buffer_acc.append(blackout_acc[i])
                imu_buffer_gyr.append(blackout_gyr[i])
                
                acc_window = np.array(imu_buffer_acc, dtype=np.float32)
                gyr_window = np.array(imu_buffer_gyr, dtype=np.float32)
                
                acc_scaled = acc_scaler.transform(acc_window)
                gyr_scaled = gyro_scaler.transform(gyr_window)
                
                gyro_tensor = torch.from_numpy(gyr_scaled).unsqueeze(0).to(device)
                pred_yaw_rate_norm = gyro_model(gyro_tensor).squeeze(0).cpu().numpy()
                
                current_yaw_rate_norm = pred_yaw_rate_norm[-1]
                current_yaw_rate_dps = current_yaw_rate_norm * y_std + y_mean
                
                current_heading_deg += current_yaw_rate_dps * (1.0 / HZ)
                current_heading_deg %= 360.0
                current_heading_rad = np.radians(current_heading_deg)
                
                if i % 10 == 9:
                    acc_t = torch.from_numpy(acc_scaled).unsqueeze(0).to(device)
                    gyr_t = torch.from_numpy(gyr_scaled).unsqueeze(0).to(device)
                    
                    vr_seq = np.zeros((1, 10, 1), dtype=np.float32)
                    seed_idx = 9 - (i // 10)
                    if seed_idx >= 0 and seed_idx < 10:
                        vr_seq[0, seed_idx, 0] = vr_seed
                    
                    vr_t = torch.from_numpy(vr_seq).to(device)
                    v0_t = torch.tensor([[current_velocity_ms]], dtype=torch.float32).to(device)
                    
                    delta_v, _, _, _ = dvse_model(acc_t, gyr_t, vr_t, v_0=v0_t, hz=int(HZ))
                    
                    v_inc = delta_v[0, -1].item()
                    current_velocity_ms += v_inc
                    
                    dist = current_velocity_ms * 1.0
                    
                    start_lat_rad = np.radians(current_lat)
                    d_north = dist * np.cos(current_heading_rad)
                    d_east = dist * np.sin(current_heading_rad)
                    
                    dlat_rad = d_north / EARTH_R
                    current_lat += np.degrees(dlat_rad)
                    
                    dlon_rad = d_east / (EARTH_R * np.cos(start_lat_rad))
                    current_lon += np.degrees(dlon_rad)
                    
                    dr_coords.append((current_lat, current_lon))
                
                headings_log.append(current_heading_deg)
                speeds_log.append(current_velocity_ms)

            # Snap the complete DR path once (O(N) instead of O(N^2))
            matched_path = snapper.snap(
                dr_coords,
                get_candidates_fn=lambda obs: road_graph.get_candidates(obs, radius=500.0),
                get_route_dist_fn=lambda c1, c2: road_graph.route_distance(c1, c2)
            )
            
            if matched_path and len(matched_path) == len(dr_coords):
                snapped_coords = [(m['lat'], m['lon']) for m in matched_path]
            else:
                snapped_coords = dr_coords

            simulation_log = []
            for i in range(total_steps):
                time_sec = (i + 1) / HZ
                idx_1hz = (i + 1) // 10
                
                c_lat, c_lon = dr_coords[idx_1hz]
                s_lat, s_lon = snapped_coords[idx_1hz]
                
                true_lat, true_lon = gt_coords_full[i + 1]
                
                raw_err = haversine(c_lat, c_lon, true_lat, true_lon)
                snap_err = haversine(s_lat, s_lon, true_lat, true_lon)
                
                simulation_log.append({
                    "time": float(time_sec),
                    "dr_lat": float(c_lat),
                    "dr_lon": float(c_lon),
                    "snap_lat": float(s_lat),
                    "snap_lon": float(s_lon),
                    "dr_heading": float(headings_log[i]),
                    "dr_speed": float(speeds_log[i]),
                    "gt_lat": float(true_lat),
                    "gt_lon": float(true_lon),
                    "gt_heading": float(gt_headings[i]),
                    "raw_err": float(raw_err),
                    "snap_err": float(snap_err)
                })
                
            all_simulations.append({
                "episode": ep_idx,
                "anchor": {"lat": start_lat, "lon": start_lon, "heading": init_heading_deg, "speed": vr_seed},
                "trajectory": simulation_log
            })
            
    with open("live_simulation_all.json", "w") as f:
        json.dump(all_simulations, f)
    print("Saved bulk simulation to live_simulation_all.json")

if __name__ == "__main__":
    main()
