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

# Add models to path
sys.path.append(os.path.abspath("models"))
from models import DVSEModel, GyroTCN
from dvse_physics import euler_to_rotation_matrix, physics_velocity_update
from viterbi_snapper import ViterbiSnapper, SimpleRoadGraph, haversine

HZ = 10.0
EARTH_R = 6371000.0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("../data"))
    p.add_argument("--test-file", type=str, default="test_blackout_windows.pkl")
    p.add_argument("--dvse-weights", type=Path, default=Path("dvse_output/best_dvse.pt"))
    p.add_argument("--gyro-weights", type=Path, default=Path("gyro_tcn_output/best_gyro_tcn_processed.pt"))
    p.add_argument("--output", type=str, default="live_simulation_log.json")
    p.add_argument("--episode", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu")

    # 1. Load Data
    test_path = args.data_dir / args.test_file
    scalers_path = args.data_dir / args.test_file.replace("test_blackout_windows", "scalers")
    with test_path.open("rb") as f:
        test_items = pickle.load(f)
    with scalers_path.open("rb") as f:
        scalers = pickle.load(f)
        
    item = test_items[args.episode]
    acc_scaler = scalers["raw_accel"]
    gyro_scaler = scalers["raw_gyro"]

    # 2. Load Models
    dvse_model = DVSEModel().to(device)
    dvse_model.load_state_dict(torch.load(args.dvse_weights, map_location=device, weights_only=False))
    dvse_model.eval()

    gyro_model = GyroTCN().to(device)
    gyro_ckpt = torch.load(args.gyro_weights, map_location=device, weights_only=False)
    gyro_model.load_state_dict(gyro_ckpt["model_state"])
    gyro_model.eval()
    
    y_mean = gyro_ckpt.get("target_mean", 0.0)
    y_std = gyro_ckpt.get("target_std", 1.0)

    # 3. Initialize Streaming State
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

    # Setup Offline Road Graph for Viterbi (using GT as mock road)
    gt_coords_full = [(start_lat, start_lon)]
    for i in range(len(gt_disp)):
        g_n = gt_disp[i, 0]
        g_e = gt_disp[i, 1]
        l_rad = np.radians(start_lat) + (g_n / EARTH_R)
        ln_rad = np.radians(start_lon) + (g_e / (EARTH_R * np.cos(np.radians(start_lat))))
        gt_coords_full.append((np.degrees(l_rad), np.degrees(ln_rad)))

    road_graph = SimpleRoadGraph([gt_coords_full])
    snapper = ViterbiSnapper(sigma_z=200.0, beta=20.0)
    
    accumulated_dr = [(start_lat, start_lon)]
    current_snap_lat = start_lat
    current_snap_lon = start_lon

    simulation_log = []
    
    # 4. Stream and Simulate (10 Hz)
    total_steps = len(blackout_acc)
    
    with torch.no_grad():
        for i in range(total_steps):
            time_sec = (i + 1) / HZ
            
            imu_buffer_acc.append(blackout_acc[i])
            imu_buffer_gyr.append(blackout_gyr[i])
            
            acc_window = np.array(imu_buffer_acc, dtype=np.float32)
            gyr_window = np.array(imu_buffer_gyr, dtype=np.float32)
            
            acc_scaled = acc_scaler.transform(acc_window)
            gyr_scaled = gyro_scaler.transform(gyr_window)
            
            # STAGE B: Heading
            gyro_tensor = torch.from_numpy(gyr_scaled).unsqueeze(0).to(device)
            pred_yaw_rate_norm = gyro_model(gyro_tensor).squeeze(0).cpu().numpy()
            
            current_yaw_rate_norm = pred_yaw_rate_norm[-1]
            current_yaw_rate_dps = current_yaw_rate_norm * y_std + y_mean
            
            current_heading_deg += current_yaw_rate_dps * (1.0 / HZ)
            current_heading_deg %= 360.0
            current_heading_rad = np.radians(current_heading_deg)
            
            # STAGE A & C: Velocity & Navigate (1 Hz)
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
                
                # --- LIVE VITERBI SNAPPING ---
                # A navigation engine does this retrospectively on the fly
                accumulated_dr.append((current_lat, current_lon))
                matched_path = snapper.snap(
                    accumulated_dr,
                    get_candidates_fn=lambda obs: road_graph.get_candidates(obs, radius=500.0),
                    get_route_dist_fn=lambda c1, c2: road_graph.route_distance(c1, c2)
                )
                if matched_path:
                    current_snap_lat = matched_path[-1]['lat']
                    current_snap_lon = matched_path[-1]['lon']
                else:
                    current_snap_lat = current_lat
                    current_snap_lon = current_lon
            
            # Ground Truth Log
            gt_n = gt_disp[i, 0]
            gt_e = gt_disp[i, 1]
            gt_lat_rad = np.radians(start_lat) + (gt_n / EARTH_R)
            gt_lon_rad = np.radians(start_lon) + (gt_e / (EARTH_R * np.cos(np.radians(start_lat))))
            true_lat = np.degrees(gt_lat_rad)
            true_lon = np.degrees(gt_lon_rad)
            
            raw_err = haversine(current_lat, current_lon, true_lat, true_lon)
            snap_err = haversine(current_snap_lat, current_snap_lon, true_lat, true_lon)
            
            simulation_log.append({
                "time": float(time_sec),
                "dr_lat": float(current_lat),
                "dr_lon": float(current_lon),
                "snap_lat": float(current_snap_lat),
                "snap_lon": float(current_snap_lon),
                "dr_heading": float(current_heading_deg),
                "dr_speed": float(current_velocity_ms),
                "gt_lat": float(true_lat),
                "gt_lon": float(true_lon),
                "gt_heading": float(gt_headings[i]),
                "raw_err": float(raw_err),
                "snap_err": float(snap_err),
                "blackout": True
            })
            
            if (i+1) % 100 == 0:
                print(f"Simulated {time_sec:.1f}s / {total_steps/HZ:.1f}s")
    
    with open(args.output, "w") as f:
        json.dump({
            "anchor": {"lat": start_lat, "lon": start_lon, "heading": init_heading_deg, "speed": vr_seed},
            "trajectory": simulation_log
        }, f)
    print(f"Saved real-time simulation trace to {args.output}")

if __name__ == "__main__":
    main()
