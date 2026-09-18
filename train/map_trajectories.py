#!/usr/bin/env python3
"""
Map Trajectory and Drift Calculator

Generates interactive Folium maps of the Dead-Reckoning (DR) vs Ground Truth (GT) 
trajectories, and calculates the drift (position error) along the path at 1 Hz.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import folium

from models import DVSEModel, GyroTCN

HZ = 10.0
EARTH_R = 6371000.0

def parse_args():
    p = argparse.ArgumentParser(description="Map Trajectories and Calculate Drift")
    p.add_argument("--data-dir", type=Path, default=Path("../data"))
    p.add_argument("--test-file", default="test_blackout_windows.pkl")
    p.add_argument("--dvse-weights", type=Path, default=Path("dvse_output/best_dvse.pt"))
    p.add_argument("--gyro-weights", type=Path, default=Path("gyro_tcn_output/best_gyro_tcn_processed.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("map_output"))
    p.add_argument("--device", default="auto")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to plot on maps")
    return p.parse_args()

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def en_to_latlon(E, N, start_lat, start_lon):
    """
    Converts relative East/North displacements to absolute lat/lon coordinates.
    """
    start_lat_rad = np.radians(start_lat)
    
    # Delta lat = N / R
    dlat_rad = N / EARTH_R
    lat_rad = start_lat_rad + dlat_rad
    lat_deg = np.degrees(lat_rad)
    
    # Delta lon = E / (R * cos(lat))
    # We use a simplified local projection where we divide by cos(start_lat_rad) or cos(mid_lat)
    dlon_rad = E / (EARTH_R * np.cos(start_lat_rad))
    lon_rad = np.radians(start_lon) + dlon_rad
    lon_deg = np.degrees(lon_rad)
    
    return lat_deg, lon_deg

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    test_path = args.data_dir / args.test_file
    scalers_path = args.data_dir / "scalers.pkl"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_items = load_pickle(test_path)
    scalers = load_pickle(scalers_path) if scalers_path.exists() else None

    # Load Models
    dvse_model = DVSEModel().to(device)
    if args.dvse_weights.exists():
        dvse_model.load_state_dict(torch.load(args.dvse_weights, map_location=device, weights_only=False))
        print(f"Loaded DVSE velocity weights from {args.dvse_weights}")
    else:
        print(f"Warning: {args.dvse_weights} not found. Running with uninitialized weights.")
    dvse_model.eval()

    gyro_model = GyroTCN().to(device)
    gyro_ckpt = None
    if args.gyro_weights.exists():
        gyro_ckpt = torch.load(args.gyro_weights, map_location=device, weights_only=False)
        gyro_model.load_state_dict(gyro_ckpt["model_state"])
        print(f"Loaded Gyro TCN weights from {args.gyro_weights}")
    else:
        print(f"Warning: {args.gyro_weights} not found. Running with uninitialized weights.")
    gyro_model.eval()

    acc_scaler = scalers["raw_accel"] if scalers else None
    gyro_scaler = scalers["raw_gyro"] if scalers else None

    y_mean = gyro_ckpt.get("target_mean", 0.0) if gyro_ckpt else 0.0
    y_std = gyro_ckpt.get("target_std", 1.0) if gyro_ckpt else 1.0

    all_drift_records = []

    with torch.no_grad():
        for ep_idx, item in enumerate(test_items):
            raw_accel = np.asarray(item["blackout"]["raw_accel"], dtype=np.float32)
            raw_gyro = np.asarray(item["blackout"]["raw_gyro"], dtype=np.float32)
            gt_disp = np.asarray(item["ground_truth"]["cumulative_displacement_ne_m"], dtype=np.float32)
            vr_seed = float(item["context"]["vr_seed_ms"])
            init_heading_deg = float(item["ground_truth"]["headings_deg"][0])
            start_lat = float(item["ground_truth"]["start_lat"])
            start_lon = float(item["ground_truth"]["start_lon"])

            # 1. Scale inputs
            acc_scaled = acc_scaler.transform(raw_accel) if acc_scaler else raw_accel
            gyro_scaled = gyro_scaler.transform(raw_gyro) if gyro_scaler else raw_gyro

            # 2. Heading Inference (10 Hz)
            gyro_tensor = torch.from_numpy(gyro_scaled).unsqueeze(0).to(device)
            pred_yaw_rate_norm = gyro_model(gyro_tensor).squeeze(0).cpu().numpy()
            pred_yaw_rate_dps = pred_yaw_rate_norm * y_std + y_mean
            
            dt_10hz = 1.0 / HZ
            heading_change_deg = np.cumsum(pred_yaw_rate_dps * dt_10hz)
            pred_heading_deg = (init_heading_deg + heading_change_deg) % 360.0
            pred_heading_rad = np.radians(pred_heading_deg)

            # 3. Velocity Inference (1 Hz sequence)
            T = len(raw_accel) // int(HZ)
            vr_seq = np.zeros((1, T, 1), dtype=np.float32)
            vr_seq[0, 0, 0] = vr_seed
            
            acc_t = torch.from_numpy(acc_scaled).unsqueeze(0).to(device)
            gyro_t = torch.from_numpy(gyro_scaled).unsqueeze(0).to(device)
            vr_t = torch.from_numpy(vr_seq).to(device)
            v0_t = torch.tensor([[vr_seed]], dtype=torch.float32).to(device)

            delta_v, pred_v_seq, _, _ = dvse_model(acc_t, gyro_t, vr_t, v_0=v0_t, hz=int(HZ))
            pred_v = pred_v_seq.squeeze(0).cpu().numpy() # [T] (1 Hz)

            # 4. Dead-Reckoning Navigation Fusion (1 Hz)
            psi_1hz = np.array([pred_heading_rad[min((i + 1) * int(HZ) - 1, len(pred_heading_rad) - 1)] for i in range(T)])
            
            dt_1s = 1.0
            delta_s = pred_v * dt_1s
            delta_E = delta_s * np.sin(psi_1hz)
            delta_N = delta_s * np.cos(psi_1hz)

            dr_E = np.cumsum(delta_E)
            dr_N = np.cumsum(delta_N)

            gt_1hz_indices = [min((i + 1) * int(HZ) - 1, len(gt_disp) - 1) for i in range(T)]
            gt_N = gt_disp[gt_1hz_indices, 0]
            gt_E = gt_disp[gt_1hz_indices, 1]

            # 5. Calculate Drift
            drift_m = np.sqrt((dr_E - gt_E) ** 2 + (dr_N - gt_N) ** 2)

            for t_sec in range(T):
                all_drift_records.append({
                    "episode_idx": ep_idx,
                    "time_sec": t_sec + 1,
                    "dr_E_m": dr_E[t_sec],
                    "dr_N_m": dr_N[t_sec],
                    "gt_E_m": gt_E[t_sec],
                    "gt_N_m": gt_N[t_sec],
                    "drift_m": drift_m[t_sec]
                })

            # 6. Map Plotting (Limit to a few episodes)
            if ep_idx < args.episodes:
                # Convert DR and GT to lat/lon
                dr_lat, dr_lon = en_to_latlon(dr_E, dr_N, start_lat, start_lon)
                gt_lat, gt_lon = en_to_latlon(gt_E, gt_N, start_lat, start_lon)

                # Insert start point (0, 0 displacement)
                dr_coords = [(start_lat, start_lon)] + list(zip(dr_lat, dr_lon))
                gt_coords = [(start_lat, start_lon)] + list(zip(gt_lat, gt_lon))

                # Create Map centered at start point
                m = folium.Map(location=[start_lat, start_lon], zoom_start=18, tiles="OpenStreetMap")

                # Add GT trajectory (Blue)
                folium.PolyLine(
                    locations=gt_coords,
                    color="blue",
                    weight=4,
                    opacity=0.8,
                    tooltip="Ground Truth"
                ).add_to(m)

                # Add DR trajectory (Red)
                folium.PolyLine(
                    locations=dr_coords,
                    color="red",
                    weight=4,
                    opacity=0.8,
                    tooltip="Cynos DR Model"
                ).add_to(m)
                
                # Add markers for start and end
                folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(dr_coords[-1], popup=f"DR End (Drift: {drift_m[-1]:.1f}m)", icon=folium.Icon(color="red")).add_to(m)
                folium.Marker(gt_coords[-1], popup="GT End", icon=folium.Icon(color="blue")).add_to(m)

                map_file = args.output_dir / f"episode_{ep_idx}_map.html"
                m.save(str(map_file))

    # Save drift records
    drift_df = pd.DataFrame(all_drift_records)
    csv_file = args.output_dir / "detailed_drift_1hz.csv"
    drift_df.to_csv(csv_file, index=False)

    print("\n" + "=" * 60)
    print("DRIFT CALCULATION & MAP GENERATION COMPLETE")
    print("=" * 60)
    print(f"Drift metrics saved to: {csv_file}")
    print(f"Maps saved to: {args.output_dir}/")
    
    # Calculate summary drift at end of blackout (60s)
    final_drifts = drift_df[drift_df["time_sec"] == 60]["drift_m"]
    if len(final_drifts) > 0:
        print("\nSummary Drift after 60s blackout:")
        print(f"  Mean:   {final_drifts.mean():.2f} m")
        print(f"  Median: {final_drifts.median():.2f} m")
        print(f"  P90:    {final_drifts.quantile(0.90):.2f} m")
        print(f"  Max:    {final_drifts.max():.2f} m")

if __name__ == "__main__":
    main()
