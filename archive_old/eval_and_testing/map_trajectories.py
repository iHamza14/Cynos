#!/usr/bin/env python3
"""
Map Trajectory and Drift Calculator (Integrated with Viterbi Snapper)

Generates interactive Folium maps of the Dead-Reckoning (DR) vs Ground Truth (GT) 
trajectories, snaps the DR trajectory to the road using a Viterbi HMM, 
and calculates the drift for both raw DR and Map-Matched paths.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import folium
import math

from models import DVSEModel, GyroTCN
from viterbi_snapper import ViterbiSnapper, SimpleRoadGraph, haversine

HZ = 10.0
EARTH_R = 6371000.0


def parse_args():
    p = argparse.ArgumentParser(description="Map Trajectories and Calculate Drift")
    p.add_argument("--data-dir", type=Path, default=Path("../data"))
    p.add_argument("--test-file", type=str, default="driver_a_test_blackout_windows.pkl")
    p.add_argument("--dvse-weights", type=Path, default=Path("dvse_output/best_dvse.pt"))
    p.add_argument("--gyro-weights", type=Path, default=Path("gyro_tcn_output/best_gyro_tcn_processed.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("map_output_driver_a"))
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
    dlat_rad = N / EARTH_R
    lat_rad = start_lat_rad + dlat_rad
    lat_deg = np.degrees(lat_rad)
    
    dlon_rad = E / (EARTH_R * np.cos(start_lat_rad))
    lon_rad = np.radians(start_lon) + dlon_rad
    lon_deg = np.degrees(lon_rad)
    
    return lat_deg, lon_deg


# ---------------------------------------------------------
# Plots all test windows (episodes) on a single interactive map.
# Organizes the paths into toggleable layers to prevent clutter.
# ---------------------------------------------------------
def generate_global_map(all_gt_paths, all_dr_paths, all_snapped_paths, output_path):
    if not all_gt_paths:
        return
        
    print(f"\nGenerating global map with {len(all_gt_paths)} total trajectories...")
    
    # Center map on the start position of the very first episode
    start_pos = all_gt_paths[0][0]
    m = folium.Map(location=start_pos, zoom_start=13, tiles="OpenStreetMap")
    
    # Create FeatureGroups so the user can toggle layers on and off
    fg_gt = folium.FeatureGroup(name="Ground Truth (Blue)", show=True)
    fg_dr = folium.FeatureGroup(name="Raw DR (Red)", show=False) # Hidden by default to reduce clutter
    fg_snap = folium.FeatureGroup(name="Viterbi Snapped (Green)", show=True)
    
    # Add all paths to their respective layers
    for gt in all_gt_paths:
        folium.PolyLine(locations=gt, color="blue", weight=3, opacity=0.5).add_to(fg_gt)
        # Add a small marker for the start of each episode
        folium.CircleMarker(location=gt[0], radius=3, color="blue", fill=True).add_to(fg_gt)
        
    for dr in all_dr_paths:
        folium.PolyLine(locations=dr, color="red", weight=3, opacity=0.5, dash_array="5, 10").add_to(fg_dr)
        
    for snap in all_snapped_paths:
        folium.PolyLine(locations=snap, color="lime", weight=4, opacity=0.8).add_to(fg_snap)
        
    # Add the layers to the map
    fg_gt.add_to(m)
    fg_dr.add_to(m)
    fg_snap.add_to(m)
    
    # Add the layer control widget to the top right
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save(str(output_path))
    print(f"Global map saved to {output_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    test_path = args.data_dir / args.test_file
    scalers_path = args.data_dir / args.test_file.replace("test_blackout_windows", "scalers")

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_items = load_pickle(test_path)
    scalers = load_pickle(scalers_path) if scalers_path.exists() else None

    dvse_model = DVSEModel().to(device)
    if args.dvse_weights.exists():
        dvse_model.load_state_dict(torch.load(args.dvse_weights, map_location=device, weights_only=False))
    dvse_model.eval()

    gyro_model = GyroTCN().to(device)
    gyro_ckpt = None
    if args.gyro_weights.exists():
        gyro_ckpt = torch.load(args.gyro_weights, map_location=device, weights_only=False)
        gyro_model.load_state_dict(gyro_ckpt["model_state"])
    gyro_model.eval()

    acc_scaler = scalers["raw_accel"] if scalers else None
    gyro_scaler = scalers["raw_gyro"] if scalers else None
    y_mean = gyro_ckpt.get("target_mean", 0.0) if gyro_ckpt else 0.0
    y_std = gyro_ckpt.get("target_std", 1.0) if gyro_ckpt else 1.0

    all_drift_records = []
    
    # Collections for the global map
    global_gt_paths = []
    global_dr_paths = []
    global_snapped_paths = []

    # Initialize Viterbi Snapper (High sigma because DR drift can be large)
    snapper = ViterbiSnapper(sigma_z=200.0, beta=20.0)

    with torch.no_grad():
        for ep_idx, item in enumerate(test_items):
            raw_accel = np.asarray(item["blackout"]["raw_accel"], dtype=np.float32)
            raw_gyro = np.asarray(item["blackout"]["raw_gyro"], dtype=np.float32)
            gt_disp = np.asarray(item["ground_truth"]["cumulative_displacement_ne_m"], dtype=np.float32)
            vr_seed = float(item["context"]["vr_seed_ms"])
            init_heading_deg = float(item["ground_truth"]["headings_deg"][0])
            start_lat = float(item["ground_truth"]["start_lat"])
            start_lon = float(item["ground_truth"]["start_lon"])

            # Inference
            acc_scaled = acc_scaler.transform(raw_accel) if acc_scaler else raw_accel
            gyro_scaled = gyro_scaler.transform(raw_gyro) if gyro_scaler else raw_gyro

            gyro_tensor = torch.from_numpy(gyro_scaled).unsqueeze(0).to(device)
            pred_yaw_rate_norm = gyro_model(gyro_tensor).squeeze(0).cpu().numpy()
            pred_yaw_rate_dps = pred_yaw_rate_norm * y_std + y_mean
            
            heading_change_deg = np.cumsum(pred_yaw_rate_dps * (1.0 / HZ))
            pred_heading_rad = np.radians((init_heading_deg + heading_change_deg) % 360.0)

            T = len(raw_accel) // int(HZ)
            vr_seq = np.zeros((1, T, 1), dtype=np.float32)
            vr_seq[0, 0, 0] = vr_seed
            
            delta_v, pred_v_seq, _, _ = dvse_model(
                torch.from_numpy(acc_scaled).unsqueeze(0).to(device),
                torch.from_numpy(gyro_scaled).unsqueeze(0).to(device),
                torch.from_numpy(vr_seq).to(device),
                v_0=torch.tensor([[vr_seed]], dtype=torch.float32).to(device),
                hz=int(HZ)
            )
            pred_v = pred_v_seq.squeeze(0).cpu().numpy()

            # DR Fusion
            psi_1hz = np.array([pred_heading_rad[min((i + 1) * int(HZ) - 1, len(pred_heading_rad) - 1)] for i in range(T)])
            delta_s = pred_v * 1.0
            dr_E = np.cumsum(delta_s * np.sin(psi_1hz))
            dr_N = np.cumsum(delta_s * np.cos(psi_1hz))

            gt_1hz_indices = [min((i + 1) * int(HZ) - 1, len(gt_disp) - 1) for i in range(T)]
            gt_N = gt_disp[gt_1hz_indices, 0]
            gt_E = gt_disp[gt_1hz_indices, 1]

            dr_lat, dr_lon = en_to_latlon(dr_E, dr_N, start_lat, start_lon)
            gt_lat, gt_lon = en_to_latlon(gt_E, gt_N, start_lat, start_lon)

            dr_coords = [(start_lat, start_lon)] + list(zip(dr_lat, dr_lon))
            gt_coords = [(start_lat, start_lon)] + list(zip(gt_lat, gt_lon))

            # --- MAP MATCHING (VITERBI SNAPPER) ---
            # We use the GT trajectory as the offline road map constraint.
            road_graph = SimpleRoadGraph([gt_coords])
            matched_path = snapper.snap(
                dr_coords,
                get_candidates_fn=lambda obs: road_graph.get_candidates(obs, radius=500.0),
                get_route_dist_fn=lambda c1, c2: road_graph.route_distance(c1, c2)
            )
            
            if matched_path and len(matched_path) == len(dr_coords):
                snapped_coords = [(m['lat'], m['lon']) for m in matched_path]
            else:
                snapped_coords = dr_coords # Fallback if snap fails

            # Store for global map
            global_gt_paths.append(gt_coords)
            global_dr_paths.append(dr_coords)
            global_snapped_paths.append(snapped_coords)

            # Calculate Drift
            drift_m = np.sqrt((dr_E - gt_E) ** 2 + (dr_N - gt_N) ** 2)
            
            for t_sec in range(T):
                snap_lat, snap_lon = snapped_coords[t_sec + 1]
                true_lat, true_lon = gt_coords[t_sec + 1]
                snapped_drift_m = haversine(snap_lat, snap_lon, true_lat, true_lon)

                all_drift_records.append({
                    "episode_idx": ep_idx,
                    "time_sec": t_sec + 1,
                    "dr_E_m": dr_E[t_sec],
                    "dr_N_m": dr_N[t_sec],
                    "gt_E_m": gt_E[t_sec],
                    "gt_N_m": gt_N[t_sec],
                    "raw_drift_m": drift_m[t_sec],
                    "snapped_drift_m": snapped_drift_m
                })

            # Individual Map Plotting
            if ep_idx < args.episodes:
                m = folium.Map(location=[start_lat, start_lon], zoom_start=18, tiles="OpenStreetMap")

                folium.PolyLine(locations=gt_coords, color="blue", weight=6, opacity=0.8, tooltip="Road Network / Ground Truth").add_to(m)
                folium.PolyLine(locations=dr_coords, color="red", weight=4, opacity=0.8, dash_array="5, 10", tooltip="Raw DR Trajectory").add_to(m)
                folium.PolyLine(locations=snapped_coords, color="lime", weight=5, opacity=0.9, tooltip="Viterbi Map-Matched DR").add_to(m)
                
                folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                
                folium.Marker(dr_coords[-1], popup=f"Raw DR End (Error: {drift_m[-1]:.1f}m)", icon=folium.Icon(color="red")).add_to(m)
                snapped_end_err = haversine(snapped_coords[-1][0], snapped_coords[-1][1], gt_coords[-1][0], gt_coords[-1][1])
                folium.Marker(snapped_coords[-1], popup=f"Snapped End (Error: {snapped_end_err:.1f}m)", icon=folium.Icon(color="green", icon="check")).add_to(m)

                map_file = args.output_dir / f"episode_{ep_idx}_map_matched.html"
                m.save(str(map_file))

    # Save records
    drift_df = pd.DataFrame(all_drift_records)
    csv_file = args.output_dir / "detailed_drift_1hz_with_snapper.csv"
    drift_df.to_csv(csv_file, index=False)

    print("\n" + "=" * 60)
    print("VITERBI SNAPPER & MAP GENERATION COMPLETE")
    print("=" * 60)
    
    # Generate the unified global map containing all trajectories
    global_map_path = args.output_dir / "all_episodes_global_map.html"
    generate_global_map(global_gt_paths, global_dr_paths, global_snapped_paths, global_map_path)
    
    final_drifts_raw = drift_df[drift_df["time_sec"] == 60]["raw_drift_m"]
    final_drifts_snapped = drift_df[drift_df["time_sec"] == 60]["snapped_drift_m"]
    
    if len(final_drifts_raw) > 0:
        print("\nSummary Drift after 60s blackout:")
        print(f"               RAW DR MODEL    |   VITERBI MAP-MATCHED")
        print(f"  Mean:       {final_drifts_raw.mean():7.2f} m     |  {final_drifts_snapped.mean():7.2f} m")
        print(f"  Median:     {final_drifts_raw.median():7.2f} m     |  {final_drifts_snapped.median():7.2f} m")
        print(f"  P90:        {final_drifts_raw.quantile(0.90):7.2f} m     |  {final_drifts_snapped.quantile(0.90):7.2f} m")


if __name__ == "__main__":
    main()
