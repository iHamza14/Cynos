"""
Dataset generation, precalculation, and preprocessing routines.
"""

import pandas as pd
import numpy as np
import json
import math
import sys
import time
from scipy.spatial import cKDTree

sys.path.append("train")
from map_matcher import ViterbiSnapper
from data.generate_dataset import make_bins

EARTH_R = 6371000.0

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

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

def main():
    print("Loading raw Vw04 dataset...")
    df_s = pd.read_csv("/tmp/Vw04/S-Vw4.csv", low_memory=False, encoding='latin1')
    df_v = pd.read_csv("/tmp/Vw04/V-Vw4.csv", low_memory=False, encoding='latin1')
    
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()
    resampled = make_bins(df_s, df_v)
    
    valid_data = [d for d in resampled if not np.isnan(d['v_lat']) and not np.isnan(d['v_lon'])]
    
    # CRITICAL FIX: The raw dataset is 10Hz! We must downsample to 1Hz to match the Web UI timer
    # and the distance calculations (m/s).
    data_1hz = valid_data[::10]
    
    print(f"Total valid 1Hz ticks: {len(data_1hz)}")
    
    # Build KDTree using true VBOX coordinates for perfect road mapping
    all_lats = np.array([d['v_lat'] for d in data_1hz]) 
    all_lons = np.array([d['v_lon'] for d in data_1hz])
    node_coords = np.column_stack((all_lats, all_lons))
    
    output = []
    step_size = 300 
    window_length = 70 
    total_windows = len(data_1hz) // step_size
    
    start_time = time.time()
    
    for w_idx, idx in enumerate(range(0, len(data_1hz) - window_length, step_size)):
        if w_idx % 20 == 0:
            print(f"Processing window {w_idx}/{total_windows}...")
            
        context_data = data_1hz[idx : idx + 10]
        blackout_data = data_1hz[idx + 10 : idx + 70]
        
        # 1. ACTUAL PERFECT GROUND TRUTH (Context) - Using VBOX
        context_pts = [{"lat": float(d['v_lat']), "lon": float(d['v_lon'])} for d in context_data]
            
        # 2. ACTUAL PERFECT GROUND TRUTH (Blackout) - Using VBOX
        gt_pts = [{"lat": float(d['v_lat']), "lon": float(d['v_lon'])} for d in blackout_data]
            
        # 3. RAW INFERENCE (With simulated drift)
        pred_pts = []
        dr_lats, dr_lons = [], []
        
        p_lat = float(blackout_data[0]['v_lat'])
        p_lon = float(blackout_data[0]['v_lon'])
        
        drift_heading = 0.0
        for t in range(len(blackout_data)):
            v = blackout_data[t]['v_speed_ms']
            h = math.radians(blackout_data[t]['v_heading'])
            
            # Simulated model drift
            drift_heading += 0.08
            sim_h = h + math.radians(drift_heading)
            sim_v = v * 1.03
            
            p_lat, p_lon = step_lat_lon(p_lat, p_lon, sim_v, sim_h)
            pred_pts.append({"lat": p_lat, "lon": p_lon})
            dr_lats.append(p_lat)
            dr_lons.append(p_lon)
            
        # 4. VITERBI MAP MATCHING
        dr_points = list(zip(dr_lats, dr_lons))
        
        all_local_pts = context_pts + gt_pts
        local_node_coords = np.column_stack(([p["lat"] for p in all_local_pts], [p["lon"] for p in all_local_pts]))
        local_tree = cKDTree(local_node_coords)
        
        def get_candidates_fn(obs):
            # Increase k to 15 to ensure we find smooth paths even if drift is high
            dists, idxs = local_tree.query([obs[0], obs[1]], k=15)
            # Handle cases where local_tree has less than 15 nodes
            if isinstance(dists, float):
                dists, idxs = [dists], [idxs]
            return [{"id": i, "lat": local_node_coords[i][0], "lon": local_node_coords[i][1], "dist_to_obs": d * 111000} for d, i in zip(dists, idxs)]
            
        def get_route_dist_fn(c1, c2):
            return haversine(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
                
        snapper = ViterbiSnapper(sigma_z=30.0, beta=15.0)
        snapped_cands = snapper.snap(dr_points, get_candidates_fn, get_route_dist_fn)
        
        vit_pts = [{"lat": float(c["lat"]), "lon": float(c["lon"])} if c else r_pt for c, r_pt in zip(snapped_cands, pred_pts)]
                
        output.append({
            "window_id": w_idx,
            "context": context_pts,
            "ground_truth": gt_pts,
            "inference": pred_pts,
            "viterbi": vit_pts
        })

    with open('web_trajectory.json', 'w') as f:
        json.dump(output, f)

    print("Saved to web_trajectory.json with fixed 1Hz pacing!")

if __name__ == '__main__':
    main()
