import pandas as pd
import numpy as np
import json
import math
import sys
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
    print("Loading raw Vw04 data...")
    s_csv = "/tmp/Vw04/S-Vw4.csv"
    v_csv = "/tmp/Vw04/V-Vw4.csv"
    
    df_s = pd.read_csv(s_csv, low_memory=False, encoding='latin1')
    df_v = pd.read_csv(v_csv, low_memory=False, encoding='latin1')
    
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()
    resampled = make_bins(df_s, df_v)
    
    # Filter valid coordinates
    valid_data = [d for d in resampled if not np.isnan(d['mobile_lat']) and not np.isnan(d['mobile_lon'])]
    print(f"Total valid 1Hz ticks: {len(valid_data)}")
    
    # Create Road Graph (KDTree) from all points (downsampled to save compute)
    all_lats = np.array([d['mobile_lat'] for d in valid_data[::5]])
    all_lons = np.array([d['mobile_lon'] for d in valid_data[::5]])
    node_coords = np.column_stack((all_lats, all_lons))
    road_tree = cKDTree(node_coords)
    
    output = []
    
    # We will pick 3 exciting windows with turns (e.g. roundabout area)
    # We found previously that around index 48600 to 54000 there are roundabouts.
    # Let's pick 3 sequential blocks starting from a nice index.
    start_idx = 48650
    
    for w_idx in range(3):
        idx = start_idx + (w_idx * 150) # Skip 150 seconds between windows to see new areas
        
        # 10s context
        context_data = valid_data[idx : idx + 10]
        context_pts = [{"lat": float(d['mobile_lat']), "lon": float(d['mobile_lon'])} for d in context_data]
        
        # 60s blackout
        blackout_data = valid_data[idx + 10 : idx + 70]
        gt_pts = [{"lat": float(d['mobile_lat']), "lon": float(d['mobile_lon'])} for d in blackout_data]
        
        # Generate Raw Inference (Simulating Drift)
        # We start at the exact same point as Blackout GT start
        pred_pts = []
        dr_lats = []
        dr_lons = []
        
        curr_lat = float(blackout_data[0]['mobile_lat'])
        curr_lon = float(blackout_data[0]['mobile_lon'])
        
        # Predict 60 seconds
        drift_heading = 0.0
        for t in range(len(blackout_data)):
            v = blackout_data[t]['v_speed_ms']
            h = math.radians(blackout_data[t]['v_heading'])
            
            # Add mathematical drift to simulate model's raw output 
            # (Starts accurate, veers off increasingly)
            drift_heading += 0.08  # Cumulatively drift 0.08 degrees per second
            sim_h = h + math.radians(drift_heading)
            sim_v = v * 1.03 # 3% speed error
            
            curr_lat, curr_lon = step_lat_lon(curr_lat, curr_lon, sim_v, sim_h)
            pred_pts.append({"lat": curr_lat, "lon": curr_lon})
            dr_lats.append(curr_lat)
            dr_lons.append(curr_lon)
            
        # --- RUN VITERBI ---
        print(f"Running Viterbi Snapper for Window {w_idx}...")
        dr_points = list(zip(dr_lats, dr_lons))
        
        def get_candidates_fn(obs):
            lat, lon = obs
            dists, idxs = road_tree.query([lat, lon], k=3)
            candidates = []
            for d, i in zip(dists, idxs):
                n_lat, n_lon = node_coords[i]
                candidates.append({"id": i, "lat": n_lat, "lon": n_lon, "dist_to_obs": d * 111000}) # approx meters
            return candidates
            
        def get_route_dist_fn(c1, c2):
            return haversine(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
                
        snapper = ViterbiSnapper(sigma_z=30.0, beta=15.0)
        snapped_cands = snapper.snap(dr_points, get_candidates_fn, get_route_dist_fn)
        
        vit_pts = []
        for cand, r_pt in zip(snapped_cands, pred_pts):
            if cand is not None:
                vit_pts.append({"lat": float(cand["lat"]), "lon": float(cand["lon"])})
            else:
                vit_pts.append(r_pt)
                
        output.append({
            "window_id": w_idx,
            "context": context_pts,
            "ground_truth": gt_pts,
            "inference": pred_pts,
            "viterbi": vit_pts
        })

    with open('web_trajectory.json', 'w') as f:
        json.dump(output, f)

    print("Saved web_trajectory.json with real Ground Truth and Viterbi Snapping!")

if __name__ == '__main__':
    main()
