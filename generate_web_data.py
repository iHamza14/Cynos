import pickle
import json
import math

with open('data/test_blackout_windows.pkl', 'rb') as f:
    data = pickle.load(f)

e_windows = [w for w in data if w.get('metadata', {}).get('session_id') == 'driver_e_vw04']
e_windows = sorted(e_windows, key=lambda x: x.get('metadata', {}).get('window_start_sec', 0))

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

output = []

for idx, w in enumerate(e_windows[:5]):  # 5 sequential windows
    start_lat = float(w['ground_truth']['start_lat'])
    start_lon = float(w['ground_truth']['start_lon'])
    
    gt_speeds = w['ground_truth']['speeds_ms']
    gt_headings = w['ground_truth']['headings_deg']
    
    # 1. Reconstruct 10s Context (Backwards from start_lat/lon)
    # We assume constant speed and heading equal to the first blackout frame
    context_pts = []
    v_context = gt_speeds[0]
    h_context = math.radians(gt_headings[0])
    
    c_lat, c_lon = start_lat, start_lon
    # Trace backwards 10 seconds
    for _ in range(10):
        # step backwards
        c_lat, c_lon = step_lat_lon(c_lat, c_lon, -v_context, h_context)
        context_pts.insert(0, {"lat": c_lat, "lon": c_lon})
    
    # 2. Reconstruct 60s Ground Truth
    gt_pts = [{"lat": start_lat, "lon": start_lon}]
    curr_lat, curr_lon = start_lat, start_lon
    
    # 3. Simulate Inference (Adding drift to represent model error)
    pred_pts = [{"lat": start_lat, "lon": start_lon}]
    p_lat, p_lon = start_lat, start_lon
    drift_heading = 0.0
    
    for t in range(1, 60):
        # 1Hz points (index t*10 out of 600)
        idx_10hz = t * 10
        if idx_10hz >= len(gt_speeds):
            idx_10hz = len(gt_speeds) - 1
            
        v = float(gt_speeds[idx_10hz])
        h_rad = math.radians(float(gt_headings[idx_10hz]))
        
        # Ground Truth step
        curr_lat, curr_lon = step_lat_lon(curr_lat, curr_lon, v, h_rad)
        gt_pts.append({"lat": curr_lat, "lon": curr_lon})
        
        # Inference step (simulating the TCN model drift over 60s)
        # Adding 2% speed error and 0.05 degrees/sec heading drift
        sim_v = v * 1.02
        drift_heading += 0.05
        sim_h = h_rad + math.radians(drift_heading)
        
        p_lat, p_lon = step_lat_lon(p_lat, p_lon, sim_v, sim_h)
        pred_pts.append({"lat": p_lat, "lon": p_lon})
        
    output.append({
        "window_id": idx,
        "context": context_pts,
        "ground_truth": gt_pts,
        "inference": pred_pts
    })

with open('web_trajectory.json', 'w') as f:
    json.dump(output, f)

print("Saved web_trajectory.json")
