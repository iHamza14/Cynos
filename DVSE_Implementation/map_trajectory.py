import pandas as pd
import numpy as np
import torch
import os
import folium
import ast
from torch.utils.data import DataLoader
from models import DVSE
from dataset import DVSEDataset
from data_sync import get_cols, extract_features

def create_trajectory_map():
    s_path = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
    
    print("Loading Vw04 Dataset...")
    df_s = pd.read_csv(s_path, encoding='latin-1')
    df_s.columns = df_s.columns.str.strip()
    
    # Time Parsing
    s_date_col = get_cols(df_s, 'DATE')[0]
    def parse_s_time(date_str):
        try:
            time_part = str(date_str).strip().split(' ')[1]
            h, m, s, ms = map(float, time_part.split(':'))
            return h * 3600 + m * 60 + s + ms/1000.0
        except:
            return np.nan
            
    df_s['UTC_TIME'] = df_s[s_date_col].apply(parse_s_time)
    df_s.dropna(subset=['UTC_TIME'], inplace=True)
    
    # 30 Minute cutoff (1800 seconds)
    start_time = df_s['UTC_TIME'].iloc[0]
    df_s = df_s[df_s['UTC_TIME'] <= start_time + 1800].copy()
    print(f"Subset to first 30 minutes: {len(df_s)} rows")
    
    s_speed_col = get_cols(df_s, 'GPS SPEED')[0]
    lat_col = get_cols(df_s, 'LATITUDE')[0]
    lon_col = get_cols(df_s, 'LONGITUDE')[0]
    orient_col = get_cols(df_s, 'GPS ORIENTATION')[0]
    
    df_s['sec_bin'] = np.floor(df_s['UTC_TIME']).astype(int)
    
    acc_cols = get_cols(df_s, 'ACCELEROMETER')[:3]
    gyro_cols = get_cols(df_s, 'GYROSCOPE')[:3]
    grav_cols = get_cols(df_s, 'GRAVITY')[:3]
    
    windows = []
    
    print("Generating 1-second sequences...")
    for sec, group in df_s.groupby('sec_bin'):
        if len(group) < 5: continue
            
        acc_data = group[acc_cols].values
        gyro_data = group[gyro_cols].values
        grav_data = group[grav_cols].values
        
        mtn_in = (np.sum(acc_data, axis=0) * (1.0/len(group))).tolist() + np.mean(grav_data, axis=0).tolist()
        
        windows.append({
            'sec': sec,
            'acc_feat': extract_features(acc_data),
            'gyro_feat': extract_features(gyro_data),
            'mtn_in': mtn_in,
            'raw_acc': np.mean(acc_data, axis=0).tolist(),
            'grav': np.mean(grav_data, axis=0).tolist(),
            'gps_speed': float(group[s_speed_col].mean()),
            'lat': float(group[lat_col].mean()),
            'lon': float(group[lon_col].mean()),
            'heading': float(group[orient_col].mean()) # degrees (0=N, 90=E)
        })
        
    win_df = pd.DataFrame(windows).sort_values('sec').reset_index(drop=True)
    win_df['delta_v_target'] = win_df['gps_speed'].diff().fillna(0)
    
    # Save temp CSV for Dataset loader
    os.makedirs('data', exist_ok=True)
    tmp_path = 'data/temp_traj.csv'
    win_df.to_csv(tmp_path, index=False)
    
    # Load Model
    model = DVSE()
    model.load_state_dict(torch.load('dvse_model.pth', weights_only=True))
    model.eval()
    
    dataset = DVSEDataset(tmp_path, seq_len=10)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    pred_speeds = []
    
    print("Running DVSE Inference...")
    with torch.no_grad():
        for batch in test_loader:
            acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_dv, target_v = batch
            pred_dv, euler = model(acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav)
            
            # Predict velocity step
            pred_v = torch.zeros_like(pred_dv)
            pred_v[:, 0] = target_v[:, 0]
            for t in range(1, pred_dv.size(1)):
                pred_v[:, t] = pred_v[:, t-1] + pred_dv[:, t]
            
            # We append the prediction for the end of this 10s sequence
            pred_speeds.append(pred_v[0, -1, 0].item())

    # We skip the first 9 seconds due to sequence length
    valid_df = win_df.iloc[9:].reset_index(drop=True)
    
    # Dead Reckoning Trajectory
    start_lat = valid_df['lat'].iloc[0]
    start_lon = valid_df['lon'].iloc[0]
    
    gt_coords = []
    pred_coords = [(start_lat, start_lon)]
    
    cur_lat = start_lat
    cur_lon = start_lon
    
    R_earth = 111320.0 # meters per degree of latitude
    
    for i in range(len(valid_df)):
        gt_coords.append((valid_df['lat'].iloc[i], valid_df['lon'].iloc[i]))
        
        if i > 0:
            v_pred = pred_speeds[i-1] # m/s
            heading = valid_df['heading'].iloc[i-1]
            if np.isnan(heading):
                heading = 0
            
            heading_rad = np.radians(heading)
            
            # Approximation for small displacements
            delta_lat = (v_pred * np.cos(heading_rad)) / R_earth
            delta_lon = (v_pred * np.sin(heading_rad)) / (R_earth * np.cos(np.radians(cur_lat)))
            
            cur_lat += delta_lat
            cur_lon += delta_lon
            pred_coords.append((cur_lat, cur_lon))

    print("Generating Map...")
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # Split ground truth into segments of "Available" (Blue) and "Unavailable/Poor" (Orange)
    gt_avail = []
    gt_unavail = []
    
    # We consider GPS unavailable if lat/lon is NaN, or if time gap > 1s, or if speed is suspiciously stagnant
    for i in range(1, len(valid_df)):
        prev = valid_df.iloc[i-1]
        curr = valid_df.iloc[i]
        
        # Check if GPS was updated or stale/missing
        time_gap = curr['sec'] - prev['sec']
        lat_diff = abs(curr['lat'] - prev['lat'])
        lon_diff = abs(curr['lon'] - prev['lon'])
        
        # If it's a huge time gap, or if it's completely stale while IMU is moving
        if time_gap > 2 or (lat_diff == 0 and lon_diff == 0 and curr['gps_speed'] == 0):
            gt_unavail.append([(prev['lat'], prev['lon']), (curr['lat'], curr['lon'])])
        else:
            gt_avail.append([(prev['lat'], prev['lon']), (curr['lat'], curr['lon'])])
            
    # Ground Truth Available (Blue)
    for segment in gt_avail:
        folium.PolyLine(
            locations=segment,
            color="blue",
            weight=5,
            opacity=0.8,
            tooltip="GPS Available"
        ).add_to(m)
        
    # Ground Truth Unavailable / Stale (Orange)
    for segment in gt_unavail:
        folium.PolyLine(
            locations=segment,
            color="orange",
            weight=5,
            opacity=0.8,
            dash_array='5',
            tooltip="GPS Unavailable / Stale"
        ).add_to(m)
    
    # Predicted (Red)
    folium.PolyLine(
        locations=pred_coords,
        color="red",
        weight=5,
        opacity=0.8,
        dash_array='10',
        tooltip="DVSE Predicted Path (IMU Only)"
    ).add_to(m)
    
    m.save('trajectory_map.html')
    print("Map successfully saved to trajectory_map.html")
    os.remove(tmp_path)

if __name__ == '__main__':
    create_trajectory_map()
