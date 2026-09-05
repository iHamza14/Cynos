import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import skew, kurtosis

def get_cols(df, keyword):
    return [c for c in df.columns if keyword in c]

def extract_features(window_data):
    features = []
    for i in range(3):
        axis = window_data[:, i]
        features.extend([
            float(np.std(axis)),
            float(np.max(axis)),
            float(np.min(axis)),
            float(np.sqrt(np.mean(axis**2))),
            float(np.clip(skew(axis), -10, 10)),
            float(np.clip(kurtosis(axis), -10, 10)),  # ← clip both
        ])
    return features

def haversine_delta(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_north = (lat2 - lat1) * R
    delta_east = (lon2 - lon1) * R * np.cos((lat1 + lat2) / 2.0)
    return [float(delta_north), float(delta_east)]

def estimate_wheel_radius(df_merged, n_samples=10000):
    mask = (
        (df_merged['v_Velocity (km/hr)'] > 10) &
        (df_merged['v_No of GPS Satellites Available'] >= 6) &
        (df_merged['GPS ACCURACY (m)'] < 8)
    )
    if mask.sum() == 0:
        print("Warning: No clean data for r_wheel estimation. Defaulting to 0.3")
        return 0.3
        
    df_clean = df_merged[mask].sample(min(n_samples, mask.sum()), random_state=42)
    v_gps = df_clean['v_Velocity (km/hr)'].values / 3.6
    w_avg = (
        df_clean['v_Wheel Speed Front Left (rad/sec)'] +
        df_clean['v_Wheel Speed Front Right (rad/sec)']
    ).values / 2
    
    r_wheel = np.dot(w_avg, v_gps) / (np.dot(w_avg, w_avg) + 1e-8)
    print(f"Estimated r_wheel: {r_wheel:.4f} m")
    return float(r_wheel)

def bin_dataset(df_s, df_v):
    print("Parsing times and syncing...")
    s_date_col = get_cols(df_s, 'DATE')[0]
    
    def parse_s_time(date_str):
        try:
            time_part = str(date_str).strip().split(' ')[1]
            h, m, s, ms = map(float, time_part.split(':'))
            return h * 3600 + m * 60 + s + ms/1000.0
        except:
            return np.nan
            
    df_s['abs_time'] = df_s[s_date_col].apply(parse_s_time)
    df_s.dropna(subset=['abs_time'], inplace=True)
    df_s.sort_values('abs_time', inplace=True)
    
    v_time_col = get_cols(df_v, 'Time Since Start')[0]
    df_v['abs_time'] = df_v[v_time_col]
    df_v.sort_values('abs_time', inplace=True)
    
    # Prefix vehicle columns
    v_rename = {c: f'v_{c}' for c in df_v.columns if c != 'abs_time'}
    df_v.rename(columns=v_rename, inplace=True)
    
    # Merge-asof
    df_merged = pd.merge_asof(
        df_s, df_v,
        on='abs_time',
        direction='nearest',
        tolerance=0.2
    )
    # Drop rows that couldn't match within tolerance
    df_merged.dropna(subset=['v_Velocity (km/hr)'], inplace=True)
    
    r_wheel = estimate_wheel_radius(df_merged)
    df_merged['v_odo_speed'] = (df_merged['v_Wheel Speed Front Left (rad/sec)'] + 
                                df_merged['v_Wheel Speed Front Right (rad/sec)']) / 2.0 * r_wheel
    
    df_merged['sec_bin'] = np.floor(df_merged['abs_time']).astype(int)
    
    acc_cols = get_cols(df_merged, 'ACCELEROMETER')[:3]
    gyro_cols = get_cols(df_merged, 'GYROSCOPE')[:3]
    grav_cols = get_cols(df_merged, 'GRAVITY')[:3]
    
    mobile_speed_col = get_cols(df_merged, 'GPS SPEED')[0]
    mobile_lat_col = get_cols(df_merged, 'LATITUDE')[0]
    mobile_lon_col = get_cols(df_merged, 'LONGITUDE')[0]
    mobile_orient_col = get_cols(df_merged, 'GPS ORIENTATION')[0]
    mobile_acc_col = get_cols(df_merged, 'GPS ACCURACY')[0]
    mobile_sat_col = get_cols(df_merged, 'GPS SATELLITES')[0]
    
    bins = []
    print("Building 1-second bins...")
    for sec, group in df_merged.groupby('sec_bin'):
        if len(group) < 5: continue
            
        acc_data = group[acc_cols].values
        gyro_data = group[gyro_cols].values
        grav_data = group[grav_cols].values
        
        avg_acc = np.mean(acc_data, axis=0)
        avg_grav = np.mean(grav_data, axis=0)
        int_acc = np.sum(acc_data, axis=0) * (1.0 / len(group))
        
        # Parse sat string if needed ("15 / 24")
        sat_val = group[mobile_sat_col].iloc[0]
        if isinstance(sat_val, str) and '/' in sat_val:
            sat_val = float(sat_val.split('/')[0].strip())
        else:
            sat_val = float(sat_val)
            
        bins.append({
            'acc_feat': extract_features(acc_data),
            'gyro_feat': extract_features(gyro_data),
            'mtn_input': int_acc.tolist() + avg_grav.tolist(),
            'raw_accel': avg_acc.tolist(),
            'gravity': avg_grav.tolist(),
            'mobile_gps_speed': float(group[mobile_speed_col].mean()), 
            'v_odo_speed': float(group['v_odo_speed'].mean()),
            'v_lat': float(group['v_Latitude (degrees)'].mean()),
            'v_lon': float(group['v_Longitude (degrees)'].mean()),
            'v_heading': float(group['v_Heading (degrees)'].mean()),
            'gps_acc': float(group[mobile_acc_col].mean()),
            'gps_sats': sat_val,
            'v_gps_sats': float(group['v_No of GPS Satellites Available'].mean()),
            'abs_sec': sec,
            'mobile_lat': float(group[mobile_lat_col].mean()),
            'mobile_lon': float(group[mobile_lon_col].mean()),
            'mobile_gps_orientation': float(group[mobile_orient_col].mean())
        })
        
    # Sort bins
    bins = sorted(bins, key=lambda x: x['abs_sec'])
    
    # Calculate v_delta_speed (per-second diff)
    for i in range(len(bins)):
        if i == 0:
            bins[i]['v_delta_speed'] = 0.0
        else:
            bins[i]['v_delta_speed'] = bins[i]['v_odo_speed'] - bins[i-1]['v_odo_speed']
            
    return bins

def build_windows(bins, window_sec=60, step_sec=15):
    windows = []
    
    print(f"Sliding windows (size {window_sec}s, step {step_sec}s)...")
    for i in range(0, len(bins) - window_sec + 1, step_sec):
        window_bins = bins[i:i+window_sec]
        
        # Check continuity
        if window_bins[-1]['abs_sec'] - window_bins[0]['abs_sec'] != window_sec - 1:
            continue
            
        start_bin = window_bins[0]
        end_bin = window_bins[-1]
        
        # Quality gates
        def is_good(b):
            return b['gps_acc'] < 10 and b['gps_sats'] >= 6 and b['v_gps_sats'] >= 4
            
        if not (is_good(start_bin) and is_good(end_bin)):
            continue
            
        bad_interior = sum([1 for b in window_bins if b['gps_acc'] > 15])
        if bad_interior / float(window_sec) > 0.2:
            continue
            
        w = {
            'acc_feat': np.stack([b['acc_feat'] for b in window_bins]),
            'gyro_feat': np.stack([b['gyro_feat'] for b in window_bins]),
            'mtn_input': np.stack([b['mtn_input'] for b in window_bins]),
            'raw_accel': np.stack([b['raw_accel'] for b in window_bins]),
            'gravity': np.stack([b['gravity'] for b in window_bins]),
            'vr_train': np.array([b['v_odo_speed'] for b in window_bins]).reshape(-1, 1),
            'vr_seed': start_bin['mobile_gps_speed'],
            'start_lat': start_bin['mobile_lat'],
            'start_lon': start_bin['mobile_lon'],
            'start_heading': start_bin['mobile_gps_orientation'],
            'delta_v_seq': np.array([b['v_delta_speed'] for b in window_bins]).reshape(-1, 1),
            'end_lat_v': end_bin['v_lat'],
            'end_lon_v': end_bin['v_lon'],
            'heading_target_rad': np.array([np.radians(b['v_heading']) for b in window_bins]),
            'target_disp': np.array(haversine_delta(
                start_bin['v_lat'], start_bin['v_lon'],
                end_bin['v_lat'], end_bin['v_lon']
            ))
        }
        windows.append(w)
        
    return windows

if __name__ == '__main__':
    s_path = '/home/wolverine/sih/2026/Cynos/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
    v_path = '/home/wolverine/sih/2026/Cynos/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'
    
    df_s = pd.read_csv(s_path, encoding='latin-1')
    df_s.columns = df_s.columns.str.strip()
    df_v = pd.read_csv(v_path, encoding='latin-1')
    df_v.columns = df_v.columns.str.strip()
    
    bins = bin_dataset(df_s, df_v)
    
    # 80/20 Chronological Split of the BINS (since we only have 1 trip)
    # To avoid data leakage, we split the continuous bins before windowing.
    split_idx = int(len(bins) * 0.8)
    train_bins = bins[:split_idx]
    test_bins = bins[split_idx:]
    
    train_windows = build_windows(train_bins, 60, 15)
    test_windows = build_windows(test_bins, 60, 15)
    
    print(f"Generated {len(train_windows)} train windows and {len(test_windows)} test windows.")
    
    os.makedirs('data', exist_ok=True)
    with open('data/train_windows.pkl', 'wb') as f:
        pickle.dump(train_windows, f)
    with open('data/test_windows.pkl', 'wb') as f:
        pickle.dump(test_windows, f)
