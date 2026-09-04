import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os
import ast

def get_cols(df, keyword):
    return [c for c in df.columns if keyword in c]

def extract_features(window_data):
    features = []
    for i in range(3):
        axis_data = window_data[:, i]
        features.extend([
            float(np.std(axis_data)),
            float(np.max(axis_data)),
            float(np.min(axis_data)),
            float(np.sqrt(np.mean(axis_data**2))), # RMS
            float(skew(axis_data)) if len(axis_data)>0 else 0.0,
            float(kurtosis(axis_data)) if len(axis_data)>0 else 0.0
        ])
    return features

def prepare_data(s_path, v_path, out_dir):
    print(f"Loading datasets...")
    df_s = pd.read_csv(s_path, encoding='latin-1')
    df_v = pd.read_csv(v_path, encoding='latin-1')
    
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()
    
    # 1. Absolute time syncing
    v_time_col = get_cols(df_v, 'Time Since Start')[0]
    df_v['UTC_TIME'] = df_v[v_time_col]
    
    s_date_col = get_cols(df_s, 'DATE')[0]
    def parse_s_time(date_str):
        try:
            time_part = str(date_str).strip().split(' ')[1]
            h, m, s, ms = map(float, time_part.split(':'))
            # Calculate absolute seconds for alignment
            return h * 3600 + m * 60 + s + ms/1000.0
        except:
            return np.nan
            
    df_s['UTC_TIME'] = df_s[s_date_col].apply(parse_s_time)
    df_s.dropna(subset=['UTC_TIME'], inplace=True)
    
    # 2. Extract targets
    s_speed_col = get_cols(df_s, 'GPS SPEED')[0]
    df_s['GPS_SPEED_MS'] = df_s[s_speed_col] # Assuming it's already m/s despite the Kmh label based on prior analysis
    
    # 3. 1-second Bins
    df_s['sec_bin'] = np.floor(df_s['UTC_TIME']).astype(int)
    
    acc_cols = get_cols(df_s, 'ACCELEROMETER')[:3]
    gyro_cols = get_cols(df_s, 'GYROSCOPE')[:3]
    grav_cols = get_cols(df_s, 'GRAVITY')[:3]
    
    windows = []
    
    print("Generating 1-second windows...")
    for sec, group in df_s.groupby('sec_bin'):
        if len(group) < 5:
            continue
            
        acc_data = group[acc_cols].values
        gyro_data = group[gyro_cols].values
        grav_data = group[grav_cols].values
        
        acc_feat = extract_features(acc_data)
        gyro_feat = extract_features(gyro_data)
        
        dt = 1.0 / len(group)
        int_acc = np.sum(acc_data, axis=0) * dt
        avg_grav = np.mean(grav_data, axis=0)
        avg_acc = np.mean(acc_data, axis=0)
        
        mtn_in = int_acc.tolist() + avg_grav.tolist()
        gps_speed = float(group['GPS_SPEED_MS'].mean())
        
        windows.append({
            'sec': sec,
            'acc_feat': acc_feat,
            'gyro_feat': gyro_feat,
            'mtn_in': mtn_in,
            'raw_acc': avg_acc.tolist(),
            'grav': avg_grav.tolist(),
            'gps_speed': gps_speed
        })
        
    win_df = pd.DataFrame(windows).sort_values('sec').reset_index(drop=True)
    win_df['delta_v_target'] = win_df['gps_speed'].diff().fillna(0)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Explicit 80/20 chronological split
    train_size = int(0.8 * len(win_df))
    train_df = win_df.iloc[:train_size]
    test_df = win_df.iloc[train_size:]
    
    train_file = os.path.join(out_dir, 'train_windows.csv')
    test_file = os.path.join(out_dir, 'test_windows.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Saved {len(train_df)} train windows to {train_file}")
    print(f"Saved {len(test_df)} test windows to {test_file}")

if __name__ == '__main__':
    # Update these paths if running on a different dataset
    s_path = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
    v_path = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'
    out_dir = '/Users/hamza/SIH/IO-VNBD/DVSE_Implementation/data'
    prepare_data(s_path, v_path, out_dir)
