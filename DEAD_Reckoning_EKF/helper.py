import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def get_cols(df, keyword):
    return [c for c in df.columns if keyword in c]

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

def cumulative_displacement(bins):
    # Extract all lats and lons into fast numpy arrays
    lats = np.radians([b['v_lat'] for b in bins])
    lons = np.radians([b['v_lon'] for b in bins])
    
    R = 6371000.0
    
    # Calculate step-by-step differences
    d_lats = np.diff(lats)
    d_lons = np.diff(lons)
    
    # Midpoint latitudes for the Easting cosine scaling
    mid_lats = (lats[:-1] + lats[1:]) / 2.0
    
    # Vectorized step deltas (North, East)
    delta_north = d_lats * R
    delta_east = d_lons * R * np.cos(mid_lats)
    
    # Stack and cumulatively sum the deltas, prepending the [0,0] origin
    deltas = np.column_stack((delta_north, delta_east))
    disps = np.vstack(([0.0, 0.0], np.cumsum(deltas, axis=0)))
    
    return disps # shape (N, 2)


def total_distance_m(bins):
    """Vectorized calculation of total ground truth trajectory distance in metres."""
    if len(bins) < 2:
        return 0.0
        
    lats = np.radians([b['v_lat'] for b in bins])
    lons = np.radians([b['v_lon'] for b in bins])

    R = 6371000.0
    
    # Calculate step-by-step angular differences
    d_lats = np.diff(lats)
    d_lons = np.diff(lons)
    
    # Midpoint latitudes for Easting cosine scaling
    mid_lats = (lats[:-1] + lats[1:]) / 2.0
    
    # Vectorized step deltas (North, East)
    delta_north = d_lats * R
    delta_east = d_lons * R * np.cos(mid_lats)
    
    # Calculate Euclidean distance for every step simultaneously, then sum
    step_distances = np.sqrt(delta_north**2 + delta_east**2)
    
    return float(np.sum(step_distances))

def bin_dataset(df_s, df_v):
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
    
    v_rename = {c: f'v_{c}' for c in df_v.columns if c != 'abs_time'}
    df_v.rename(columns=v_rename, inplace=True)
    
    df_merged = pd.merge_asof(
        df_s, df_v,
        on='abs_time',
        direction='nearest',
        tolerance=0.2
    )
    df_merged.dropna(subset=['v_Velocity (km/hr)'], inplace=True)
    
    r_wheel = estimate_wheel_radius(df_merged)
    df_merged['v_odo_speed'] = (df_merged['v_Wheel Speed Front Left (rad/sec)'] + 
                                df_merged['v_Wheel Speed Front Right (rad/sec)']) / 2.0 * r_wheel
    
    sat_col = get_cols(df_merged, 'GPS SATELLITES')[0]
    
    def clean_sats(val):
        if isinstance(val, str) and '/' in val:
            return float(val.split('/')[0].strip())
        return float(val)
        
    df_merged[sat_col] = df_merged[sat_col].apply(clean_sats)
    
    start_time = np.ceil(df_merged['abs_time'].min() * 10) / 10.0
    end_time = np.floor(df_merged['abs_time'].max() * 10) / 10.0
    grid_times = np.arange(start_time, end_time + 0.1, 0.1)
    
    df_resampled = pd.DataFrame({'abs_time': grid_times})
    
    acc_x = get_cols(df_merged, 'ACCELEROMETER X')[0]
    acc_y = get_cols(df_merged, 'ACCELEROMETER Y')[0]
    acc_z = get_cols(df_merged, 'ACCELEROMETER Z')[0]
    gyr_x = get_cols(df_merged, 'GYROSCOPE Yaw')[0]
    gyr_y = get_cols(df_merged, 'GYROSCOPE Pitch')[0]
    gyr_z = get_cols(df_merged, 'GYROSCOPE Roll')[0]
    grv_x = get_cols(df_merged, 'GRAVITY X')[0]
    grv_y = get_cols(df_merged, 'GRAVITY Y')[0]
    grv_z = get_cols(df_merged, 'GRAVITY Z')[0]
    mob_spd = get_cols(df_merged, 'GPS SPEED')[0]
    mob_lat = get_cols(df_merged, 'GPS LATITUDE')[0]
    mob_lon = get_cols(df_merged, 'GPS LONGITUDE')[0]
    mob_ori = get_cols(df_merged, 'GPS ORIENTATION')[0]
    mob_acc = get_cols(df_merged, 'GPS ACCURACY')[0]
    v_lat = get_cols(df_merged, 'v_Latitude')[0]
    v_lon = get_cols(df_merged, 'v_Longitude')[0]
    v_hdg = get_cols(df_merged, 'v_Heading')[0]
    
    cols_to_interp = [
        acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, grv_x, grv_y, grv_z,
        mob_spd, mob_lat, mob_lon, mob_ori, mob_acc, sat_col,
        'v_odo_speed', v_lat, v_lon, v_hdg
    ]
    
    for c in cols_to_interp:
        df_resampled[c] = np.interp(grid_times, df_merged['abs_time'], df_merged[c])
        
    bins = []
    for i, row in df_resampled.iterrows():
        bins.append({
            'raw_accel': [row[acc_x], row[acc_y], row[acc_z]],
            'raw_gyro': [row[gyr_x], row[gyr_y], row[gyr_z]],
            'gravity': [row[grv_x], row[grv_y], row[grv_z]],
            'mobile_gps_speed': row[mob_spd],
            'mobile_lat': row[mob_lat],
            'mobile_lon': row[mob_lon],
            'mobile_gps_orientation': row[mob_ori],
            'gps_acc': row[mob_acc],
            'gps_sats': row[sat_col],
            'v_odo_speed': row['v_odo_speed'],
            'v_lat': row[v_lat],
            'v_lon': row[v_lon],
            'v_heading': row[v_hdg],
            'abs_sec': row['abs_time']
        })
        
    for i in range(len(bins)):
        if i == 0:
            bins[i]['v_delta_speed'] = 0.0
        else:
            bins[i]['v_delta_speed'] = bins[i]['v_odo_speed'] - bins[i-1]['v_odo_speed']
            
    return bins