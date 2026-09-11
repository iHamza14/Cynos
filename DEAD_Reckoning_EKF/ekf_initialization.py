import numpy as np
import pandas as pd

def process_and_initialize_ekf(csv_path: str, variance_threshold: float = 0.05):
    # Data Preprocessing with encoding fallback
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except (UnicodeDecodeError, Exception):
        df = pd.read_csv(csv_path, encoding='latin1')
        
    df.columns = df.columns.str.strip()
    
    drop_keywords = ['GRAVITY', 'MAGNETIC FIELD', 'ORIENTATION']
    cols_to_drop = [c for c in df.columns if any(k in c.upper() for k in drop_keywords)]
    df = df.drop(columns=cols_to_drop)
    
    speed_cols = [c for c in df.columns if 'GPS SPEED' in c.upper()]
    if speed_cols:
        speed_col = speed_cols[0]
        df[speed_col] = df[speed_col] / 3.6
    else:
        speed_col = 'GPS SPEED'
        
    accel_cols = [
        [c for c in df.columns if 'ACCEL' in c.upper() and 'X' in c.upper()][0],
        [c for c in df.columns if 'ACCEL' in c.upper() and 'Y' in c.upper()][0],
        [c for c in df.columns if 'ACCEL' in c.upper() and 'Z' in c.upper()][0]
    ]
    gyro_cols = [
        [c for c in df.columns if 'GYRO' in c.upper() and 'YAW' in c.upper()][0],
        [c for c in df.columns if 'GYRO' in c.upper() and 'PITCH' in c.upper()][0],
        [c for c in df.columns if 'GYRO' in c.upper() and 'ROLL' in c.upper()][0]
    ]
    
    # Identify the static calibration window
    # 10 samples represent a 1-second window at 10 Hz
    rolling_var = df[accel_cols].rolling(window=10).var().sum(axis=1)
    static_mask = (df[speed_col] == 0) & (rolling_var < variance_threshold)
    static_data = df[static_mask]
    
    if static_data.empty:
        # If strict speed == 0 is too sparse due to noise, fallback to lowest variance window
        static_data = df.nsmallest(50, rolling_var.name) if rolling_var.name in df else df.iloc[:50]
        if static_data.empty:
            raise ValueError("Could not find a valid static window.")
        
    # Zero-Velocity Analysis & Noise Characterization
    gyro_bias = static_data[gyro_cols].mean().values
    
    accel_mean = static_data[accel_cols].mean().values
    roll_init = np.arctan2(accel_mean[1], accel_mean[2])
    pitch_init = np.arctan2(-accel_mean[0], np.sqrt(accel_mean[1]**2 + accel_mean[2]**2))
    
    g = 9.80665
    ideal_g = np.array([
        -g * np.sin(pitch_init),
        g * np.cos(pitch_init) * np.sin(roll_init),
        g * np.cos(pitch_init) * np.cos(roll_init)
    ])
    accel_bias = accel_mean - ideal_g
    
    accel_var = static_data[accel_cols].var().values
    gyro_var = static_data[gyro_cols].var().values

    # EKF Initialization
    x = np.zeros(15)
    x[3:6] = 0.0
    x[6:9] = [roll_init, pitch_init, 0.0]
    x[9:12] = accel_bias
    x[12:15] = gyro_bias

    R = np.diag(np.concatenate((accel_var, gyro_var)))

    P = np.zeros((15, 15))
    P[0:3, 0:3] = np.eye(3) * 10.0   
    P[3:6, 3:6] = np.eye(3) * 1e-8   
    P[6:9, 6:9] = np.eye(3) * 1e-2   
    P[9:12, 9:12] = np.eye(3) * 1e-8 
    P[12:15, 12:15] = np.eye(3) * 1e-8 

    return x, P, R
