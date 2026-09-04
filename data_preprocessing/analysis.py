import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
import os

# Create plots directory if it doesn't exist
out_dir = './plots'
os.makedirs(out_dir, exist_ok=True)

mobile_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/S-S4.csv'
vehicle_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/V-S4.csv'

print("Loading datasets...")
# Try reading with latin1 to handle special characters if utf-8 fails
try:
    df_m = pd.read_csv(mobile_path)
except:
    df_m = pd.read_csv(mobile_path, encoding='latin1')
    
try:
    df_v = pd.read_csv(vehicle_path)
except:
    df_v = pd.read_csv(vehicle_path, encoding='latin1')

df_m.columns = [c.strip() for c in df_m.columns]
df_v.columns = [c.strip() for c in df_v.columns]

# Helper to find column by prefix
def get_col(df, name):
    for c in df.columns:
        if name in c:
            return c
    return None

# --- Phase 1: Data Quality ---
print("Phase 1: Data Quality")

time_m_col = get_col(df_m, 'TIME SINCE START')
if time_m_col:
    df_m['time_s'] = df_m[time_m_col] / 1000.0
else:
    df_m['time_s'] = np.arange(len(df_m)) * 0.1

time_v_col = get_col(df_v, 'Time Since Start of Day')
if time_v_col:
    df_v['time_s'] = df_v[time_v_col]
    df_v['time_s'] = df_v['time_s'] - df_v['time_s'].iloc[0]
else:
    df_v['time_s'] = np.arange(len(df_v)) * 0.1

dt_m = np.diff(df_m['time_s'])
dt_v = np.diff(df_v['time_s'])

plt.figure(figsize=(10, 4))
plt.plot(dt_m, label='Mobile dt')
plt.plot(dt_v, label='Vehicle dt', alpha=0.7)
plt.title('Sampling Rate Consistency (dt between timestamps)')
plt.ylabel('dt (seconds)')
plt.xlabel('Sample Index')
plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/1_sampling_rate.png')
plt.close()

gps_speed_col = get_col(df_m, 'GPS SPEED')
if gps_speed_col:
    missing_gps = df_m[gps_speed_col].isnull() | (df_m[gps_speed_col] == 0)
    plt.figure(figsize=(10, 2))
    plt.plot(df_m['time_s'], missing_gps, label='Missing/Zero GPS Speed')
    plt.title('Mobile GPS Dropouts over Time')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/1_missing_gps.png')
    plt.close()

v_speed_col = get_col(df_v, 'Indicated Vehicle Speed')
if gps_speed_col and v_speed_col:
    plt.figure(figsize=(10, 4))
    plt.plot(df_m['time_s'], df_m[gps_speed_col], label='Mobile GPS Speed')
    plt.plot(df_v['time_s'], df_v[v_speed_col], label='Vehicle Indicated Speed')
    plt.title('Velocity Time Alignment (Mobile vs Vehicle)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/1_time_alignment.png')
    plt.close()


# --- Phase 2: Noise Profile ---
print("Phase 2: Noise Profile")
if gps_speed_col:
    stationary_mask = df_m[gps_speed_col] < 0.5
    df_stat = df_m[stationary_mask]

    if len(df_stat) > 100:
        gyro_yaw = get_col(df_m, 'GYROSCOPE Yaw')
        gyro_pitch = get_col(df_m, 'GYROSCOPE Pitch')
        gyro_roll = get_col(df_m, 'GYROSCOPE Roll')
        gyro_cols = [c for c in [gyro_yaw, gyro_pitch, gyro_roll] if c]
        
        if gyro_cols:
            df_stat[gyro_cols].plot(figsize=(10, 4), alpha=0.7)
            plt.title('Gyroscope Raw Signal (Stationary)')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/2_gyro_stationary.png')
            plt.close()

            plt.figure(figsize=(10, 4))
            for col in gyro_cols:
                y = df_stat[col].dropna().values
                if len(y) > 100:
                    a = acf(y, nlags=50, fft=True)
                    plt.plot(a, label=col)
            plt.title('Autocorrelation of Gyro Axes (Stationary)')
            plt.xlabel('Lags')
            plt.ylabel('ACF')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{out_dir}/2_autocorrelation.png')
            plt.close()

            df_stat[gyro_cols].hist(bins=50, figsize=(10, 6), alpha=0.7)
            plt.suptitle('Histogram of Gyro Noise (Stationary)')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/2_histogram.png')
            plt.close()

            try:
                sns.pairplot(df_stat[gyro_cols].dropna(), kind='hist')
                plt.savefig(f'{out_dir}/2_cross_correlation.png')
                plt.close()
            except Exception as e:
                print("Could not plot pairplot:", e)
    else:
        print("Not enough stationary data.")

# --- Phase 3: Phone Pose Problem ---
print("Phase 3: Phone Pose Problem")
df_v_interp = pd.DataFrame()
for col in df_v.columns:
    if np.issubdtype(df_v[col].dtype, np.number):
        df_v_interp[col] = np.interp(df_m['time_s'], df_v['time_s'], df_v[col])

orient_yaw = get_col(df_m, 'ORIENTATION (Yaw)')
heading = get_col(df_v, 'Heading')
if orient_yaw and heading in df_v_interp.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df_m['time_s'], df_m[orient_yaw], label='Phone Yaw')
    plt.plot(df_m['time_s'], df_v_interp[heading], label='Vehicle Heading')
    plt.title('Phone Yaw vs Vehicle Heading')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/3_orientation_diff.png')
    plt.close()

accel_x = get_col(df_m, 'ACCELEROMETER X')
accel_y = get_col(df_m, 'ACCELEROMETER Y')
steering = get_col(df_v_interp, 'Steering Angle')
if accel_x and accel_y and steering and v_speed_col:
    straight_mask = (df_v_interp[steering].abs() < 2) & (df_v_interp[v_speed_col] > 10)
    if straight_mask.sum() > 0:
        df_straight = df_m[straight_mask]
        plt.figure(figsize=(6, 6))
        plt.scatter(df_straight[accel_x], df_straight[accel_y], alpha=0.1)
        plt.title('Accel X vs Y during straight driving')
        plt.xlabel('Accel X')
        plt.ylabel('Accel Y')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/3_accel_straight.png')
        plt.close()

# --- Phase 4: Velocity Ground Truth Quality ---
print("Phase 4: Velocity Ground Truth Quality")
plt.figure(figsize=(12, 5))
if gps_speed_col:
    plt.plot(df_m['time_s'], df_m[gps_speed_col], label='GPS Speed (Mobile)', alpha=0.7)
if v_speed_col:
    plt.plot(df_m['time_s'], df_v_interp[v_speed_col], label='Vehicle Speed', alpha=0.7)

ws_fl = get_col(df_v_interp, 'Wheel Speed Front Left')
ws_fr = get_col(df_v_interp, 'Wheel Speed Front Right')
ws_rl = get_col(df_v_interp, 'Wheel Speed Rear Left')
ws_rr = get_col(df_v_interp, 'Wheel Speed Rear Right')
ws_cols = [c for c in [ws_fl, ws_fr, ws_rl, ws_rr] if c]

if len(ws_cols) == 4:
    r = 0.3 # assumed wheel radius in meters
    avg_ws = df_v_interp[ws_cols].mean(axis=1) # rad/s
    wheel_speed_kmh = avg_ws * r * 3.6
    plt.plot(df_m['time_s'], wheel_speed_kmh, label='Derived Wheel Speed (r=0.3m)', alpha=0.7)

plt.title('Velocity Ground Truth Comparison (First 5 mins)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.xlim(0, 300)
plt.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/4_velocity_comparison.png')
plt.close()

if ws_fl and ws_fr and steering:
    plt.figure(figsize=(10, 4))
    ws_diff = df_v_interp[ws_fl] - df_v_interp[ws_fr]
    plt.plot(df_m['time_s'], ws_diff, label='FL - FR wheel speed (rad/s)')
    plt.plot(df_m['time_s'], df_v_interp[steering] / 10, label='Steering Angle (scaled / 10)')
    plt.title('Wheel Speed Diff vs Steering Angle')
    plt.xlim(0, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/4_wheel_speed_turns.png')
    plt.close()

print("Preprocessing and plots complete!")
