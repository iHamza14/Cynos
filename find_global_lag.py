import pandas as pd
import numpy as np
import scipy.signal

print("Loading data...")
S_PATH = "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"
V_PATH = "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"

def parse_mobile_clock(value):
    text = str(value).strip()
    try:
        time_part = text.split()[-1]
        fields = time_part.replace("_", ":").split(":")
        if len(fields) != 4: return np.nan
        hh, mm, ss, ms = fields
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except: return np.nan

def find_col(df, prefix):
    for c in df.columns:
        if c.startswith(prefix): return c
    raise KeyError(prefix)

s = pd.read_csv(S_PATH, encoding="latin-1")
v = pd.read_csv(V_PATH, encoding="latin-1")

s.columns = s.columns.str.strip()
v.columns = v.columns.str.strip()

s["abs_time"] = s[find_col(s, "DATE")].map(parse_mobile_clock)
s = s.dropna(subset=["abs_time"]).sort_values("abs_time")

v["abs_time"] = pd.to_numeric(v[find_col(v, "Time Since Start of Day")], errors="coerce")
v = v.dropna(subset=["abs_time"]).sort_values("abs_time")

s_accel_y = pd.to_numeric(s[find_col(s, "ACCELEROMETER Y")], errors='coerce').fillna(0).to_numpy()
s_time = s["abs_time"].to_numpy()

v_speed = pd.to_numeric(v[find_col(v, "Velocity (km/hr)")], errors='coerce').fillna(0).to_numpy() / 3.6
v_time = v["abs_time"].to_numpy()

print("Interpolating to common 10Hz grid...")
start = max(s_time[0], v_time[0])
end = min(s_time[-1], v_time[-1])
grid = np.arange(start, end, 0.1)

accel_grid = np.interp(grid, s_time, s_accel_y)
speed_grid = np.interp(grid, v_time, v_speed)
v_accel_grid = np.gradient(speed_grid, 0.1)

print("Computing cross-correlation...")
A = v_accel_grid - np.mean(v_accel_grid)
B = accel_grid - np.mean(accel_grid)

corr = scipy.signal.correlate(A, B, mode='full')
lags = scipy.signal.correlation_lags(len(A), len(B))
best_lag_idx = np.argmax(corr)
best_lag_samples = lags[best_lag_idx]
best_lag_sec = best_lag_samples * 0.1

print(f"Optimal global shift: shift IMU by {best_lag_sec} seconds to align with OBD.")
