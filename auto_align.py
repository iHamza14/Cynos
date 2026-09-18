import pandas as pd
import numpy as np
import scipy.signal

print("Loading data...")
S_PATH = "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"
V_PATH = "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"

def parse_mobile_clock(value):
    try:
        fields = str(value).strip().split()[-1].replace("_", ":").split(":")
        return int(fields[0])*3600 + int(fields[1])*60 + int(fields[2]) + int(fields[3])/1000.
    except: return np.nan

def find_col(df, prefix):
    for c in df.columns:
        if c.startswith(prefix): return c

s = pd.read_csv(S_PATH, encoding="latin-1")
v = pd.read_csv(V_PATH, encoding="latin-1")
s.columns = s.columns.str.strip()
v.columns = v.columns.str.strip()

s["abs_time"] = s[find_col(s, "DATE")].map(parse_mobile_clock)
s = s.dropna(subset=["abs_time"]).sort_values("abs_time")
v["abs_time"] = pd.to_numeric(v[find_col(v, "Time Since Start of Day")], errors="coerce")
v = v.dropna(subset=["abs_time"]).sort_values("abs_time")

s_t = s["abs_time"].to_numpy()
s_a = pd.to_numeric(s[find_col(s, "ACCELEROMETER Y")], errors='coerce').fillna(0).to_numpy()
v_t = v["abs_time"].to_numpy()
v_s = pd.to_numeric(v[find_col(v, "Velocity (km/hr)")], errors='coerce').fillna(0).to_numpy() / 3.6

print("Interpolating to 10Hz grid...")
start = max(s_t[0], v_t[0])
end = min(s_t[-1], v_t[-1])
grid = np.arange(start, end, 0.1)

accel = np.interp(grid, s_t, s_a)
speed = np.interp(grid, v_t, v_s)
vaccel = np.gradient(speed, 0.1)

print("Computing dynamic alignment...")
chunk_size = 1200 # 2 minutes
overlap = 600

times = []
lags = []
confidences = []

for i in range(0, len(grid) - chunk_size, chunk_size - overlap):
    A = vaccel[i:i+chunk_size]
    B = accel[i:i+chunk_size]
    
    A_var = np.var(A)
    B_var = np.var(B)
    
    if A_var > 0.1 and B_var > 0.1:
        A_norm = A - np.mean(A)
        B_norm = B - np.mean(B)
        
        corr = scipy.signal.correlate(A_norm, B_norm, mode='full')
        corr_lags = scipy.signal.correlation_lags(len(A), len(B))
        
        best_lag_idx = np.argmax(corr)
        lag = corr_lags[best_lag_idx]
        
        times.append(grid[i + chunk_size//2])
        lags.append(lag * 0.1)
        
        # Calculate a pseudo-confidence
        confidences.append(corr[best_lag_idx] / (np.linalg.norm(A_norm) * np.linalg.norm(B_norm)))
        
times = np.array(times)
lags = np.array(lags)
confidences = np.array(confidences)

print(f"Found {len(lags)} aligned segments.")

# Filter out bad alignments
valid = confidences > 0.2
times = times[valid]
lags = lags[valid]

print(f"Kept {len(lags)} high-confidence segments.")

if len(lags) > 0:
    # Interpolate lag for all points in s
    # We want to shift s["abs_time"] by the lag!
    # If lag is positive, B (accel) is shifted right relative to A.
    # We want to shift B left, so we subtract lag from s["abs_time"].
    interpolated_lags = np.interp(s["abs_time"].to_numpy(), times, lags)
    
    s["abs_time_corrected"] = s["abs_time"] + interpolated_lags
    
    # Save the aligned s dataset
    s.to_csv("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4-Aligned.csv", index=False)
    print("Saved S-Vw4-Aligned.csv")
else:
    print("Could not find any reliable alignments!")

