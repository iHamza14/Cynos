import pandas as pd
import numpy as np
import scipy.signal

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

start = max(s_t[0], v_t[0])
end = min(s_t[-1], v_t[-1])
grid = np.arange(start, end, 0.1)

accel = np.interp(grid, s_t, s_a)
speed = np.interp(grid, v_t, v_s)
vaccel = np.gradient(speed, 0.1)

# Chunk every 10 minutes (6000 samples)
chunk_size = 6000
lags = []
times = []
for i in range(0, len(grid) - chunk_size, chunk_size):
    A = vaccel[i:i+chunk_size]
    B = accel[i:i+chunk_size]
    A -= np.mean(A)
    B -= np.mean(B)
    # Only if there's enough variance
    if np.var(A) > 0.05 and np.var(B) > 0.05:
        corr = scipy.signal.correlate(A, B, mode='full')
        lag = np.argmax(corr) - len(B) + 1
        lags.append(lag * 0.1)
        times.append(grid[i])

for t, l in zip(times, lags):
    print(f"Time: {t:.1f}, Lag: {l:.1f}s")
