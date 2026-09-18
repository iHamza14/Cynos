import pandas as pd
s = pd.read_csv("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv", encoding="latin-1")
s.columns = s.columns.str.strip()
import numpy as np

# 'TIME SINCE START (ms)'
time_ms = s['TIME SINCE START (ms)'].dropna().to_numpy(dtype=float)
dt = np.diff(time_ms)
jumps = np.where(dt > 1000)[0] # jumps greater than 1s

print(f"Number of time jumps > 1s: {len(jumps)}")
for j in jumps[:5]:
    print(f"Jump at index {j}: dt={dt[j]} ms")

