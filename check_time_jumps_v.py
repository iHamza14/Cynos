import pandas as pd
v = pd.read_csv("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv", encoding="latin-1")
v.columns = v.columns.str.strip()
import numpy as np

time = v['Time Since Start of Day (seconds)'].dropna().to_numpy(dtype=float)
dt = np.diff(time)
jumps = np.where(dt > 2)[0] # jumps > 2s

print(f"Number of time jumps > 2s: {len(jumps)}")
for j in jumps[:5]:
    print(f"Jump at index {j}: dt={dt[j]} s")
