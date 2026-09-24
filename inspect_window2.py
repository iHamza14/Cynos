import pandas as pd
import numpy as np
import sys
sys.path.append("train")
from data.generate_dataset import make_bins

df_s = pd.read_csv("/tmp/Vw04/S-Vw4.csv", low_memory=False, encoding='latin1')
df_v = pd.read_csv("/tmp/Vw04/V-Vw4.csv", low_memory=False, encoding='latin1')
df_s.columns = df_s.columns.str.strip()
df_v.columns = df_v.columns.str.strip()
resampled = make_bins(df_s, df_v)
valid_data = [d for d in resampled if not np.isnan(d['v_lat']) and not np.isnan(d['v_lon'])]

window_2_blackout = valid_data[610:670]
print("Time, v_lat, v_lon, v_speed_ms")
for i, d in enumerate(window_2_blackout[:20]):
    print(f"{i}: {d['v_lat']:.6f}, {d['v_lon']:.6f}, {d['v_speed_ms']:.2f}")
