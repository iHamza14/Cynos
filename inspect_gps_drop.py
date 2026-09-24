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

w = valid_data[300+10:300+70]  # Window 1 blackout
for i in range(15):
    print(f"t={i}: lat={w[i]['v_lat']:.6f}, speed={w[i]['v_speed_ms']:.2f}")
