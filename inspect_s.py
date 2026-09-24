import sys
sys.path.append("train")
from data.generate_dataset import make_bins
import pandas as pd
df_s = pd.read_csv("/tmp/Vw04/S-Vw4.csv", encoding='latin1', low_memory=False, nrows=100)
df_v = pd.read_csv("/tmp/Vw04/V-Vw4.csv", encoding='latin1', low_memory=False, nrows=100)
df_s.columns = df_s.columns.str.strip()
df_v.columns = df_v.columns.str.strip()
bins = make_bins(df_s, df_v)
print(bins[0].keys())
