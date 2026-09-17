import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bin_builder import bin_dataset
from data.window_builder import build_blackout_windows

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    s_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
    v_path = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'
    
    # If those paths don't exist, we should warn
    if not os.path.exists(s_path) or not os.path.exists(v_path):
        print(f"Warning: Expected datasets at {s_path} and {v_path}")
        print("Please place the CSV files in the correct relative directory or update the paths.")
        return

    df_s = pd.read_csv(s_path, encoding='latin-1')
    df_s.columns = df_s.columns.str.strip()
    df_v = pd.read_csv(v_path, encoding='latin-1')
    df_v.columns = df_v.columns.str.strip()

    bins = bin_dataset(df_s, df_v)

    # Chronological split — blackout windows come from TEST bins only
    # You never evaluate DR on data the model trained on
    split_idx = int(len(bins) * 0.8)
    train_bins = bins[:split_idx]
    test_bins = bins[split_idx:]

    print("Generating Training Windows (Fixed 60s for batching)...")
    train_windows = build_blackout_windows(train_bins, blackout_durations=[60], step_sec=15)
    
    print("Generating Test Windows (15s, 30s, 60s)...")
    test_windows = build_blackout_windows(test_bins, blackout_durations=[15, 30, 60], step_sec=30)
    
    all_raw_accel = np.vstack([w['blackout']['raw_accel'] for w in train_windows])
    all_raw_gyro = np.vstack([w['blackout']['raw_gyro'] for w in train_windows])
    all_gravity = np.vstack([w['blackout']['gravity'] for w in train_windows])

    scalers = {
        'raw_accel': StandardScaler().fit(all_raw_accel),
        'raw_gyro': StandardScaler().fit(all_raw_gyro),
        'gravity': StandardScaler().fit(all_gravity)
    }

    os.makedirs('data', exist_ok=True)
    with open('data/train_blackout_windows.pkl', 'wb') as f:
        pickle.dump(train_windows, f)
    with open('data/test_blackout_windows.pkl', 'wb') as f:
        pickle.dump(test_windows, f)
    with open('data/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    print(f"Saved {len(train_windows)} Train windows and {len(test_windows)} Test windows + Scalers to data/")

if __name__ == "__main__":
    main()
