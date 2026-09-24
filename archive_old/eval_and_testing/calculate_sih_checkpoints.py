"""
Utility module: calculate_sih_checkpoints.py.
"""

import pandas as pd
import pickle
import numpy as np

# Load test windows to get ground truth speed
with open("../data/test_blackout_windows.pkl", "rb") as f:
    test_items = pickle.load(f)

# Load drift data
drift_df = pd.read_csv("map_output/detailed_drift_1hz.csv")

checkpoints = [15, 30, 60]
results = []
sample_data = {cp: [] for cp in checkpoints}

for cp in checkpoints:
    cp_drift = drift_df[drift_df["time_sec"] == cp]
    
    pass_count = 0
    total_valid = 0
    drift_pcts = []
    
    for _, row in cp_drift.iterrows():
        ep_idx = int(row["episode_idx"])
        drift_m = row["drift_m"]
        
        # Ground truth speeds at 10Hz. cp seconds = cp * 10 samples
        speeds = test_items[ep_idx]["ground_truth"]["speeds_ms"]
        # Total distance traveled up to checkpoint
        dist_m = np.sum(speeds[:cp * 10]) * 0.1
        
        if dist_m > 1.0: # avoid division by zero
            pct = (drift_m / dist_m) * 100
            drift_pcts.append(pct)
            if pct < 10.0:
                pass_count += 1
            total_valid += 1
            
            if ep_idx < 3: # Save a few examples
                sample_data[cp].append({
                    "Episode": ep_idx,
                    "Dist (m)": dist_m,
                    "Drift (m)": drift_m,
                    "Drift %": pct
                })
    
    if total_valid > 0:
        mean_pct = np.mean(drift_pcts)
        median_pct = np.median(drift_pcts)
        pass_rate = (pass_count / total_valid) * 100
        
        results.append({
            "Checkpoint": f"{cp}s",
            "Mean Drift %": f"{mean_pct:.2f}%",
            "Median Drift %": f"{median_pct:.2f}%",
            "Pass Rate (<10%)": f"{pass_rate:.1f}%"
        })

print("=== OVERALL BENCHMARK METRICS ===")
out_df = pd.DataFrame(results)
print(out_df.to_string(index=False))

print("\n=== EXAMPLES (Episodes 0, 1, 2) ===")
for cp in checkpoints:
    print(f"\n-- {cp} Seconds --")
    ex_df = pd.DataFrame(sample_data[cp])
    print(ex_df.to_string(index=False, float_format="%.2f"))

