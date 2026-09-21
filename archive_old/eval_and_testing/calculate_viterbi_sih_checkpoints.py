import pandas as pd
import pickle
import numpy as np

# Load test windows to get ground truth speed
with open("./data/test_blackout_windows.pkl", "rb") as f:
    test_items = pickle.load(f)

# Load drift data (with Viterbi snapper results)
drift_df = pd.read_csv("./train/map_output_driver_e/detailed_drift_1hz_with_snapper.csv")

checkpoints = [15, 30, 60]
results = []

for cp in checkpoints:
    cp_drift = drift_df[drift_df["time_sec"] == cp]
    
    pass_count = 0
    total_valid = 0
    raw_pcts = []
    viterbi_pcts = []
    
    for _, row in cp_drift.iterrows():
        ep_idx = int(row["episode_idx"])
        raw_drift = row["raw_drift_m"]
        snapped_drift = row["snapped_drift_m"]
        
        speeds = test_items[ep_idx]["ground_truth"]["speeds_ms"]
        dist_m = np.sum(speeds[:cp * 10]) * 0.1
        
        if dist_m > 1.0:
            raw_pct = (raw_drift / dist_m) * 100
            viterbi_pct = (snapped_drift / dist_m) * 100
            
            raw_pcts.append(raw_pct)
            viterbi_pcts.append(viterbi_pct)
            
            if viterbi_pct < 10.0:
                pass_count += 1
            total_valid += 1
            
    if total_valid > 0:
        results.append({
            "Checkpoint": f"{cp}s",
            "Raw Mean %": f"{np.mean(raw_pcts):.2f}%",
            "Viterbi Mean %": f"{np.mean(viterbi_pcts):.2f}%",
            "Viterbi Median %": f"{np.median(viterbi_pcts):.2f}%",
            "Pass Rate (<10%)": f"{(pass_count / total_valid) * 100:.1f}%"
        })

out_df = pd.DataFrame(results)
print(out_df.to_string(index=False))
