import pandas as pd
import pickle

# Load test windows to get ground truth total distance
with open("../data/test_blackout_windows.pkl", "rb") as f:
    test_items = pickle.load(f)

distances = []
for i, item in enumerate(test_items):
    dist = item["ground_truth"].get("total_distance_m", 0.0)
    distances.append({"episode_idx": i, "total_distance_m": dist})

dist_df = pd.DataFrame(distances)

# Load drift data
drift_df = pd.read_csv("map_output/detailed_drift_1hz.csv")

# Get final drift at 60 seconds
final_drift = drift_df[drift_df["time_sec"] == 60][["episode_idx", "drift_m"]]

# Merge
merged = pd.merge(final_drift, dist_df, on="episode_idx")

# Prevent division by zero
merged["total_distance_m"] = merged["total_distance_m"].replace(0, 1e-6)

# Calculate Percentage Drift
merged["drift_percentage"] = (merged["drift_m"] / merged["total_distance_m"]) * 100

# Compute Benchmark Metrics
mean_drift_pct = merged["drift_percentage"].mean()
median_drift_pct = merged["drift_percentage"].median()
pass_rate = (merged["drift_percentage"] < 10.0).mean() * 100

print(f"Total Test Episodes: {len(merged)}")
print(f"Mean Drift (% of distance): {mean_drift_pct:.2f}%")
print(f"Median Drift (% of distance): {median_drift_pct:.2f}%")
print(f"Episodes meeting < 10% benchmark: {pass_rate:.1f}%")
print("\nSample Episodes (first 5):")
print(merged[["episode_idx", "total_distance_m", "drift_m", "drift_percentage"]].head(5).to_string(index=False, float_format="%.2f"))
