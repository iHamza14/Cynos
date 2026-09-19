import pandas as pd

df = pd.read_csv("map_output/detailed_drift_1hz.csv")
checkpoints = [2, 5, 10, 15, 30, 60]
results = []

for cp in checkpoints:
    cp_data = df[df["time_sec"] == cp]["drift_m"]
    if len(cp_data) > 0:
        results.append({
            "Time(s)": cp,
            "Mean(m)": round(cp_data.mean(), 2),
            "Median(m)": round(cp_data.median(), 2),
            "P90(m)": round(cp_data.quantile(0.90), 2),
            "Max(m)": round(cp_data.max(), 2)
        })

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
