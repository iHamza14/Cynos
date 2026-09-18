import sys
import os
import pickle
import numpy as np
import scipy.signal
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data.blackout_dataset import make_bins, S_PATH, V_PATH, HZ, CONTEXT_SEC, BLACKOUT_SEC, MIN_MEAN_BLACKOUT_SPEED_KMH, TRAIN_FRACTION, TRAIN_STRIDE_SEC, TEST_STRIDE_SEC

print("Loading CSVs...")
df_s = pd.read_csv(S_PATH, encoding="latin-1")
df_v = pd.read_csv(V_PATH, encoding="latin-1")
print("Making bins...")
bins = make_bins(df_s, df_v)
print(f"Resampled bins: {len(bins)}")

def build_windows_aligned(partition_bins, stride_sec, is_train=True):
    context_n = int(CONTEXT_SEC * HZ)
    blackout_n = int(BLACKOUT_SEC * HZ)
    stride_n = int(stride_sec * HZ)
    windows = []
    rejected = Counter()

    last_lag = 0
    search_pad = 600

    for start in range(0, len(partition_bins) - context_n - blackout_n + 1, stride_n):
        # We need search_pad for correlation
        search_start = max(0, start - search_pad)
        search_end = min(len(partition_bins), start + context_n + blackout_n + search_pad)
        
        search_bins = partition_bins[search_start:search_end]
        
        imu_accel = np.array([b["raw_accel"][1] for b in search_bins])
        gt_speed = np.array([b["v_speed_ms"] for b in search_bins])
        gt_accel = np.gradient(gt_speed)
        
        A = gt_accel
        B = imu_accel
        
        if len(A) == len(B) and len(A) > 200:
            var_A = np.var(A)
            var_B = np.var(B)
            
            if var_A > 0.005 and var_B > 0.5:
                A_norm = A - np.mean(A)
                B_norm = B - np.mean(B)
                corr = scipy.signal.correlate(A_norm, B_norm, mode='full')
                lags = scipy.signal.correlation_lags(len(A), len(B))
                best_idx = np.argmax(corr)
                
                # Confidence check
                confidence = corr[best_idx] / (np.linalg.norm(A_norm) * np.linalg.norm(B_norm))
                
                if confidence > 0.2:
                    last_lag = lags[best_idx]
                else:
                    rejected["low_confidence"] += 1
        
        # Apply lag to find OBD start
        obd_start = start + last_lag
        
        if obd_start < 0 or obd_start + context_n + blackout_n > len(partition_bins):
            rejected["alignment_out_of_bounds"] += 1
            continue
            
        ctx = partition_bins[start:start + context_n]
        blk = partition_bins[start + context_n:start + context_n + blackout_n]
        all_bins = ctx + blk
        
        obd_ctx = partition_bins[obd_start:obd_start + context_n]
        obd_blk = partition_bins[obd_start + context_n:obd_start + context_n + blackout_n]
        obd_all = obd_ctx + obd_blk

        if len(ctx) != context_n or len(blk) != blackout_n or len(obd_ctx) != context_n or len(obd_blk) != blackout_n:
            rejected["wrong_length"] += 1
            continue
            
        times = np.array([b["abs_sec"] for b in all_bins])
        obd_times = np.array([b["abs_sec"] for b in obd_all])
        if not np.all(np.isfinite(times)) or np.max(np.diff(times)) > 0.15 or not np.all(np.isfinite(obd_times)) or np.max(np.diff(obd_times)) > 0.15:
            rejected["time_gap"] += 1
            continue

        mean_speed = float(np.nanmean([b["v_speed_kmh"] for b in obd_blk]))
        if not np.isfinite(mean_speed) or mean_speed <= MIN_MEAN_BLACKOUT_SPEED_KMH:
            rejected["low_speed"] += 1
            continue

        required_arrays = (
            [b["raw_accel"] for b in all_bins]
            + [b["raw_gyro"] for b in all_bins]
            + [b["gravity"] for b in all_bins]
        )
        required_gt = [
            b["v_speed_ms"] for b in obd_blk
        ] + [b["v_heading"] for b in obd_blk] + [
            b["v_lat"] for b in obd_blk
        ] + [b["v_lon"] for b in obd_blk]
        
        if not all(np.all(np.isfinite(a)) for a in required_arrays) or not np.all(np.isfinite(required_gt)):
            rejected["nonfinite"] += 1
            continue

        vr_seed = float(obd_ctx[-1]["v_speed_ms"])
        gt_speeds = np.array([b["v_speed_ms"] for b in obd_blk], dtype=float)
        gt_delta_v = np.empty(blackout_n, dtype=float)
        gt_delta_v[0] = gt_speeds[0] - float(obd_ctx[-1]["v_speed_ms"])
        gt_delta_v[1:] = np.diff(gt_speeds)

        gps_eval = {
            "mobile_speed_ms": np.array([b["mobile_gps_speed"] for b in obd_blk], dtype=float),
            "mobile_lat": np.array([b["mobile_lat"] for b in obd_blk], dtype=float),
            "mobile_lon": np.array([b["mobile_lon"] for b in obd_blk], dtype=float),
            "mobile_alt": np.array([b["mobile_alt"] for b in obd_blk], dtype=float),
            "gps_acc": np.array([b["gps_acc"] for b in obd_blk], dtype=float),
            "gps_sats": np.array([b["gps_sats"] for b in obd_blk], dtype=float),
        }

        w = {
            "context": {
                "raw_accel": np.array([b["raw_accel"] for b in ctx], dtype=float),
                "raw_gyro": np.array([b["raw_gyro"] for b in ctx], dtype=float),
                "gravity": np.array([b["gravity"] for b in ctx], dtype=float),
                "vr_seed_ms": vr_seed,
            },
            "blackout": {
                "raw_accel": np.array([b["raw_accel"] for b in blk], dtype=float),
                "raw_gyro": np.array([b["raw_gyro"] for b in blk], dtype=float),
                "gravity": np.array([b["gravity"] for b in blk], dtype=float),
            },
            "ground_truth": {
                "speeds_ms": gt_speeds,
                "delta_v_ms": gt_delta_v,
                "label": {"primary": "aligned", "tags": [], "metrics": {}},
                "gps_eval": gps_eval,
                "debug": {"lag_applied": last_lag}
            },
        }
        windows.append(w)
    return windows, rejected

print("Building aligned windows...")
split_idx = int(len(bins) * TRAIN_FRACTION)
train_bins, test_bins = bins[:split_idx], bins[split_idx:]

train_windows, train_rej = build_windows_aligned(train_bins, TRAIN_STRIDE_SEC, is_train=True)
test_windows, test_rej = build_windows_aligned(test_bins, TEST_STRIDE_SEC, is_train=False)

print(f"Train windows: {len(train_windows)} (Rejected: {train_rej})")
print(f"Test windows: {len(test_windows)} (Rejected: {test_rej})")

scalers = {}
for field in ("raw_accel", "raw_gyro", "gravity"):
    values = np.vstack([w["blackout"][field] for w in train_windows])
    scalers[field] = StandardScaler().fit(values)

os.makedirs("data", exist_ok=True)
with open("data/train_blackout_windows.pkl", "wb") as f:
    pickle.dump(train_windows, f)
with open("data/test_blackout_windows.pkl", "wb") as f:
    pickle.dump(test_windows, f)

print("Saved aligned dataset!")
with open("data/scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)
