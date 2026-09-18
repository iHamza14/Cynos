import numpy as np
import pandas as pd
import pickle
import os

from helper import bin_dataset, cumulative_displacement, total_distance_m
from sklearn.preprocessing import StandardScaler


# ── Config ─────────────────────────────────────────────────────────────────────
CONTEXT_SEC = 10
BLACKOUT_DURATIONS = [15, 30, 60]
BLACKOUT_STEP = 30
MIN_SPEED_KMPH = 5
MAX_GPS_ACC = 10
MIN_GPS_SATS = 6
HZ = 10

# check.py reported IMU-speed lag of -5 samples.
# Assumption: IMU lags vehicle speed by 5 samples.
IMU_LAG_SAMPLES = 5


# ── IMU-Speed Alignment ────────────────────────────────────────────────────────
def apply_imu_speed_lag(bins, lag_samples=IMU_LAG_SAMPLES):
    """
    Align IMU signals with vehicle-speed timestamps.

    Assumption:
        IMU lags vehicle speed by lag_samples.

    Only IMU fields are shifted:
        raw_accel, raw_gyro, gravity

    Vehicle speed, heading, position, GPS fields, and timestamps
    remain unchanged.
    """
    if lag_samples < 0:
        raise ValueError("lag_samples must be non-negative")

    if lag_samples == 0:
        return bins

    if len(bins) <= lag_samples:
        raise ValueError("Not enough bins to apply requested lag")

    aligned_bins = []

    # For each target timestamp i, use the IMU sample i + lag.
    for i in range(len(bins) - lag_samples):
        aligned_bin = bins[i].copy()
        imu_source = bins[i + lag_samples]

        aligned_bin["raw_accel"] = imu_source["raw_accel"]
        aligned_bin["raw_gyro"] = imu_source["raw_gyro"]
        aligned_bin["gravity"] = imu_source["gravity"]

        aligned_bins.append(aligned_bin)

    print(
        f"Applied IMU-speed lag correction: {lag_samples} samples "
        f"({lag_samples / HZ:.1f}s)"
    )
    print(f"Bins before alignment: {len(bins)}")
    print(f"Bins after alignment : {len(aligned_bins)}")

    return aligned_bins


# ── Main 10Hz Builder ─────────────────────────────────────────────────────────
def build_blackout_windows(
    bins,
    blackout_durations=BLACKOUT_DURATIONS,
    context_sec=CONTEXT_SEC,
    step_sec=BLACKOUT_STEP,
):
    windows = []
    total_steps = len(bins)

    context_steps = context_sec * HZ
    step_size = step_sec * HZ

    for blackout_dur_sec in blackout_durations:
        blackout_steps = blackout_dur_sec * HZ
        window_len_steps = context_steps + blackout_steps

        for i in range(
            0,
            total_steps - window_len_steps + 1,
            step_size,
        ):
            context_bins = bins[i : i + context_steps]
            blackout_bins = bins[
                i + context_steps : i + window_len_steps
            ]

            # ── 1. Structural Guards ──
            if len(context_bins) != context_steps:
                continue

            if len(blackout_bins) != blackout_steps:
                continue

            times = [b["abs_sec"] for b in context_bins + blackout_bins]

            if np.max(np.diff(times)) > 0.15:
                continue

            # ── 2. Quality Guards ──
            bad_gps = sum(
                1
                for b in context_bins
                if b["gps_acc"] > MAX_GPS_ACC
                or b["gps_sats"] < MIN_GPS_SATS
            )

            if bad_gps > 0:
                continue

            avg_speed = np.mean(
                [b["v_odo_speed"] * 3.6 for b in blackout_bins]
            )

            if avg_speed < MIN_SPEED_KMPH:
                continue

            # ── 3. Context Pack ──
            last_ctx = context_bins[-1]

            context = {
                "raw_accel": np.stack(
                    [b["raw_accel"] for b in context_bins]
                ),
                "raw_gyro": np.stack(
                    [b["raw_gyro"] for b in context_bins]
                ),
                "gravity": np.stack(
                    [b["gravity"] for b in context_bins]
                ),
                "vr_seed": float(last_ctx["mobile_gps_speed"]),
            }

            # ── 4. Blackout Pack ──
            blackout = {
                "raw_accel": np.stack(
                    [b["raw_accel"] for b in blackout_bins]
                ),
                "raw_gyro": np.stack(
                    [b["raw_gyro"] for b in blackout_bins]
                ),
                "gravity": np.stack(
                    [b["gravity"] for b in blackout_bins]
                ),
            }

            # ── 5. Ground Truth ──
            gt_cum_disp = cumulative_displacement(blackout_bins)

            gt_speeds = np.array(
                [b["v_odo_speed"] for b in blackout_bins]
            )

            gt_headings = np.array(
                [np.radians(b["v_heading"]) for b in blackout_bins]
            )

            gt_delta_v = np.zeros(blackout_steps)
            gt_delta_v[0] = gt_speeds[0] - context["vr_seed"]

            if blackout_steps > 1:
                gt_delta_v[1:] = np.diff(gt_speeds)

            # ── 6. Cumulative Distance ──
            cum_dist_array = np.zeros(blackout_steps)

            if blackout_steps > 1:
                lats = np.radians(
                    [b["v_lat"] for b in blackout_bins]
                )
                lons = np.radians(
                    [b["v_lon"] for b in blackout_bins]
                )

                R = 6371000.0

                d_lats = np.diff(lats)
                d_lons = np.diff(lons)
                mid_lats = (lats[:-1] + lats[1:]) / 2.0

                step_dists = np.sqrt(
                    (d_lats * R) ** 2
                    + (d_lons * R * np.cos(mid_lats)) ** 2
                )

                cum_dist_array[1:] = np.cumsum(step_dists)

            ground_truth = {
                "cum_disp_m": gt_cum_disp,
                "speeds_ms": gt_speeds,
                "headings_rad": gt_headings,
                "delta_v": gt_delta_v,
                "total_distance_m": total_distance_m(blackout_bins),
                "cumulative_dist_m": cum_dist_array,
                "start_lat": float(blackout_bins[0]["v_lat"]),
                "start_lon": float(blackout_bins[0]["v_lon"]),
                "end_lat": float(blackout_bins[-1]["v_lat"]),
                "end_lon": float(blackout_bins[-1]["v_lon"]),
            }

            windows.append(
                {
                    "context": context,
                    "blackout": blackout,
                    "ground_truth": ground_truth,
                    "blackout_dur_sec": blackout_dur_sec,
                    "context_dur_sec": context_sec,
                    "window_start_sec": context_bins[0]["abs_sec"],
                }
            )

    print(
        f"Generated {len(windows)} 10Hz blackout windows "
        f"across durations {blackout_durations}s"
    )

    for duration in blackout_durations:
        count = sum(
            1
            for w in windows
            if w["blackout_dur_sec"] == duration
        )
        print(f"  {duration:3d}s blackout: {count} windows")

    return windows


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s_path = (
        "../Synchronised V abd S datasets/"
        "Categorised IOVNB Dataset/Vw (Driver E)/"
        "Vw04/S-Vw4.csv"
    )

    v_path = (
        "../Synchronised V abd S datasets/"
        "Categorised IOVNB Dataset/Vw (Driver E)/"
        "Vw04/V-Vw4.csv"
    )

    df_s = pd.read_csv(s_path, encoding="latin-1")
    df_s.columns = df_s.columns.str.strip()

    df_v = pd.read_csv(v_path, encoding="latin-1")
    df_v.columns = df_v.columns.str.strip()

    # Initial synchronization remains unchanged.
    bins = bin_dataset(df_s, df_v)

    # Apply ONLY the IMU-to-speed lag correction.
    bins = apply_imu_speed_lag(
        bins,
        lag_samples=IMU_LAG_SAMPLES,
    )

    # Chronological split.
    split_idx = int(len(bins) * 0.8)
    train_bins = bins[:split_idx]
    test_bins = bins[split_idx:]

    print("\nGenerating Training Windows (Fixed 60s)...")
    train_windows = build_blackout_windows(
        train_bins,
        blackout_durations=[60],
        step_sec=15,
    )

    print("\nGenerating Test Windows (15s, 30s, 60s)...")
    test_windows = build_blackout_windows(
        test_bins,
        blackout_durations=[15, 30, 60],
        step_sec=30,
    )

    # ── Fit scalers on training windows only ──
    all_raw_accel = np.vstack(
        [w["blackout"]["raw_accel"] for w in train_windows]
    )

    all_raw_gyro = np.vstack(
        [w["blackout"]["raw_gyro"] for w in train_windows]
    )

    all_gravity = np.vstack(
        [w["blackout"]["gravity"] for w in train_windows]
    )

    scalers = {
        "raw_accel": StandardScaler().fit(all_raw_accel),
        "raw_gyro": StandardScaler().fit(all_raw_gyro),
        "gravity": StandardScaler().fit(all_gravity),
    }

    os.makedirs("data", exist_ok=True)

    with open("data/train_blackout_windows.pkl", "wb") as f:
        pickle.dump(train_windows, f)

    with open("data/test_blackout_windows.pkl", "wb") as f:
        pickle.dump(test_windows, f)

    with open("data/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    print(
        f"\nSaved {len(train_windows)} train windows and "
        f"{len(test_windows)} test windows + scalers to data/"
    )