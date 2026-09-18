"""
Build fixed-context, 60-second blackout episodes.

Assumptions explicitly supplied by the project owner:
- 10 Hz resampling; context = 10 s (100 samples).
- One 60 s blackout per episode (600 samples); evaluate prefixes at configured checkpoints.
- Mobile GPS SPEED column is mislabeled "Kmh" but values are to be treated as m/s.
- Vehicle Velocity is km/h and is converted to m/s.
- Use vehicle Velocity directly; do not estimate wheel radius or wheel speed.
- No IMU/speed lag correction in dataset generation.
- Keep GPS observations/labels in blackout for evaluation, but isolate them under
  `evaluation_only` so they cannot accidentally be fed to the model.
- Keep chronological 80/20 split and train-only scaler fitting.
- Apply only the requested speed filter: reject windows whose mean vehicle speed
  over the 60 s blackout is <= 5 km/h. No GPS-quality filter.

Road labels are heuristic descriptive tags, not ground-truth road classifications.
They use wrapped vehicle-heading change, vehicle steering angle, and yaw rate.
Thresholds are exposed below and should be reviewed against plotted examples.
"""

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────
HZ = 10
CONTEXT_SEC = 10
BLACKOUT_SEC = 60
CHECKPOINTS_SEC = [2, 5, 10, 15, 30, 60]
TRAIN_STRIDE_SEC = 15
TEST_STRIDE_SEC = 30
TRAIN_FRACTION = 0.80
MIN_MEAN_BLACKOUT_SPEED_KMH = 5.0

# Nominal road-label thresholds; labels are descriptive, not authoritative.
STRAIGHT_MAX_HEADING_RATE_DPS = 5.0
CURVE_MIN_HEADING_RATE_DPS = 5.0
UTURN_MIN_ABS_HEADING_CHANGE_DEG = 135.0
ROUNDABOUT_MIN_ABS_HEADING_CHANGE_DEG = 180.0
ROUNDABOUT_MIN_DURATION_SEC = 8.0
STEERING_ACTIVE_DEG = 8.0

S_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"
)
V_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"
)
OUT_DIR = "data"


def find_col(df, *needles):
    """Prefer exact normalized names, then require a unique substring match."""
    normalized = {" ".join(str(c).strip().split()).casefold(): c for c in df.columns}
    exact = " ".join(" ".join(needles).split()).casefold()
    if exact in normalized:
        return normalized[exact]
    hits = [
        c for c in df.columns
        if all(n.casefold() in str(c).strip().casefold() for n in needles)
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one column matching {needles}; found {len(hits)}: {hits}\n"
            f"Available columns: {list(df.columns)}"
        )
    return hits[0]


def parse_mobile_clock(value):
    """Parse DATE values like YYYY-MM-DD HH:MM:SS:ms (also accepts SS_ms)."""
    text = str(value).strip()
    try:
        time_part = text.split()[-1]  # tolerate optional date prefix
        fields = time_part.replace("_", ":").split(":")
        if len(fields) != 4:
            return np.nan
        hh, mm, ss, ms = fields
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except (ValueError, TypeError, IndexError):
        return np.nan


def wrap_degrees(angle):
    """Map angles to [-180, 180)."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def make_bins(df_s, df_v):
    """Nearest-time join, then interpolate onto a 10 Hz grid."""
    s = df_s.copy()
    v = df_v.copy()
    s.columns = s.columns.str.strip()
    v.columns = v.columns.str.strip()

    s_date = find_col(s, "DATE")
    s["abs_time"] = s[s_date].map(parse_mobile_clock)
    s = s.dropna(subset=["abs_time"]).sort_values("abs_time")

    v_time = find_col(v, "Time Since Start of Day")
    v["abs_time"] = pd.to_numeric(v[v_time], errors="coerce")
    v = v.dropna(subset=["abs_time"]).sort_values("abs_time")

    # Select the documented vehicle fields. Do not derive speed from wheel rates.
    vehicle_cols = {
        "v_speed_kmh": find_col(v, "Velocity (km/hr)"),
        "v_heading_deg": find_col(v, "Heading (degrees)"),
        "v_lat": find_col(v, "Latitude (degrees)"),
        "v_lon": find_col(v, "Longitude (degrees)"),
        "v_height_km": find_col(v, "Height (km)"),
        "v_steering_deg": find_col(v, "Steering Angle (degrees)"),
        "v_yaw_rate_dps": find_col(v, "Yaw Rate (deg/sec)"),
    }
    v_keep = v[["abs_time", *vehicle_cols.values()]].rename(
        columns={col: key for key, col in vehicle_cols.items()}
    )

    # Mobile fields, retaining the original values and units stated by user.
    mobile_cols = {
        "mobile_lat_deg": find_col(s, "GPS LATITUDE"),
        "mobile_lon_deg": find_col(s, "GPS LONGITUDE"),
        "mobile_alt_m": find_col(s, "GPS ALTITUDE"),
        "mobile_speed_ms": find_col(s, "GPS SPEED"),
        "gps_accuracy_m": find_col(s, "GPS ACCURACY"),
        "gps_orientation_deg": find_col(s, "GPS ORIENTATION"),
        "gps_satellites": find_col(s, "GPS SATELLITES IN RANGE"),
        "acc_x": find_col(s, "ACCELEROMETER X"),
        "acc_y": find_col(s, "ACCELEROMETER Y"),
        "acc_z": find_col(s, "ACCELEROMETER Z"),
        "gravity_x": find_col(s, "GRAVITY X"),
        "gravity_y": find_col(s, "GRAVITY Y"),
        "gravity_z": find_col(s, "GRAVITY Z"),
        "gyro_yaw": find_col(s, "GYROSCOPE Yaw"),
        "gyro_pitch": find_col(s, "GYROSCOPE Pitch"),
        "gyro_roll": find_col(s, "GYROSCOPE Roll"),
        "mag_x": find_col(s, "MAGNETIC FIELD X"),
        "mag_y": find_col(s, "MAGNETIC FIELD Y"),
        "mag_z": find_col(s, "MAGNETIC FIELD Z"),
        "orientation_yaw_deg": find_col(s, "ORIENTATION (Yaw)"),
        "orientation_pitch_deg": find_col(s, "ORIENTATION (Pitch)"),
        "orientation_roll_deg": find_col(s, "ORIENTATION (Roll"),
    }
    s_keep = s[["abs_time", *mobile_cols.values()]].rename(
        columns={col: key for key, col in mobile_cols.items()}
    )

    merged = pd.merge_asof(
        s_keep.sort_values("abs_time"),
        v_keep.sort_values("abs_time"),
        on="abs_time",
        direction="nearest",
        tolerance=0.2,
    ).dropna(subset=["v_speed_kmh"])

    # Coerce relevant values to numeric, preserving NaNs for explicit window checks.
    for col in merged.columns:
        if col != "abs_time":
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    start = np.ceil(merged["abs_time"].min() * HZ) / HZ
    end = np.floor(merged["abs_time"].max() * HZ) / HZ
    grid = np.round(np.arange(start, end + 0.5 / HZ, 1.0 / HZ), 6)

    numeric_cols = [c for c in merged.columns if c != "abs_time"]
    resampled = pd.DataFrame({"abs_time": grid})
    source_t = merged["abs_time"].to_numpy(dtype=float, copy=True)

    for col in numeric_cols:
        vals = merged[col].to_numpy(dtype=float, copy=True)
        valid = np.isfinite(source_t) & np.isfinite(vals)
        if valid.sum() < 2:
            resampled[col] = np.nan
        elif col in {"v_heading_deg", "gps_orientation_deg"}:
            radians = np.radians(vals[valid])
            sin_i = np.interp(grid, source_t[valid], np.sin(radians))
            cos_i = np.interp(grid, source_t[valid], np.cos(radians))
            resampled[col] = np.degrees(np.arctan2(sin_i, cos_i)) % 360.0
        else:
            resampled[col] = np.interp(grid, source_t[valid], vals[valid])

    # Circular interpolation prevents artificial jumps across 0/360 degrees.
    # It does not validate the vehicle heading's physical convention or quality.
    bins = []
    for _, row in resampled.iterrows():
        bins.append({
            "abs_sec": float(row["abs_time"]),
            "raw_accel": np.array([row.acc_x, row.acc_y, row.acc_z], dtype=float),
            "raw_gyro": np.array([row.gyro_yaw, row.gyro_pitch, row.gyro_roll], dtype=float),
            "gravity": np.array([row.gravity_x, row.gravity_y, row.gravity_z], dtype=float),
            "magnetic_field": np.array([row.mag_x, row.mag_y, row.mag_z], dtype=float),
            "mobile_gps_speed": float(row.mobile_speed_ms),
            "mobile_lat": float(row.mobile_lat_deg),
            "mobile_lon": float(row.mobile_lon_deg),
            "mobile_alt": float(row.mobile_alt_m),
            "mobile_gps_orientation": float(row.gps_orientation_deg),
            "gps_acc": float(row.gps_accuracy_m),
            "gps_sats": float(row.gps_satellites),
            "mobile_orientation_ypr": np.array(
                [row.orientation_yaw_deg, row.orientation_pitch_deg, row.orientation_roll_deg],
                dtype=float,
            ),
            "v_speed_kmh": float(row.v_speed_kmh),
            "v_speed_ms": float(row.v_speed_kmh) / 3.6,
            "v_heading": float(row.v_heading_deg),
            "v_lat": float(row.v_lat),
            "v_lon": float(row.v_lon),
            "v_height_km": float(row.v_height_km),
            "v_steering_deg": float(row.v_steering_deg),
            "v_yaw_rate_dps": float(row.v_yaw_rate_dps),
        })
    return bins


def label_road_segment(blackout_bins):
    """Return nominal window-level labels using vehicle signals only."""
    headings = np.array([b["v_heading"] for b in blackout_bins], dtype=float)
    steering = np.array([b["v_steering_deg"] for b in blackout_bins], dtype=float)
    yaw_rate = np.array([b["v_yaw_rate_dps"] for b in blackout_bins], dtype=float)

    valid_h = np.isfinite(headings)
    if valid_h.sum() < 2:
        return {"primary": "unknown", "tags": ["unknown"], "metrics": {}}

    # Wrapped step changes avoid a false ~360-degree jump at north.
    dhead = wrap_degrees(np.diff(headings))
    heading_rate = dhead * HZ
    mean_abs_rate = float(np.nanmean(np.abs(heading_rate))) if len(heading_rate) else np.nan
    total_abs_change = float(np.nansum(np.abs(dhead)))
    net_change = float(abs(wrap_degrees(headings[-1] - headings[0])))
    steering_valid = np.abs(steering[np.isfinite(steering)])
    yaw_valid = np.abs(yaw_rate[np.isfinite(yaw_rate)])
    steering_active = bool(
        steering_valid.size and np.percentile(steering_valid, 75) >= STEERING_ACTIVE_DEG
    )
    yaw_active = bool(
        yaw_valid.size and np.percentile(yaw_valid, 75) >= CURVE_MIN_HEADING_RATE_DPS
    )

    tags = []
    duration = len(blackout_bins) / HZ
    if net_change >= ROUNDABOUT_MIN_ABS_HEADING_CHANGE_DEG and duration >= ROUNDABOUT_MIN_DURATION_SEC:
        tags.append("roundabout_candidate")
    if net_change >= UTURN_MIN_ABS_HEADING_CHANGE_DEG:
        tags.append("uturn_candidate")

    if np.isfinite(mean_abs_rate) and mean_abs_rate <= STRAIGHT_MAX_HEADING_RATE_DPS and not steering_active:
        tags.append("straight_candidate")
    elif (
        (np.isfinite(mean_abs_rate) and mean_abs_rate >= CURVE_MIN_HEADING_RATE_DPS)
        or steering_active
        or yaw_active
    ):
        tags.append("curved_candidate")

    if not tags:
        tags.append("uncertain")
    # Primary is a nominal convenience label; preserve all candidate tags too.
    priority = ["roundabout_candidate", "uturn_candidate", "curved_candidate", "straight_candidate"]
    primary = next((x for x in priority if x in tags), "uncertain")
    return {
        "primary": primary,
        "tags": tags,
        "metrics": {
            "net_heading_change_deg": net_change,
            "total_absolute_heading_change_deg": total_abs_change,
            "mean_absolute_heading_rate_dps": mean_abs_rate,
            "steering_active": steering_active,
            "yaw_rate_active": yaw_active,
        },
    }


def displacement_from_latlon(bins):
    lat = np.radians(np.array([b["v_lat"] for b in bins], dtype=float))
    lon = np.radians(np.array([b["v_lon"] for b in bins], dtype=float))
    R = 6_371_000.0
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    mid = (lat[:-1] + lat[1:]) / 2
    north = dlat * R
    east = dlon * R * np.cos(mid)
    return np.vstack(([0.0, 0.0], np.cumsum(np.column_stack((north, east)), axis=0)))


def build_windows(bins, stride_sec):
    context_n = CONTEXT_SEC * HZ
    blackout_n = BLACKOUT_SEC * HZ
    stride_n = stride_sec * HZ
    windows = []
    rejected = Counter()

    for start in range(0, len(bins) - context_n - blackout_n + 1, stride_n):
        ctx = bins[start:start + context_n]
        blk = bins[start + context_n:start + context_n + blackout_n]
        all_bins = ctx + blk

        if len(ctx) != context_n or len(blk) != blackout_n:
            rejected["wrong_length"] += 1
            continue
        times = np.array([b["abs_sec"] for b in all_bins])
        if not np.all(np.isfinite(times)) or np.max(np.diff(times)) > 0.15:
            rejected["time_gap_or_invalid_time"] += 1
            continue

        mean_speed_kmh = float(np.nanmean([b["v_speed_kmh"] for b in blk]))
        if not np.isfinite(mean_speed_kmh) or mean_speed_kmh <= MIN_MEAN_BLACKOUT_SPEED_KMH:
            rejected["mean_blackout_speed_le_5_kmh"] += 1
            continue

        # Reject invalid model IMU or required GT values. GPS fields may be NaN;
        # they remain evaluation-only and are not quality-filtered.
        required_arrays = (
            [b["raw_accel"] for b in all_bins]
            + [b["raw_gyro"] for b in all_bins]
            + [b["gravity"] for b in all_bins]
        )
        required_gt = [
            b["v_speed_ms"] for b in blk
        ] + [b["v_heading"] for b in blk] + [
            b["v_lat"] for b in blk
        ] + [b["v_lon"] for b in blk]
        if not all(np.all(np.isfinite(a)) for a in required_arrays) or not np.all(np.isfinite(required_gt)):
            rejected["nonfinite_required_imu_or_gt"] += 1
            continue

        vr_seed = float(ctx[-1]["mobile_gps_speed"])  # user specifies m/s
        gt_speeds = np.array([b["v_speed_ms"] for b in blk], dtype=float)
        gt_delta_v = np.empty(blackout_n, dtype=float)
        gt_delta_v[0] = gt_speeds[0] - float(ctx[-1]["v_speed_ms"])
        gt_delta_v[1:] = np.diff(gt_speeds)

        gps_eval = {
            "mobile_speed_ms": np.array([b["mobile_gps_speed"] for b in blk], dtype=float),
            "mobile_lat_deg": np.array([b["mobile_lat"] for b in blk], dtype=float),
            "mobile_lon_deg": np.array([b["mobile_lon"] for b in blk], dtype=float),
            "mobile_alt_m": np.array([b["mobile_alt"] for b in blk], dtype=float),
            "gps_accuracy_m": np.array([b["gps_acc"] for b in blk], dtype=float),
            "gps_orientation_deg": np.array([b["mobile_gps_orientation"] for b in blk], dtype=float),
            "gps_satellites": np.array([b["gps_sats"] for b in blk], dtype=float),
        }

        context = {
            "raw_accel": np.stack([b["raw_accel"] for b in ctx]),
            "raw_gyro": np.stack([b["raw_gyro"] for b in ctx]),
            "gravity": np.stack([b["gravity"] for b in ctx]),
            "vr_seed_ms": vr_seed,
        }
        blackout = {
            "raw_accel": np.stack([b["raw_accel"] for b in blk]),
            "raw_gyro": np.stack([b["raw_gyro"] for b in blk]),
            "gravity": np.stack([b["gravity"] for b in blk]),
        }

        disp = displacement_from_latlon(blk)
        road_label = label_road_segment(blk)
        ground_truth = {
            "speeds_ms": gt_speeds,
            "speeds_kmh": np.array([b["v_speed_kmh"] for b in blk], dtype=float),
            "headings_deg": np.array([b["v_heading"] for b in blk], dtype=float),
            "headings_rad": np.radians([b["v_heading"] for b in blk]),
            "delta_v_ms": gt_delta_v,
            "cumulative_displacement_ne_m": disp,
            "total_distance_m": float(np.sum(np.linalg.norm(np.diff(disp, axis=0), axis=1))),
            "start_lat": float(blk[0]["v_lat"]),
            "start_lon": float(blk[0]["v_lon"]),
            "end_lat": float(blk[-1]["v_lat"]),
            "end_lon": float(blk[-1]["v_lon"]),
        }

        windows.append({
            "context": context,
            "blackout": blackout,
            "ground_truth": ground_truth,
            "evaluation_only": {"gps": gps_eval},
            "metadata": {
                "context_sec": CONTEXT_SEC,
                "blackout_sec": BLACKOUT_SEC,
                "checkpoints_sec": CHECKPOINTS_SEC,
                "window_start_sec": float(ctx[0]["abs_sec"]),
                "blackout_start_sec": float(blk[0]["abs_sec"]),
                "stride_sec": stride_sec,
                "road_label": road_label,
                "mean_vehicle_speed_kmh": mean_speed_kmh,
                "imu_lag_applied": False,
            },
        })

    return windows, rejected


def main():
    df_s = pd.read_csv(S_PATH, encoding="latin-1")
    df_v = pd.read_csv(V_PATH, encoding="latin-1")
    bins = make_bins(df_s, df_v)
    print(f"Resampled bins: {len(bins)}")

    # Keep chronological split. Split boundary is explicit; windows are generated
    # within each partition, so no window crosses the boundary.
    split_idx = int(len(bins) * TRAIN_FRACTION)
    train_bins, test_bins = bins[:split_idx], bins[split_idx:]

    train_windows, train_rej = build_windows(train_bins, TRAIN_STRIDE_SEC)
    test_windows, test_rej = build_windows(test_bins, TEST_STRIDE_SEC)

    if not train_windows:
        raise RuntimeError("No training windows generated; inspect data/filter thresholds.")

    # Fit scalers on training blackout IMU only, matching the old scaling scope.
    # This excludes all test samples and all GPS/ground-truth fields.
    scalers = {}
    for field in ("raw_accel", "raw_gyro", "gravity"):
        values = np.vstack([w["blackout"][field] for w in train_windows])
        scalers[field] = StandardScaler().fit(values)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train_blackout_windows.pkl"), "wb") as f:
        pickle.dump(train_windows, f)
    with open(os.path.join(OUT_DIR, "test_blackout_windows.pkl"), "wb") as f:
        pickle.dump(test_windows, f)
    with open(os.path.join(OUT_DIR, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(OUT_DIR, "blackout_manifest.pkl"), "wb") as f:
        pickle.dump({
            "hz": HZ,
            "context_sec": CONTEXT_SEC,
            "blackout_sec": BLACKOUT_SEC,
            "checkpoints_sec": CHECKPOINTS_SEC,
            "train_stride_sec": TRAIN_STRIDE_SEC,
            "test_stride_sec": TEST_STRIDE_SEC,
            "train_count": len(train_windows),
            "test_count": len(test_windows),
            "train_rejected": dict(train_rej),
            "test_rejected": dict(test_rej),
            "imu_lag_applied": False,
            "speed_filter": f"mean blackout vehicle speed > {MIN_MEAN_BLACKOUT_SPEED_KMH} km/h",
        }, f)

    print(f"Train windows: {len(train_windows)}; rejected: {dict(train_rej)}")
    print(f"Test windows:  {len(test_windows)}; rejected: {dict(test_rej)}")
    print(f"Saved outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
