#!/usr/bin/env python3
"""
Heading/yaw diagnostic for Cynos.

Fixes the misleading full-drive overlay:
- preserves ZERO lag correction and nearest-time synchronization
- handles heading wrap with circular differences
- compares yaw-rate-derived heading change over multiple horizons
- reports both raw sign conventions and an optional fitted scale
- excludes windows crossing large timestamp gaps / invalid heading or yaw samples
- creates a readable 3-panel diagnostic over a configurable time slice
- does not filter or modify the source data

Run from project root:
  python train/heading_yaw_diagnostic_fixed.py
Optional:
  python train/heading_yaw_diagnostic_fixed.py --start 400 --duration 120
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_MOBILE = Path(
    "Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    "Vw (Driver E)/Vw04/S-Vw4.csv"
)
DEFAULT_VEHICLE = Path(
    "Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    "Vw (Driver E)/Vw04/V-Vw4.csv"
)
DEFAULT_OUT = Path("data/heading_diagnostics_fixed")
DT_MAX_S = 0.30
SPEED_THRESHOLD_KMH = 5.0
HORIZONS_S = (1, 5, 10, 30, 60)


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_col(df, candidates, required=True):
    cols = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in cols:
            return cols[key]
    for candidate in candidates:
        key = norm(candidate)
        hits = [c for nk, c in cols.items() if key in nk]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"Missing one of {candidates}; available={list(df.columns)}")
    return None


def wrap_deg(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def get_elapsed(df, candidates, scale):
    col = find_col(df, candidates, required=False)
    if col is None:
        raise KeyError(f"Could not find elapsed-time column from {candidates}")
    return pd.to_numeric(df[col], errors="coerce") * scale, col


def load_sync(mobile_path, vehicle_path):
    m = pd.read_csv(mobile_path, encoding="latin1")
    v = pd.read_csv(vehicle_path, encoding="latin1")
    m.columns = m.columns.str.strip()
    v.columns = v.columns.str.strip()

    mt, mt_name = get_elapsed(m, ["TIME SINCE START (ms)", "Time Since Start (ms)", "TIME SINCE START"], 0.001)
    vt, vt_name = get_elapsed(v, ["Time Since Start of Day (seconds)", "Time Since Start of Day", "Time Since Start (seconds)"], 1.0)
    m["_t"] = mt
    v["_t"] = vt
    m = m.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
    v = v.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
    m["_t"] -= m["_t"].iloc[0]
    v["_t"] -= v["_t"].iloc[0]

    # Use documented phone gyroscope fields; names are kept neutral because
    # phone axes are not assumed to coincide with vehicle axes.
    gyro_cols = [
        find_col(m, ["GYROSCOPE Yaw", "GYROSCOPE X"]),
        find_col(m, ["GYROSCOPE Pitch", "GYROSCOPE Y"]),
        find_col(m, ["GYROSCOPE Roll", "GYROSCOPE Z"]),
    ]
    yaw_col = find_col(v, ["Yaw Rate (deg/sec)", "Yaw Rate", "Yaw rate"])
    head_col = find_col(v, ["Heading (degrees)", "Heading (deg)", "Heading"])
    speed_col = find_col(v, ["Velocity (km/hr)", "Velocity (km/h)", "Velocity"])

    mm = m[["_t", *gyro_cols]].copy()
    mm.columns = ["_t", "gyro_1", "gyro_2", "gyro_3"]
    vv = v[["_t", yaw_col, head_col, speed_col]].copy()
    vv.columns = ["_t", "vehicle_yaw_rate_deg_s", "vehicle_heading_deg", "vehicle_velocity_kmh"]

    for c in mm.columns[1:]:
        mm[c] = pd.to_numeric(mm[c], errors="coerce")
    for c in vv.columns[1:]:
        vv[c] = pd.to_numeric(vv[c], errors="coerce")

    merged = pd.merge_asof(
        vv.sort_values("_t"), mm.sort_values("_t"),
        on="_t", direction="nearest", tolerance=0.2
    )
    merged = merged.dropna(subset=["vehicle_heading_deg", "vehicle_yaw_rate_deg_s"]).reset_index(drop=True)
    print(f"Mobile timestamp: {mt_name}; vehicle timestamp: {vt_name}")
    return merged


def make_horizon_table(df):
    t = df["_t"].to_numpy(float)
    h = df["vehicle_heading_deg"].to_numpy(float)
    yaw = df["vehicle_yaw_rate_deg_s"].to_numpy(float)
    speed = df["vehicle_velocity_kmh"].to_numpy(float)

    rows = []
    # Pair endpoints nearest the requested duration, but require all samples
    # between endpoints to have normal cadence and valid fields.
    for horizon in HORIZONS_S:
        step = int(round(horizon / np.median(np.diff(t))))
        for i in range(0, len(df) - step):
            j = i + step
            if abs((t[j] - t[i]) - horizon) > 0.15:
                continue
            sl = slice(i, j + 1)
            ts = t[sl]
            if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0) or np.any(np.diff(ts) > DT_MAX_S):
                continue
            if not np.all(np.isfinite(h[sl])) or not np.all(np.isfinite(yaw[sl])):
                continue
            # Reject intervals containing obvious heading telemetry jumps.
            dh_step = wrap_deg(np.diff(h[sl]))
            if np.any(np.abs(dh_step) >= 30.0):
                continue
            # Trapezoidal integration over the vehicle yaw-rate series.
            integrated = np.trapezoid(yaw[sl], ts)
            true_change = float(wrap_deg(h[j] - h[i]))
            rows.append({
                "horizon_s": horizon,
                "t_start_s": t[i],
                "t_end_s": t[j],
                "speed_start_kmh": speed[i],
                "true_heading_change_deg": true_change,
                "integrated_yaw_deg": float(integrated),
                "error_plus_deg": float(wrap_deg(integrated - true_change)),
                "error_minus_deg": float(wrap_deg(-integrated - true_change)),
            })
    return pd.DataFrame(rows)


def summarize(table):
    print("\nHORIZON-WISE VEHICLE YAW-RATE vs VEHICLE HEADING CHANGE")
    print("Windows with heading jumps >=30° or timestamp gaps >0.3s are excluded.")
    if table.empty:
        print("No valid windows.")
        return
    for horizon, g in table.groupby("horizon_s"):
        y = g["true_heading_change_deg"].to_numpy()
        x = g["integrated_yaw_deg"].to_numpy()
        # Fit scale on this diagnostic population, explicitly label as fitted.
        denom = float(np.dot(x, x))
        scale = float(np.dot(x, y) / denom) if denom > 1e-12 else np.nan
        fitted_err = wrap_deg(scale * x - y)
        print(f"\n{int(horizon):>2}s: N={len(g)}")
        for label, err in [
            ("raw +yaw", wrap_deg(x-y)),
            ("raw -yaw", wrap_deg(-x-y)),
            ("fitted scale", fitted_err),
        ]:
            ae = np.abs(err)
            print(f"  {label:14s} MAE={np.mean(ae):7.3f}°  median={np.median(ae):7.3f}°  P90={np.percentile(ae,90):7.3f}°")
        print(f"  fitted scale: {scale:+.6f} (diagnostic calibration, not held-out)")


def plot_slice(df, table, out_dir, start, duration):
    end = start + duration
    d = df[(df["_t"] >= start) & (df["_t"] <= end)].copy()
    if d.empty:
        print(f"No samples in requested plot range [{start}, {end}] s")
        return

    t = d["_t"].to_numpy(float)
    h = d["vehicle_heading_deg"].to_numpy(float)
    yaw = d["vehicle_yaw_rate_deg_s"].to_numpy(float)
    speed = d["vehicle_velocity_kmh"].to_numpy(float)

    # Unwrap only for visualization; raw heading remains in CSV.
    h_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(h)))
    # Trapezoidal integration, anchored at first heading in this slice.
    integ = np.zeros(len(d), dtype=float)
    if len(d) > 1:
        integ[1:] = np.cumsum((yaw[:-1] + yaw[1:]) * 0.5 * np.diff(t))
    pred_unwrapped = h_unwrapped[0] - integ  # negative sign per observed calibration

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(t, h_unwrapped, label="vehicle heading (unwrapped)", lw=1.1)
    axes[0].plot(t, pred_unwrapped, label="heading from integrated -yaw rate", lw=1.0)
    axes[0].set_ylabel("Unwrapped heading (°)")
    axes[0].legend()
    axes[0].grid(alpha=.25)
    yaw=-yaw
    axes[1].plot(t, yaw, label="vehicle yaw rate", lw=.8)
    axes[1].plot(t, np.gradient(h_unwrapped, t), label="d(unwrapped heading)/dt", lw=.8, alpha=.8)
    axes[1].set_ylabel("Rate (°/s)")
    axes[1].legend()
    axes[1].grid(alpha=.25)

    axes[2].plot(t, speed, lw=.8, label="vehicle speed")
    axes[2].axhline(SPEED_THRESHOLD_KMH, ls="--", lw=.8, label="5 km/h reference")
    axes[2].set_ylabel("Speed (km/h)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(alpha=.25)

    fig.suptitle(f"Heading/yaw diagnostic: {start:.1f}–{end:.1f}s (zero lag)")
    fig.tight_layout()
    fig.savefig(out_dir / f"heading_yaw_slice_{int(start)}_{int(duration)}s.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mobile", type=Path, default=DEFAULT_MOBILE)
    p.add_argument("--vehicle", type=Path, default=DEFAULT_VEHICLE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--start", type=float, default=0.0, help="plot start time in seconds")
    p.add_argument("--duration", type=float, default=120.0, help="plot duration in seconds")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Mobile: {args.mobile}\nVehicle: {args.vehicle}\nOutput: {args.out.resolve()}")
    df = load_sync(args.mobile, args.vehicle)
    print(f"Synchronized rows: {len(df)}")
    dt = np.diff(df["_t"].to_numpy(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt):
        print(f"Median dt={np.median(dt):.4f}s; approximate rate={1/np.median(dt):.2f}Hz")

    table = make_horizon_table(df)
    table.to_csv(args.out / "heading_yaw_horizon_windows.csv", index=False)
    summarize(table)
    plot_slice(df, table, args.out, args.start, args.duration)
    df.to_csv(args.out / "synchronized_heading_yaw_fixed.csv", index=False)
    print(f"\nWrote diagnostics to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
