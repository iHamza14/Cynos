#!/usr/bin/env python3
"""
Leakage-aware phone-gyro heading diagnostic with a minimal blackout EKF.

Pipeline:
  1. Synchronize mobile and vehicle streams (zero relative-origin lag).
  2. Subtract supplied stationary phone gyro bias (rad/s).
  3. Fit gyro-to-vehicle-yaw-rate calibration on the first chronological 80%.
  4. Freeze calibration and evaluate on the final 20%.
  5. During test inference, integrate phone gyro only. The EKF receives NO
     vehicle yaw-rate or heading measurements during the test interval.
  6. Save metrics, timeseries CSV, and plots.

Important: calibration labels come from vehicle yaw rate in the training split.
This is an offline supervised calibration diagnostic, not a claim that the
mapping generalizes across phones, mounting orientations, or trips.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

GYRO_BIAS_RAD_S = np.array([-0.0020, 0.0047, -0.0022], dtype=float)  # yaw,pitch,roll
TRAIN_FRACTION = 0.80
MAX_DT_S = 0.3
MAX_HEADING_JUMP_DEG = 30.0


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_col(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None) -> str:
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    if contains:
        hits = [c for c in df.columns if all(term in norm(c) for term in contains)]
        if len(hits) == 1:
            return hits[0]
    raise KeyError(f"Cannot find column candidates={candidates}; columns={list(df.columns)}")


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def mobile_time_seconds(df: pd.DataFrame) -> pd.Series:
    col = find_col(df, ["TIME SINCE START (ms)", "Time Since Start (ms)", "TIME SINCE START"],
                   contains=["timesincestart"])
    return numeric(df, col) / 1000.0


def vehicle_time_seconds(df: pd.DataFrame) -> pd.Series:
    col = find_col(df, ["Time Since Start of Day (seconds)", "Time Since Start Of Day"],
                   contains=["timesin"])
    return numeric(df, col)


def integrate_rate(rate_deg_s: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros(len(rate_deg_s), dtype=float)
    if len(out) > 1:
        dt = np.diff(t)
        valid = np.isfinite(dt) & (dt > 0) & (dt <= MAX_DT_S)
        increments = np.zeros(len(dt), dtype=float)
        increments[valid] = 0.5 * (rate_deg_s[:-1][valid] + rate_deg_s[1:][valid]) * dt[valid]
        out[1:] = np.cumsum(increments)
    return out


def minimal_ekf_integrate(rate_deg_s: np.ndarray, t: np.ndarray,
                          initial_bias_deg_s: float = 0.0,
                          bias_process_noise: float = 1e-5):
    """
    Minimal 2-state inertial EKF: state=[relative heading (deg), gyro bias (deg/s)].
    It has prediction only during blackout: no vehicle/GNSS measurement updates.
    Bias is initialized to zero because the supplied stationary bias was already
    subtracted before calibration. Process noise lets uncertainty grow; without
    observations the bias estimate itself remains unobservable and fixed.
    """
    n = len(t)
    heading = np.zeros(n)
    bias = np.full(n, initial_bias_deg_s)
    if n == 0:
        return heading, bias

    P = np.diag([1.0, 0.25])
    Q_bias = float(bias_process_noise)
    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt <= 0 or dt > MAX_DT_S:
            heading[k] = heading[k - 1]
            bias[k] = bias[k - 1]
            continue
        heading[k] = heading[k - 1] + (rate_deg_s[k - 1] - bias[k - 1]) * dt
        bias[k] = bias[k - 1]
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        Q = np.diag([1e-4 * dt, Q_bias * dt])
        P = F @ P @ F.T + Q
    return heading, bias


def weighted_fit(X: np.ndarray, y: np.ndarray, turn_weight: float = 3.0):
    # Upweight nontrivial turning rates, but cap influence.
    weights = 1.0 + (turn_weight - 1.0) * np.clip(np.abs(y) / 15.0, 0.0, 1.0)
    model = LinearRegression().fit(X, y, sample_weight=weights)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobile", type=Path, default=Path(
        "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"))
    ap.add_argument("--vehicle", type=Path, default=Path(
        "Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"))
    ap.add_argument("--synced-csv", type=Path, default=None,
                    help="CSV with _t, vehicle_yaw_rate_deg_s, vehicle_heading_deg, vehicle_speed_kmh, gyro_1_rad_s..gyro_3_rad_s")
    ap.add_argument("--out", type=Path, default=Path("data/heading_diagnostics/blackout_ekf"))
    ap.add_argument("--tolerance-ms", type=float, default=200.0)
    ap.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    ap.add_argument("--plot-start", type=float, default=400.0)
    ap.add_argument("--plot-duration", type=float, default=120.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synced_csv:
        df = pd.read_csv(args.synced_csv, encoding="latin1", low_memory=False)
        df.columns = [str(c).strip().lstrip("\\ufeff") for c in df.columns]
    else:
        m = pd.read_csv(args.mobile, encoding="latin1", low_memory=False, skipinitialspace=True)
        v = pd.read_csv(args.vehicle, encoding="latin1", low_memory=False, skipinitialspace=True)
        m.columns = [str(c).strip().lstrip("\\ufeff") for c in m.columns]
        v.columns = [str(c).strip().lstrip("\\ufeff") for c in v.columns]

        m["_t_raw"] = mobile_time_seconds(m)
        v["_t_raw"] = vehicle_time_seconds(v)
        m = m[np.isfinite(m["_t_raw"])].sort_values("_t_raw").copy()
        v = v[np.isfinite(v["_t_raw"])].sort_values("_t_raw").copy()
        m["_t"] = m["_t_raw"] - m["_t_raw"].iloc[0]
        v["_t"] = v["_t_raw"] - v["_t_raw"].iloc[0]

        gyro_cols = [
            find_col(m, ["GYROSCOPE Yaw"], contains=["gyroscope", "yaw"]),
            find_col(m, ["GYROSCOPE Pitch"], contains=["gyroscope", "pitch"]),
            find_col(m, ["GYROSCOPE Roll"], contains=["gyroscope", "roll"]),
        ]
        yaw_col = find_col(v, ["Yaw Rate (deg/sec)", "Yaw Rate (deg/s)", "Yaw Rate"], contains=["yaw", "rate"])
        head_col = find_col(v, ["Heading (degrees)", "Heading (degree)", "Heading"], contains=["heading"])
        speed_col = find_col(v, ["Velocity (km/hr)", "Velocity (km/h)", "Velocity"], contains=["velocity"])

        ms = m[["_t", *gyro_cols]].copy()
        for i, c in enumerate(gyro_cols, 1):
            ms[f"gyro_{i}_rad_s"] = numeric(ms, c)
        ms = ms[["_t", "gyro_1_rad_s", "gyro_2_rad_s", "gyro_3_rad_s"]]
        vs = v[["_t", yaw_col, head_col, speed_col]].copy()
        vs["vehicle_yaw_rate_deg_s"] = numeric(vs, yaw_col)
        vs["vehicle_heading_deg"] = numeric(vs, head_col)
        vs["vehicle_speed_kmh"] = numeric(vs, speed_col)
        df = pd.merge_asof(vs.sort_values("_t"), ms.sort_values("_t"), on="_t",
                           direction="nearest", tolerance=args.tolerance_ms / 1000.0)
        df = df.dropna(subset=["vehicle_yaw_rate_deg_s", "gyro_1_rad_s", "gyro_2_rad_s", "gyro_3_rad_s"])

    required = ["_t", "vehicle_yaw_rate_deg_s", "vehicle_heading_deg",
                "gyro_1_rad_s", "gyro_2_rad_s", "gyro_3_rad_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("_t").drop_duplicates("_t").reset_index(drop=True)
    t = df["_t"].to_numpy(float)
    dt = np.diff(t, prepend=t[0])
    vehicle_ref = -pd.to_numeric(df["vehicle_yaw_rate_deg_s"], errors="coerce").to_numpy(float)
    raw_gyro = df[["gyro_1_rad_s", "gyro_2_rad_s", "gyro_3_rad_s"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    corrected_deg_s = np.rad2deg(raw_gyro - GYRO_BIAS_RAD_S)

    raw_heading = pd.to_numeric(df["vehicle_heading_deg"], errors="coerce").to_numpy(float)
    jump = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        dhead = (raw_heading[1:] - raw_heading[:-1] + 180.0) % 360.0 - 180.0
        jump[1:] = np.abs(dhead) >= MAX_HEADING_JUMP_DEG

    valid = (np.isfinite(t) & np.isfinite(vehicle_ref) &
             np.all(np.isfinite(corrected_deg_s), axis=1) &
             (dt >= 0) & (dt <= MAX_DT_S) & (~jump))
    if len(valid):
        valid[0] = False

    split = int(len(df) * args.train_fraction)
    train_mask = valid.copy()
    train_mask[split:] = False
    test_mask = valid.copy()
    test_mask[:split] = False
    if train_mask.sum() < 20 or test_mask.sum() < 20:
        raise RuntimeError(f"Not enough valid train/test rows: train={train_mask.sum()}, test={test_mask.sum()}")

    # Train-only weighted linear mapping from corrected phone XYZ to vehicle yaw rate.
    fusion_model = weighted_fit(corrected_deg_s[train_mask], vehicle_ref[train_mask])
    calibrated_rate = fusion_model.predict(corrected_deg_s)

    # Test inference: reset relative heading at test boundary, integrate phone-only rate.
    test_indices = np.flatnonzero(test_mask)
    first, last = int(test_indices[0]), int(test_indices[-1]) + 1
    contiguous = np.arange(first, last)
    t_test = t[contiguous]
    rate_test = calibrated_rate[contiguous]
    ref_test = vehicle_ref[contiguous]
    # Keep invalid points as NaN for plots, but integration holds over invalid gaps.
    rate_for_ekf = np.where(np.isfinite(rate_test), rate_test, 0.0)
    ekf_heading, ekf_bias = minimal_ekf_integrate(rate_for_ekf, t_test)
    ref_heading = integrate_rate(np.where(np.isfinite(ref_test), ref_test, 0.0), t_test)
    phone_integrated = integrate_rate(rate_for_ekf, t_test)

    # Evaluate only valid contiguous horizon windows, no window crosses a gap.
    rows = []
    median_dt = float(np.nanmedian(np.diff(t_test)))
    for horizon in [1, 5, 10, 30, 60]:
        steps = max(1, int(round(horizon / median_dt)))
        if steps >= len(t_test):
            continue
        errs = []
        for i in range(0, len(t_test) - steps):
            j = i + steps
            if not np.all(np.isfinite(ref_test[i:j+1])) or not np.all(np.isfinite(rate_test[i:j+1])):
                continue
            dts = np.diff(t_test[i:j+1])
            if np.any(dts <= 0) or np.any(dts > MAX_DT_S):
                continue
            target = ref_heading[j] - ref_heading[i]
            pred = ekf_heading[j] - ekf_heading[i]
            errs.append(abs(pred - target))
        rows.append({"horizon_s": horizon, "N": len(errs),
                     "MAE_heading_change_deg": float(np.mean(errs)) if errs else np.nan,
                     "median_abs_error_deg": float(np.median(errs)) if errs else np.nan,
                     "P90_abs_error_deg": float(np.percentile(errs, 90)) if errs else np.nan})
    pd.DataFrame(rows).to_csv(args.out / "heldout_ekf_heading_metrics.csv", index=False)

    coef = fusion_model.coef_
    print(f"Rows={len(df)}; train valid={train_mask.sum()}; held-out valid={test_mask.sum()}")
    print("Train-fitted phone XYZ -> -vehicle yaw-rate mapping:")
    print(f"  coefficients (per deg/s): {coef}")
    print(f"  intercept (deg/s): {fusion_model.intercept_:.6f}")
    print("Held-out heading-change metrics:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot test segment: zero all relative headings at test interval start.
    plot_idx = np.arange(len(t_test))
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    axes[0].plot(t_test, ref_heading, label="integrated -vehicle yaw rate (reference)", linewidth=1.4)
    axes[0].plot(t_test, ekf_heading, label="phone-only EKF prediction", linewidth=1.1)
    axes[0].set_ylabel("Relative heading (deg)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_test, ref_test, label="reference -vehicle yaw rate", linewidth=1.0)
    axes[1].plot(t_test, rate_test, label="train-calibrated phone XYZ rate", linewidth=0.9)
    axes[1].set_ylabel("Yaw rate (deg/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    if "vehicle_speed_kmh" in df.columns:
        speed_test = pd.to_numeric(df["vehicle_speed_kmh"], errors="coerce").to_numpy(float)[contiguous]
        axes[2].plot(t_test, speed_test, label="vehicle speed (evaluation only)")
        axes[2].axhline(5.0, linestyle="--", linewidth=1.0, label="5 km/h reference")
        axes[2].legend(loc="best")
    else:
        axes[2].text(0.02, 0.5, "vehicle_speed_kmh unavailable", transform=axes[2].transAxes)
    axes[2].set_ylabel("Speed (km/h)")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].grid(True, alpha=0.25)

    axes[0].set_xlim(t_test[0], t_test[-1])
    fig.suptitle("Held-out phone-only gyro integration + minimal prediction-only EKF")
    fig.tight_layout()
    fig.savefig(args.out / "heldout_phone_only_ekf.png", dpi=160)
    plt.close(fig)

    out = df.iloc[contiguous].copy()
    out["reference_yaw_rate_deg_s"] = ref_test
    out["train_calibrated_phone_rate_deg_s"] = rate_test
    out["reference_integrated_heading_deg"] = ref_heading
    out["phone_only_ekf_heading_deg"] = ekf_heading
    out["ekf_bias_estimate_deg_s"] = ekf_bias
    out.to_csv(args.out / "heldout_phone_only_ekf_timeseries.csv", index=False)
    print(f"Outputs written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
