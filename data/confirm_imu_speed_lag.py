"""
Deterministic diagnostic for possible IMU-to-vehicle-speed temporal lag.

This script does NOT shift or modify the dataset. It:
1. Recreates synchronized 10 Hz signals using the same nearest merge/resampling idea.
2. Computes correlations between vehicle longitudinal acceleration proxy
   (d(vehicle speed)/dt) and each raw IMU axis, plus correlations between
   mobile GPS speed and vehicle speed.
3. Sweeps integer lags in +/- MAX_LAG_SAMPLES and reports the best lag for each
   signal, including overlap count and peak-vs-zero-lag difference.
4. Saves plots so the chosen lag can be visually checked.

Important: correlation peak alone does not prove physical sensor latency. Road
grade, turning, phone orientation, smoothing, and speed quantization can shift
the peak. Treat this as evidence to inspect, not an automatic correction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HZ = 10
MAX_LAG_SAMPLES = 50
MIN_VALID_PAIRS = 100
S_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"
)
V_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"
)
OUT_DIR = "data/lag_diagnostic"


def find_col(df, *needles):
    # First: exact, case-insensitive match
    exact_hits = [
        c for c in df.columns
        if c.strip().casefold() in {
            n.strip().casefold() for n in needles
        }
    ]

    if len(exact_hits) == 1:
        return exact_hits[0]

    if len(exact_hits) > 1:
        raise ValueError(
            f"Multiple exact matches for {needles}: {exact_hits}"
        )

    # Fallback: substring matching
    hits = [
        c for c in df.columns
        if all(n.casefold() in c.casefold() for n in needles)
    ]

    if len(hits) != 1:
        raise ValueError(
            f"Expected one match for {needles}, "
            f"got {hits}. Columns={list(df.columns)}"
        )

    return hits[0]

def parse_mobile_clock(value):
    try:
        time_part = str(value).strip().split()[-1]
        hh, mm, ss, ms = time_part.split(":")
        return (
            int(hh) * 3600
            + int(mm) * 60
            + int(ss)
            + int(ms) / 1000
        )
    except (ValueError, TypeError, IndexError):
        return np.nan

def corr_at_lag(x, y, lag):
    # Positive lag means x[i + lag] is compared with y[i].
    if lag > 0:
        xa, ya = x[lag:], y[:-lag]
    elif lag < 0:
        xa, ya = x[:lag], y[-lag:]
    else:
        xa, ya = x, y
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < MIN_VALID_PAIRS:
        return np.nan, int(mask.sum())
    if np.std(xa[mask]) == 0 or np.std(ya[mask]) == 0:
        return np.nan, int(mask.sum())
    return float(np.corrcoef(xa[mask], ya[mask])[0, 1]), int(mask.sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = pd.read_csv(S_PATH, encoding="latin-1")
    v = pd.read_csv(V_PATH, encoding="latin-1")
    s.columns = s.columns.str.strip()
    v.columns = v.columns.str.strip()

    s["abs_time"] = s[find_col(s, "DATE")].map(parse_mobile_clock)
    v["abs_time"] = pd.to_numeric(v[find_col(v, "Time Since Start of Day")], errors="coerce")
    s = s.dropna(subset=["abs_time"]).sort_values("abs_time")
    v = v.dropna(subset=["abs_time"]).sort_values("abs_time")

    mobile = {
        "gps_speed_ms": find_col(s, "GPS SPEED"),
        "acc_x": find_col(s, "ACCELEROMETER X"),
        "acc_y": find_col(s, "ACCELEROMETER Y"),
        "acc_z": find_col(s, "ACCELEROMETER Z"),
        "gyro_yaw": find_col(s, "GYROSCOPE Yaw"),
        "gyro_pitch": find_col(s, "GYROSCOPE Pitch"),
        "gyro_roll": find_col(s, "GYROSCOPE Roll"),
    }
    vehicle_speed = find_col(v, "Velocity (km/hr)")
    merged = pd.merge_asof(
        s[["abs_time", *mobile.values()]].sort_values("abs_time"),
        v[["abs_time", vehicle_speed]].rename(columns={vehicle_speed: "vehicle_speed_kmh"}).sort_values("abs_time"),
        on="abs_time", direction="nearest", tolerance=0.2,
    ).dropna(subset=["vehicle_speed_kmh"])

    t0 = np.ceil(merged.abs_time.min() * HZ) / HZ
    t1 = np.floor(merged.abs_time.max() * HZ) / HZ
    print(f"t0={t0}, t1={t1}, HZ={HZ}")
    print(f"finite: t0={np.isfinite(t0)}, t1={np.isfinite(t1)}")

    if not np.isfinite(t0) or not np.isfinite(t1):
        raise ValueError(f"Invalid time bounds: t0={t0}, t1={t1}")

    if t1 <= t0:
        raise ValueError(f"Invalid time range: t0={t0}, t1={t1}")

    if not np.isfinite(HZ) or HZ <= 0:
        raise ValueError(f"Invalid HZ: {HZ}")
    grid = np.round(np.arange(t0, t1 + 0.5/HZ, 1/HZ), 6)
    res = pd.DataFrame({"time_s": grid})
    src_t = merged.abs_time.to_numpy(dtype=float, copy=True)
    for key, col in mobile.items():
        vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float, copy=True)
        ok = np.isfinite(src_t) & np.isfinite(vals)
        res[key] = np.interp(grid, src_t[ok], vals[ok]) if ok.sum() >= 2 else np.nan
    vals = pd.to_numeric(merged.vehicle_speed_kmh, errors="coerce").to_numpy(dtype=float, copy=True)
    ok = np.isfinite(src_t) & np.isfinite(vals)
    res["vehicle_speed_ms"] = np.interp(grid, src_t[ok], vals[ok]) / 3.6 if ok.sum() >= 2 else np.nan

    # Smooth only for derivative stability; retain raw speed in saved diagnostic.
    res["vehicle_accel_ms2"] = np.gradient(res.vehicle_speed_ms.to_numpy(), 1/HZ)
    for col in mobile:
        res[col] = pd.to_numeric(res[col], errors="coerce")

    targets = {
        "mobile GPS speed vs vehicle speed": ("gps_speed_ms", "vehicle_speed_ms"),
        "acc_x vs vehicle longitudinal acceleration proxy": ("acc_x", "vehicle_accel_ms2"),
        "acc_y vs vehicle longitudinal acceleration proxy": ("acc_y", "vehicle_accel_ms2"),
        "acc_z vs vehicle longitudinal acceleration proxy": ("acc_z", "vehicle_accel_ms2"),
        "gyro_yaw vs vehicle acceleration proxy": ("gyro_yaw", "vehicle_accel_ms2"),
        "gyro_pitch vs vehicle acceleration proxy": ("gyro_pitch", "vehicle_accel_ms2"),
        "gyro_roll vs vehicle acceleration proxy": ("gyro_roll", "vehicle_accel_ms2"),
    }
    lags = np.arange(-MAX_LAG_SAMPLES, MAX_LAG_SAMPLES + 1)
    rows = []
    curves = {}

    for label, (xcol, ycol) in targets.items():
        scores, counts = [], []
        for lag in lags:
            score, count = corr_at_lag(
                res[xcol].to_numpy(dtype=float),
                res[ycol].to_numpy(dtype=float),
                int(lag),
            )
            scores.append(score)
            counts.append(count)
        scores = np.asarray(scores)
        curves[label] = scores
        finite = np.isfinite(scores)
        if finite.any():
            best_idx = np.nanargmax(np.abs(scores))
            zero_idx = np.where(lags == 0)[0][0]
            rows.append({
                "comparison": label,
                "x_signal": xcol,
                "y_signal": ycol,
                "best_lag_samples": int(lags[best_idx]),
                "best_lag_seconds": float(lags[best_idx] / HZ),
                "best_signed_correlation": float(scores[best_idx]),
                "zero_lag_correlation": float(scores[zero_idx]),
                "peak_abs_corr_minus_zero_abs_corr": float(abs(scores[best_idx]) - abs(scores[zero_idx])),
                "valid_pairs_at_best": int(counts[best_idx]),
            })
        else:
            rows.append({"comparison": label, "best_lag_samples": np.nan})

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(OUT_DIR, "lag_sweep.csv"), index=False)
    res.to_csv(os.path.join(OUT_DIR, "aligned_signals.csv"), index=False)

    plt.figure(figsize=(11, 6))
    for label, scores in curves.items():
        plt.plot(lags / HZ, scores, marker=".", label=label)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Lag (seconds); positive means use x later than y")
    plt.ylabel("Pearson correlation")
    plt.title("Lag sweep: IMU/mobile signals vs vehicle speed-derived signals")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lag_sweep.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(res.time_s, res.vehicle_speed_ms, label="Vehicle Velocity (m/s)")
    axes[0].plot(res.time_s, res.gps_speed_ms, label="Mobile GPS SPEED (treated as m/s)", alpha=0.8)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[1].plot(res.time_s, res.vehicle_accel_ms2, label="d(vehicle speed)/dt")
    for col in ("acc_x", "acc_y", "acc_z"):
        axes[1].plot(res.time_s, res[col], label=col, alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration / raw IMU units")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "signal_overlay.png"), dpi=160)
    plt.close(fig)

    print(report.to_string(index=False))
    print(f"\nWrote CSVs and plots to {OUT_DIR}/")
    print("\nInterpretation caution: a correlation peak is not deterministic proof of sensor latency.")


if __name__ == "__main__":
    main()
