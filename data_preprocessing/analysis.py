
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf
from scipy.signal import correlate


# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = "./plots2"
os.makedirs(OUT_DIR, exist_ok=True)

MOBILE_PATH = (
    "Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    "Vw (Driver E)/Vw04/S-Vw4.csv"
)
VEHICLE_PATH = (
    "Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    "Vw (Driver E)/Vw04/V-Vw4.csv"
)

IMU_HZ = 10
GPS_EXPECTED_HZ = 0.5
GPS_EXPECTED_DT = 1 / GPS_EXPECTED_HZ

# Inspect short segments to make timing offsets visible.
PLOT_START_SEC = 100
PLOT_DURATION_SEC = 30

# Candidate lags for IMU-vs-vehicle-speed-change diagnostic.
MAX_LAG_SEC = 3.0


# ── Load ──────────────────────────────────────────────────────────────────────
def read_csv(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df.columns = df.columns.str.strip()
    return df


print("Loading datasets...")
df_m = read_csv(MOBILE_PATH)
df_v = read_csv(VEHICLE_PATH)


def get_col(df, name):
    """Return first column containing name, case-insensitive."""
    for col in df.columns:
        if name.lower() in col.lower():
            return col
    return None


def numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")


# ── Timestamps ────────────────────────────────────────────────────────────────
time_m_col = get_col(df_m, "TIME SINCE START")
time_v_col = get_col(df_v, "Time Since Start of Day")

if time_m_col:
    df_m["time_s"] = numeric_series(df_m, time_m_col) / 1000.0
else:
    df_m["time_s"] = np.arange(len(df_m)) / IMU_HZ

if time_v_col:
    df_v["time_s"] = numeric_series(df_v, time_v_col)
    df_v["time_s"] -= df_v["time_s"].iloc[0]
else:
    df_v["time_s"] = np.arange(len(df_v)) / IMU_HZ

df_m = (
    df_m.dropna(subset=["time_s"])
    .sort_values("time_s")
    .reset_index(drop=True)
)
df_v = (
    df_v.dropna(subset=["time_s"])
    .sort_values("time_s")
    .reset_index(drop=True)
)

# ── Columns ───────────────────────────────────────────────────────────────────
gps_speed_col = get_col(df_m, "GPS SPEED")
v_speed_col = get_col(df_v, "Indicated Vehicle Speed")

gyro_yaw = get_col(df_m, "GYROSCOPE Yaw")
gyro_pitch = get_col(df_m, "GYROSCOPE Pitch")
gyro_roll = get_col(df_m, "GYROSCOPE Roll")
gyro_cols = [c for c in [gyro_yaw, gyro_pitch, gyro_roll] if c]

accel_x = get_col(df_m, "ACCELEROMETER X")
accel_y = get_col(df_m, "ACCELEROMETER Y")
accel_z = get_col(df_m, "ACCELEROMETER Z")
accel_cols = [c for c in [accel_x, accel_y, accel_z] if c]

orient_yaw = get_col(df_m, "ORIENTATION (Yaw)")
vehicle_heading = get_col(df_v, "Heading")


# ── Helpers ───────────────────────────────────────────────────────────────────
def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}", dpi=150)
    plt.close()


def segment_mask(time, start, duration):
    return (time >= start) & (time <= start + duration)


def wrap_angle_rad(angle):
    """Wrap angular differences to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def interp_vehicle_to_mobile(col):
    """Interpolate one vehicle signal onto mobile timestamps."""
    return np.interp(
        df_m["time_s"].to_numpy(),
        df_v["time_s"].to_numpy(),
        numeric_series(df_v, col).to_numpy(),
    )


# ── Phase 1: Sampling cadence ─────────────────────────────────────────────────
print("Phase 1: Sampling cadence")

dt_m = np.diff(df_m["time_s"])
dt_v = np.diff(df_v["time_s"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_m["time_s"].iloc[1:], dt_m, label="Mobile dt", alpha=0.8)
ax.plot(df_v["time_s"].iloc[1:], dt_v, label="Vehicle dt", alpha=0.8)
ax.axhline(0.1, linestyle="--", label="100 ms (10 Hz)")
ax.axhline(2.0, linestyle=":", label="2 s (0.5 Hz)")
ax.set_title("Sampling intervals on timestamp axis")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Δt (s)")
ax.legend()
savefig("1_sampling_intervals.png")

print(
    f"Mobile median dt: {np.nanmedian(dt_m):.4f}s "
    f"({1 / np.nanmedian(dt_m):.2f} Hz)"
)
print(
    f"Vehicle median dt: {np.nanmedian(dt_v):.4f}s "
    f"({1 / np.nanmedian(dt_v):.2f} Hz)"
)


# ── Phase 2: GPS update cadence ───────────────────────────────────────────────
print("Phase 2: GPS update cadence")

if gps_speed_col:
    gps = numeric_series(df_m, gps_speed_col)
    gps_valid = gps.notna() & (gps >= 0)

    gps_df = df_m.loc[gps_valid, ["time_s"]].copy()
    gps_df["gps_speed"] = gps[gps_valid].to_numpy()

    # A value-change detector is only a proxy for update times:
    # a new GPS fix can report the same speed as the previous fix.
    gps_df["speed_changed"] = (
        gps_df["gps_speed"].diff().abs() > 1e-6
    )

    change_times = gps_df.loc[
        gps_df["speed_changed"], "time_s"
    ].to_numpy()

    change_dt = np.diff(change_times)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(
        gps_df["time_s"],
        gps_df["gps_speed"] * 3.6,
        where="post",
        label="Mobile GPS speed (held values)",
    )
    ax.set_title("Mobile GPS speed: observed values, not interpolated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    savefig("2_gps_step_signal.png")

    if len(change_dt):
        plt.figure(figsize=(10, 4))
        plt.hist(change_dt, bins=40)
        plt.axvline(
            GPS_EXPECTED_DT,
            linestyle="--",
            label="Expected 2 s interval",
        )
        plt.title("Intervals between GPS speed value changes")
        plt.xlabel("Time between value changes (s)")
        plt.ylabel("Count")
        plt.legend()
        savefig("2_gps_change_intervals.png")

        print(
            "Median interval between GPS speed changes: "
            f"{np.median(change_dt):.3f}s"
        )
        print(
            "Note: this is not necessarily the GPS fix rate; "
            "unchanged values are indistinguishable from no update."
        )


# ── Phase 3: GPS vs vehicle speed ─────────────────────────────────────────────
print("Phase 3: GPS vs vehicle speed")

if gps_speed_col and v_speed_col:
    vehicle_interp = interp_vehicle_to_mobile(v_speed_col)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Raw GPS points are shown as points; no fabricated intermediate fixes.
    ax.scatter(
        df_m["time_s"],
        numeric_series(df_m, gps_speed_col) * 3.6,
        s=8,
        alpha=0.65,
        label="Mobile GPS observations",
    )

    ax.plot(
        df_m["time_s"],
        vehicle_interp,
        label="Vehicle speed interpolated to phone timeline",
        alpha=0.8,
    )

    ax.set_title("GPS observations vs vehicle speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    savefig("3_gps_vehicle_speed.png")

    mask = segment_mask(
        df_m["time_s"],
        PLOT_START_SEC,
        PLOT_DURATION_SEC,
    )

    plt.figure(figsize=(12, 4))
    plt.scatter(
        df_m.loc[mask, "time_s"],
        numeric_series(df_m.loc[mask], gps_speed_col) * 3.6,
        s=25,
        label="GPS observations",
    )
    plt.plot(
        df_m.loc[mask, "time_s"],
        vehicle_interp[mask],
        label="Vehicle speed",
    )
    plt.title("Short-window GPS vs vehicle speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    savefig("3_gps_vehicle_short_window.png")


# ── Phase 4: IMU and vehicle-speed changes ────────────────────────────────────
print("Phase 4: IMU-to-vehicle timing diagnostic")

if accel_cols and v_speed_col:
    # Interpolate vehicle speed onto the phone's 10 Hz timeline.
    vehicle_speed = interp_vehicle_to_mobile(v_speed_col)

    # Convert speed to m/s if the vehicle column is in km/h.
    # The original builder treated v_odo_speed as m/s, so verify
    # the actual units before using this diagnostic.
    vehicle_speed_ms = vehicle_speed

    # Approximate longitudinal acceleration from vehicle-speed derivative.
    t = df_m["time_s"].to_numpy()
    vehicle_accel = np.gradient(vehicle_speed_ms, t)

    # IMU acceleration magnitude is not longitudinal acceleration.
    # Plot axes separately instead of pretending magnitude is equivalent.
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in accel_cols:
        ax.plot(
            df_m["time_s"],
            numeric_series(df_m, col),
            label=col,
            alpha=0.7,
        )
    ax.set_title("Raw IMU acceleration axes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (column units)")
    ax.legend()
    savefig("4_imu_acceleration_axes.png")

    # Short aligned view: acceleration axes and vehicle speed derivative.
    mask = segment_mask(
        df_m["time_s"],
        PLOT_START_SEC,
        PLOT_DURATION_SEC,
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    for col in accel_cols:
        ax1.plot(
            df_m.loc[mask, "time_s"],
            numeric_series(df_m.loc[mask], col),
            label=col,
            alpha=0.65,
        )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("IMU acceleration (column units)")

    ax2 = ax1.twinx()
    ax2.plot(
        df_m.loc[mask, "time_s"],
        vehicle_accel[mask],
        color="black",
        label="Vehicle speed derivative",
        linewidth=1.5,
    )
    ax2.set_ylabel("Vehicle acceleration (speed units/s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("IMU acceleration vs vehicle-speed changes")
    savefig("4_imu_vs_vehicle_acceleration.png")


# ── Phase 5: Gyro stationary noise ────────────────────────────────────────────
print("Phase 5: Gyro stationary noise")

if gps_speed_col and gyro_cols:
    gps_speed = numeric_series(df_m, gps_speed_col)

    # GPS is sparse, so this is only a rough stationary proxy.
    stationary_mask = gps_speed < 0.5
    df_stat = df_m.loc[stationary_mask].copy()

    if len(df_stat) > 100:
        # Plot individual stationary segments to avoid joining gaps.
        stationary_indices = np.flatnonzero(stationary_mask.to_numpy())
        segments = np.split(
            stationary_indices,
            np.where(np.diff(stationary_indices) > 1)[0] + 1,
        )

        plt.figure(figsize=(12, 4))
        for col in gyro_cols:
            for segment in segments:
                if len(segment) < 2:
                    continue
                plt.plot(
                    df_m["time_s"].iloc[segment],
                    numeric_series(df_m.iloc[segment], col),
                    alpha=0.5,
                )

        plt.title("Gyroscope signals during stationary-proxy segments")
        plt.xlabel("Time (s)")
        plt.ylabel("Gyroscope value (column units)")
        savefig("5_gyro_stationary_segments.png")

        plt.figure(figsize=(10, 4))
        for col in gyro_cols:
            y = numeric_series(df_stat, col).dropna().to_numpy()
            if len(y) > 100:
                a = acf(y, nlags=50, fft=True)
                plt.plot(
                    np.arange(len(a)) / IMU_HZ,
                    a,
                    label=col,
                )

        plt.axhline(0, color="black", linewidth=0.7)
        plt.title("Stationary gyro autocorrelation")
        plt.xlabel("Lag (seconds)")
        plt.ylabel("ACF")
        plt.legend()
        savefig("5_gyro_acf.png")

        df_stat[gyro_cols].hist(
            bins=50,
            figsize=(10, 6),
        )
        plt.suptitle("Stationary gyro value distributions")
        savefig("5_gyro_histograms.png")

        try:
            sns.pairplot(df_stat[gyro_cols].dropna(), kind="hist")
            plt.savefig(
                f"{OUT_DIR}/5_gyro_pairplot.png",
                dpi=150,
            )
            plt.close()
        except Exception as e:
            print("Could not plot gyro pairplot:", e)


# ── Phase 6: Phone yaw vs vehicle heading ─────────────────────────────────────
print("Phase 6: Heading comparison")

if orient_yaw and vehicle_heading:
    phone_yaw = numeric_series(df_m, orient_yaw).to_numpy()
    vehicle_yaw = interp_vehicle_to_mobile(vehicle_heading)

    # Assumes both values are in degrees.
    # If either is radians, convert before calculating.
    heading_diff_deg = (
        (phone_yaw - vehicle_yaw + 180) % 360
    ) - 180

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_m["time_s"], phone_yaw, label="Phone yaw")
    ax.plot(df_m["time_s"], vehicle_yaw, label="Vehicle heading")
    ax.set_title("Phone yaw and vehicle heading")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees, verify conventions)")
    ax.legend()
    savefig("6_phone_vs_vehicle_heading.png")

    plt.figure(figsize=(12, 4))
    plt.plot(df_m["time_s"], heading_diff_deg)
    plt.axhline(0, color="black", linewidth=0.7)
    plt.axhline(30, linestyle="--")
    plt.axhline(-30, linestyle="--")
    plt.title("Wrapped phone-yaw minus vehicle-heading difference")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference (degrees)")
    savefig("6_heading_difference.png")


# ── Phase 7: Acceleration scatter ────────────────────────────────────────────
print("Phase 7: Acceleration scatter")

if accel_x and accel_y:
    plt.figure(figsize=(6, 6))
    plt.scatter(
        numeric_series(df_m, accel_x),
        numeric_series(df_m, accel_y),
        s=5,
        alpha=0.15,
    )
    plt.title("Raw accelerometer X vs Y")
    plt.xlabel(accel_x)
    plt.ylabel(accel_y)
    plt.grid(True)
    savefig("7_accel_xy_scatter.png")


print("\nPreprocessing and plots complete.")
print(f"Plots saved in: {OUT_DIR}")


# ── Phase 7: Preprocessed Gyro + Magnetometer Fusion ──────────────────────────
from scipy.signal import medfilt, butter, filtfilt

print("Phase 7: Preprocessed Gyro + Magnetometer Fusion")

mag_x = get_col(df_m, "MAGNETIC FIELD X")
mag_y = get_col(df_m, "MAGNETIC FIELD Y")
gyro_yaw = get_col(df_m, "GYROSCOPE Yaw")
vehicle_heading = get_col(df_v, "Heading")

if mag_x and mag_y and gyro_yaw and vehicle_heading:

    t = df_m["time_s"].to_numpy()
    dt = np.diff(t, prepend=t[0])

    mx = pd.to_numeric(df_m[mag_x], errors="coerce").to_numpy()
    my = pd.to_numeric(df_m[mag_y], errors="coerce").to_numpy()
    gyro = pd.to_numeric(df_m[gyro_yaw], errors="coerce").to_numpy()

    # Assumes vehicle heading is in degrees.
    vehicle_deg = interp_vehicle_to_mobile(vehicle_heading) % 360

    # ── 1. Raw magnetometer azimuth ──
    mag_deg = np.degrees(np.arctan2(my, mx)) % 360
    mag_rad = np.radians(mag_deg)

    # ── 2. Circular median-like filtering via sin/cos ──
    # Median-filter X and Y first to suppress isolated spikes.
    # Kernel 5 samples = 0.5 seconds at 10 Hz.
    mx_med = medfilt(mx, kernel_size=5)
    my_med = medfilt(my, kernel_size=5)

    mag_med = np.arctan2(my_med, mx_med)

    # Smooth on the unit circle, not directly in angle space.
    sin_mag = np.sin(mag_med)
    cos_mag = np.cos(mag_med)

    # 1-second moving average, centered approximately.
    window = 10
    kernel = np.ones(window) / window

    sin_smooth = np.convolve(sin_mag, kernel, mode="same")
    cos_smooth = np.convolve(cos_mag, kernel, mode="same")

    mag_smooth = np.arctan2(sin_smooth, cos_smooth)

    # ── 3. Estimate calibration offset on first 60 seconds ──
    calibration_mask = (
        (t < t[0] + 60)
        & np.isfinite(mag_smooth)
        & np.isfinite(vehicle_deg)
    )

    diff = np.radians(vehicle_deg[calibration_mask]) - mag_smooth[calibration_mask]

    offset = np.arctan2(
        np.mean(np.sin(diff)),
        np.mean(np.cos(diff)),
    )

    mag_calibrated = mag_smooth + offset

    # ── 4. Smooth gyro angular velocity ──
    # Assumes gyro values are rad/s and sampling is approximately 10 Hz.
    gyro_clean = pd.Series(gyro).interpolate(limit_direction="both").to_numpy()

    # Low-pass filter cutoff 1 Hz, sampling rate 10 Hz.
    b, a = butter(2, 1.0 / (IMU_HZ / 2), btype="low")
    gyro_smooth = filtfilt(b, a, gyro_clean)

    # ── 5. Complementary filter ──
    alpha = 0.98
    fused = np.full(len(t), np.nan)

    valid_start = np.flatnonzero(
        np.isfinite(mag_calibrated)
        & np.isfinite(gyro_smooth)
    )

    if len(valid_start):
        start = valid_start[0]
        fused[start] = mag_calibrated[start]

        for i in range(start + 1, len(t)):
            step_dt = t[i] - t[i - 1]

            if (
                not np.isfinite(step_dt)
                or step_dt <= 0
                or step_dt > 0.5
            ):
                fused[i] = mag_calibrated[i]
                continue

            predicted = fused[i - 1] + gyro_smooth[i] * step_dt

            mag_error = (
                mag_calibrated[i] - predicted + np.pi
            ) % (2 * np.pi) - np.pi

            fused[i] = predicted + (1 - alpha) * mag_error

    # ── 6. Convert to degrees and unwrap for plotting ──
    mag_calibrated_deg = np.degrees(mag_calibrated)
    fused_deg = np.degrees(fused)

    vehicle_unwrapped = np.degrees(
        np.unwrap(np.radians(vehicle_deg))
    )
    mag_unwrapped = np.degrees(np.unwrap(mag_calibrated))
    fused_unwrapped = np.degrees(np.unwrap(fused))

    # ── Plot full recording ──
    plt.figure(figsize=(14, 5))
    plt.plot(t, vehicle_unwrapped, label="Vehicle heading", alpha=0.8)
    plt.plot(t, mag_unwrapped, label="Smoothed magnetometer", alpha=0.7)
    plt.plot(t, fused_unwrapped, label="Gyro + magnetometer fused", alpha=0.9)

    plt.title("Preprocessed Heading Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped heading (degrees)")
    plt.legend()
    savefig("7_preprocessed_fusion_vs_vehicle.png")

    # ── Plot short window ──
    short = (
        (t >= PLOT_START_SEC)
        & (t <= PLOT_START_SEC + PLOT_DURATION_SEC)
    )

    plt.figure(figsize=(14, 5))
    plt.plot(t[short], vehicle_unwrapped[short], label="Vehicle heading")
    plt.plot(t[short], mag_unwrapped[short], label="Smoothed magnetometer")
    plt.plot(t[short], fused_unwrapped[short], label="Fused heading")

    plt.title("Short-Window Preprocessed Heading")
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped heading (degrees)")
    plt.legend()
    savefig("7_preprocessed_fusion_short.png")

    # ── Wrapped error ──
    error = (
        np.degrees(fused - np.radians(vehicle_deg)) + 180
    ) % 360 - 180

    valid = np.isfinite(error)

    plt.figure(figsize=(14, 4))
    plt.plot(t[valid], error[valid])
    plt.axhline(0, color="black", linewidth=0.7)
    plt.axhline(15, linestyle="--")
    plt.axhline(-15, linestyle="--")
    plt.title("Fused Heading Error (Wrapped)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (degrees)")
    savefig("7_preprocessed_fusion_error.png")

    print(f"Calibration offset: {np.degrees(offset):.2f}°")
    print(f"Fused heading MAE: {np.mean(np.abs(error[valid])):.2f}°")
    print(f"Fused heading std: {np.std(error[valid]):.2f}°")

else:
    print("Missing required columns.")
    print("Mag X:", mag_x)
    print("Mag Y:", mag_y)
    print("Gyro:", gyro_yaw)
    print("Vehicle heading:", vehicle_heading)