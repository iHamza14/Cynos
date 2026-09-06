"""
Alignment Diagnostic — no ML, pure statistics.
Run this BEFORE your preprocessing pipeline to verify sync quality.
"""
import pandas as pd
import numpy as np
import sys

# ── 0. Load ────────────────────────────────────────────────────────────────────
S_PATH = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'
V_PATH = '../Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'
    
S = pd.read_csv(S_PATH, encoding='latin1')
V = pd.read_csv(V_PATH, encoding='latin1')

S.columns = S.columns.str.strip()
V.columns = V.columns.str.strip()
# ── 1. Timestamp normalisation ─────────────────────────────────────────────────
# Phone: TIME SINCE START (ms) → seconds
S["t_s"] = S["TIME SINCE START (ms)"] / 1000.0

# Vehicle: Time Since Start of Day (seconds) — already seconds
# But it's "since start of day", phone is "since start of recording".
# They need the SAME zero reference. Detect offset:
V["t_s"] = V["Time Since Start of Day (seconds)"] - V["Time Since Start of Day (seconds)"].min() + S["TIME SINCE START (ms)"].min() / 1000.0

print("=== 1. RAW TIMESTAMP RANGES ===")
print(f"  Phone  : {S['t_s'].min():.1f}s  →  {S['t_s'].max():.1f}s   ({S['t_s'].max()-S['t_s'].min():.1f}s total)")
print(f"  Vehicle: {V['t_s'].min():.1f}s  →  {V['t_s'].max():.1f}s   ({V['t_s'].max()-V['t_s'].min():.1f}s total)")

overlap_start = max(S["t_s"].min(), V["t_s"].min())
overlap_end   = min(S["t_s"].max(), V["t_s"].max())
print(f"  Overlap: {overlap_end - overlap_start:.1f}s   (should be >80% of both durations)")

# ── 2. Sampling rate audit ─────────────────────────────────────────────────────
S_dt = S["t_s"].diff().dropna()
V_dt = V["t_s"].diff().dropna()

print("\n=== 2. SAMPLING RATES ===")
print(f"  Phone   median Δt={S_dt.median()*1000:.1f}ms  std={S_dt.std()*1000:.1f}ms  "
      f"→ ~{1/S_dt.median():.0f} Hz")
print(f"  Vehicle median Δt={V_dt.median()*1000:.1f}ms  std={V_dt.std()*1000:.1f}ms  "
      f"→ ~{1/V_dt.median():.0f} Hz")
print(f"  Gaps >0.5s in phone   : {(S_dt>0.5).sum()}")
print(f"  Gaps >0.5s in vehicle : {(V_dt>0.5).sum()}")

# ── 3. merge_asof alignment ────────────────────────────────────────────────────
S_s = S.sort_values("t_s")
V_s = V.sort_values("t_s")

merged = pd.merge_asof(
    S_s, V_s,
    on="t_s",
    direction="nearest",
    tolerance=0.2,          # ±200ms
    suffixes=("_phone","_veh")
)
merged["vehicle_speed_kmh"] = merged["Velocity (km/hr)"] /3.6
matched   = merged["Velocity (km/hr)"].notna().sum()
unmatched = merged["Velocity (km/hr)"].isna().sum()
print(f"\n=== 3. merge_asof RESULTS (tolerance=200ms) ===")
print(f"  Matched   : {matched}  ({matched/len(merged)*100:.1f}%)")
print(f"  Unmatched : {unmatched}  ({unmatched/len(merged)*100:.1f}%)")
print("  ✓ PASS" if matched/len(merged) > 0.85 else "  ✗ FAIL — check timestamp zero reference")

# ── 4. Speed cross-check ───────────────────────────────────────────────────────
# Phone GPS speed vs Vehicle GPS speed — should correlate tightly
ok = merged.dropna(subset=["GPS SPEED (Kmh)", "vehicle_speed_kmh"])
ok = ok[ok["vehicle_speed_kmh"] > 5]

# Original (no shift)
corr = ok["GPS SPEED (Kmh)"].corr(ok["vehicle_speed_kmh"])
bias = (ok["GPS SPEED (Kmh)"] - ok["vehicle_speed_kmh"]).mean()
rmse = np.sqrt(((ok["GPS SPEED (Kmh)"] - ok["vehicle_speed_kmh"])**2).mean())

# Lag-corrected: shift phone GPS speed forward 5 samples (500ms)
phone_shifted = ok["GPS SPEED (Kmh)"].shift(-10)
ok2 = ok.copy()
ok2["phone_shifted"] = phone_shifted
ok2 = ok2.dropna(subset=["phone_shifted"])

corr_s = ok2["phone_shifted"].corr(ok2["vehicle_speed_kmh"])
bias_s = (ok2["phone_shifted"] - ok2["vehicle_speed_kmh"]).mean()
rmse_s = np.sqrt(((ok2["phone_shifted"] - ok2["vehicle_speed_kmh"])**2).mean())

print(f"\n=== 4. SPEED CROSS-CHECK (phone GPS vs vehicle odometry, moving only) ===")
print(f"  [No shift]  r={corr:.4f}  bias={bias:.3f} km/h  RMSE={rmse:.3f} km/h")
print(f"  [10-sample shift]  r={corr_s:.4f}  bias={bias_s:.3f} km/h  RMSE={rmse_s:.3f} km/h")

if corr_s > corr:
    print(f"  → Shift improves correlation by {corr_s-corr:.4f}; lag is real, use sliding loss matching")
else:
    print(f"  → Shift doesn't help; lag finding was noise, no correction needed")
# ── 5. Heading cross-check ─────────────────────────────────────────────────────
phone_heading_col = next(
    c for c in merged.columns
    if "GPS ORIENTATION" in c.upper()
)

vehicle_heading_col = next(
    c for c in merged.columns
    if "HEADING (DEGREES)" in c.upper()
)

ok2 = merged.dropna(
    subset=[phone_heading_col, vehicle_heading_col]
)

ok2 = ok2[ok2["Velocity (km/hr)"] > 10]

head_diff = (
    (ok2[phone_heading_col] - ok2[vehicle_heading_col] + 180)
    % 360
    - 180
)
print(f"\n=== 5. HEADING CROSS-CHECK (moving, >10 km/h) ===")
print(f"  Mean diff : {head_diff.mean():.2f}°  (should be ~0)")
print(f"  Std diff  : {head_diff.std():.2f}°   (<15° = good)")
print(f"  |diff|>30°: {(head_diff.abs()>30).sum()} samples  ({(head_diff.abs()>30).mean()*100:.1f}%)")

# ── 6. Accelerometer gravity sanity ───────────────────────────────────────────
# |gravity vector| should be ~9.81 m/s²
grav_mag = np.sqrt(S["GRAVITY X (m/s²)"]**2 + S["GRAVITY Y (m/s²)"]**2 + S["GRAVITY Z (m/s²)"]**2)
print(f"\n=== 6. GRAVITY VECTOR MAGNITUDE ===")
print(f"  Mean: {grav_mag.mean():.3f}  Std: {grav_mag.std():.3f}  (expect 9.81 ± 0.1)")
print("  ✓ PASS" if abs(grav_mag.mean()-9.81)<0.3 else "  ✗ FAIL — gravity looks wrong")

# ── 7. IMU–speed temporal lag check ──────────────────────────────────────────
# Cross-correlate |linear accel| with |dv/dt| from vehicle to detect lag
lin_acc_mag = np.sqrt(
    (S["ACCELEROMETER X (m/s²)"] - S["GRAVITY X (m/s²)"])**2 +
    (S["ACCELEROMETER Y (m/s²)"] - S["GRAVITY Y (m/s²)"])**2 +
    (S["ACCELEROMETER Z (m/s²)"] - S["GRAVITY Z (m/s²)"])**2
)
S_1hz = S.set_index("t_s")["lin_acc_mag"] if "lin_acc_mag" in S.columns else None
# Resample both to 1Hz on merged
merged["lin_acc_mag"] = np.sqrt(
    (merged["ACCELEROMETER X (m/s²)"] - merged["GRAVITY X (m/s²)"])**2 +
    (merged["ACCELEROMETER Y (m/s²)"] - merged["GRAVITY Y (m/s²)"])**2 +
    (merged["ACCELEROMETER Z (m/s²)"] - merged["GRAVITY Z (m/s²)"])**2
)
merged["dv"] = merged["Velocity (km/hr)"].diff().abs()

a = merged["lin_acc_mag"].dropna().values[:2000]
b = merged["dv"].dropna().values[:2000]
min_len = min(len(a), len(b))
xcorr = np.correlate(a[:min_len]-a[:min_len].mean(), b[:min_len]-b[:min_len].mean(), mode='full')
lags  = np.arange(-min_len+1, min_len)
best_lag = lags[np.argmax(xcorr)]
phone_rate = 1/S_dt.median()
print(f"\n=== 7. IMU-SPEED TEMPORAL LAG ===")
print(f"  Best lag: {best_lag} samples @ ~{phone_rate:.0f}Hz = {best_lag/phone_rate*1000:.0f}ms")
print("  ✓ PASS" if abs(best_lag) < phone_rate*0.3 else
      f"  ⚠ WARN — {best_lag/phone_rate*1000:.0f}ms lag; apply shift before windowing")

print("\n=== DONE ===")
