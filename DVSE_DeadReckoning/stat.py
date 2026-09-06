import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional map plotting
try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# ============================================================================
# CONFIG
# ============================================================================

PKL_PATH = "data/blackout_windows.pkl"
OUT_DIR = "blackout_eval"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# LOAD
# ============================================================================

print("=" * 80)
print("BLACKOUT WINDOW DATASET EVALUATION")
print("=" * 80)

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(
        f"\nCould not find:\n  {PKL_PATH}\n\n"
        "Run your blackout-window generation script first."
    )

with open(PKL_PATH, "rb") as f:
    windows = pickle.load(f)

print(f"\nLoaded: {PKL_PATH}")
print(f"Total windows: {len(windows)}")


if len(windows) == 0:
    raise RuntimeError("Dataset contains ZERO blackout windows.")


# ============================================================================
# 1. WINDOW STRUCTURE
# ============================================================================

print("\n" + "=" * 80)
print("1. WINDOW STRUCTURE")
print("=" * 80)

first = windows[0]

print("\nTop-level keys:")
print(list(first.keys()))

print("\nContext keys:")
print(list(first["context"].keys()))

print("\nBlackout keys:")
print(list(first["blackout"].keys()))

print("\nGround-truth keys:")
print(list(first["ground_truth"].keys()))


# ============================================================================
# 2. DURATION DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("2. BLACKOUT DURATION DISTRIBUTION")
print("=" * 80)

durations = [
    int(w["blackout_dur_sec"])
    for w in windows
]

duration_counts = pd.Series(durations).value_counts().sort_index()

for duration, count in duration_counts.items():
    print(
        f"  {duration:3d}s : {count:6d} windows"
    )


# ============================================================================
# 3. ARRAY SHAPES
# ============================================================================

print("\n" + "=" * 80)
print("3. ARRAY SHAPES")
print("=" * 80)

for i, w in enumerate(windows[:5]):

    print(f"\nWindow {i}")

    for section in ["context", "blackout", "ground_truth"]:

        print(f"\n  {section}:")

        for key, value in w[section].items():

            arr = np.asarray(value)

            print(
                f"    {key:25s} "
                f"shape={str(arr.shape):15s} "
                f"dtype={arr.dtype}"
            )


# ============================================================================
# 4. EXPECTED 60-SECOND SHAPES
# ============================================================================

print("\n" + "=" * 80)
print("4. SHAPE VALIDATION")
print("=" * 80)

shape_errors = []

for i, w in enumerate(windows):

    d = int(w["blackout_dur_sec"])

    b = w["blackout"]

    expected = {
        "raw_accel": (d, 3),
        "gravity": (d, 3),
        "mtn_input": (d, 6),
        "acc_feat": (d, 18),
        "gyro_feat": (d, 18),
        "raw_gyro": (d, 3),
    }

    for key, expected_shape in expected.items():
        if key not in b:
            shape_errors.append((i, key, "MISSING", expected_shape))
            continue
        actual = np.asarray(b[key]).shape
        if actual != expected_shape:
            shape_errors.append((i, key, actual, expected_shape))

    # Also check ground truth keys
    gt_expected = {
        "speeds_ms": (d,),
        "headings_rad": (d,),
        "delta_v": (d,),
    }
    gt_block = w["ground_truth"]
    for key, expected_shape in gt_expected.items():
        if key not in gt_block:
            shape_errors.append((i, f"gt_{key}", "MISSING", expected_shape))
            continue
        actual = np.asarray(gt_block[key]).shape
        if actual != expected_shape:
            shape_errors.append((i, f"gt_{key}", actual, expected_shape))

if shape_errors:
    print(f"  ✗ {len(shape_errors)} shape errors")
    for err in shape_errors[:20]:
        print("   ", err)
else:
    print("  ✓ All blackout arrays have expected shapes")

print("\n--- Diagnostic Stats ---")
rows = []
for i, w in enumerate(windows):
    gt = w["ground_truth"]
    dist = float(gt["total_distance_m"])
    disp = np.asarray(gt["cum_disp_m"], dtype=float)
    speeds = np.asarray(gt["speeds_ms"], dtype=float)
    delta_v = np.asarray(gt["delta_v"], dtype=float)

    final_disp = float(np.linalg.norm(disp[-1]))

    headings = np.asarray(gt["headings_rad"], dtype=float)
    
    # Wrap heading change to [-pi, pi)
    heading_change_raw = headings[-1] - headings[0]
    heading_change = (heading_change_raw + np.pi) % (2 * np.pi) - np.pi
# ============================================================================
# 5. NUMERICAL HEALTH
# ============================================================================

print("\n" + "=" * 80)
print("5. NaN / INF CHECK")
print("=" * 80)

numeric_bad = []

for i, w in enumerate(windows):

    for section in ["context", "blackout", "ground_truth"]:

        for key, value in w[section].items():

            arr = np.asarray(value)

            if not np.issubdtype(arr.dtype, np.number):
                continue

            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()

            if nan_count or inf_count:

                numeric_bad.append(
                    (
                        i,
                        section,
                        key,
                        int(nan_count),
                        int(inf_count)
                    )
                )


if numeric_bad:

    print(
        f"  ✗ Found {len(numeric_bad)} arrays containing NaN/Inf"
    )

    for x in numeric_bad[:30]:
        print(
            f"    window={x[0]} "
            f"{x[1]}/{x[2]} "
            f"NaN={x[3]} INF={x[4]}"
        )

else:

    print("  ✓ No NaN/Inf values found")


# ============================================================================
# 6. WINDOW-LEVEL STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("6. WINDOW STATISTICS")
print("=" * 80)

rows = []

for i, w in enumerate(windows):

    gt = w["ground_truth"]
    b = w["blackout"]

    speeds = np.asarray(
        gt["speeds_ms"],
        dtype=float
    )

    delta_v = np.asarray(
        gt["delta_v"],
        dtype=float
    )

    dist = float(
        gt["total_distance_m"]
    )

    disp = np.asarray(
        gt["cum_disp_m"],
        dtype=float
    )

    final_disp = float(
        np.linalg.norm(disp[-1])
    )

    headings = np.asarray(
        gt["headings_rad"],
        dtype=float
    )

    heading_change = np.unwrap(headings)[-1] - np.unwrap(headings)[0]

    rows.append({
        "window": i,
        "duration_sec": w["blackout_dur_sec"],
        "start_sec": w["window_start_sec"],
        "total_distance_m": dist,
        "final_displacement_m": final_disp,
        "mean_speed_ms": speeds.mean(),
        "max_speed_ms": speeds.max(),
        "speed_std_ms": speeds.std(),
        "delta_v_mean": delta_v.mean(),
        "delta_v_std": delta_v.std(),
        "delta_v_abs_max": np.abs(delta_v).max(),
        "heading_change_deg": np.degrees(heading_change),
    })


stats_df = pd.DataFrame(rows)

stats_df.to_csv(
    os.path.join(
        OUT_DIR,
        "blackout_window_summary.csv"
    ),
    index=False
)

print("\nDistance:")
print(
    stats_df[
        "total_distance_m"
    ].describe()
)

print("\nMean speed:")
print(
    stats_df[
        "mean_speed_ms"
    ].describe()
)

print("\nHeading change:")
print(
    stats_df[
        "heading_change_deg"
    ].describe()
)


# ============================================================================
# 7. 60-SECOND WINDOWS SPECIFICALLY
# ============================================================================

windows_60 = [
    (i, w)
    for i, w in enumerate(windows)
    if int(w["blackout_dur_sec"]) == 60
]

print("\n" + "=" * 80)
print("7. 60-SECOND WINDOWS")
print("=" * 80)

print(
    f"\n60-second windows: {len(windows_60)}"
)

if windows_60:

    d60 = stats_df[
        stats_df["duration_sec"] == 60
    ]

    print("\n60-second distance statistics:")
    print(
        d60[
            "total_distance_m"
        ].describe()
    )

    print("\n60-second speed statistics:")
    print(
        d60[
            "mean_speed_ms"
        ].describe()
    )


# ============================================================================
# 8. PHYSICAL CONSISTENCY CHECK
# ============================================================================

print("\n" + "=" * 80)
print("8. PHYSICAL CONSISTENCY")
print("=" * 80)


# Distance should approximately agree with
# integral of ground-truth speed.

distance_errors = []

for i, w in enumerate(windows):

    gt = w["ground_truth"]

    speed = np.asarray(
        gt["speeds_ms"],
        dtype=float
    )

    duration = int(
        w["blackout_dur_sec"]
    )

    # Data is binned at ~1 Hz.
    integrated_distance = np.sum(
        speed
    )

    reported_distance = float(
        gt["total_distance_m"]
    )

    if reported_distance > 1:

        rel_error = (
            abs(integrated_distance - reported_distance)
            / reported_distance
        )

        distance_errors.append(
            rel_error
        )


distance_errors = np.asarray(
    distance_errors
)

print(
    f"  Median distance consistency error: "
    f"{np.median(distance_errors) * 100:.2f}%"
)

print(
    f"  95th percentile: "
    f"{np.percentile(distance_errors, 95) * 100:.2f}%"
)

print(
    f"  Maximum: "
    f"{distance_errors.max() * 100:.2f}%"
)


# ============================================================================
# 9. DELTA-V CONSISTENCY
# ============================================================================

print("\n" + "=" * 80)
print("9. DELTA-V CONSISTENCY")
print("=" * 80)

dv_errors = []

for i, w in enumerate(windows):

    gt = w["ground_truth"]

    speed = np.asarray(
        gt["speeds_ms"],
        dtype=float
    )

    delta_v = np.asarray(
        gt["delta_v"],
        dtype=float
    )

    # Compare supplied delta-v with speed differences.
    if len(speed) > 1:

        true_dv = np.diff(speed)

        # Accommodate either dV[t] = v[t]-v[t-1]
        # or a first element representing initial delta.
        if len(delta_v) == len(speed):

            candidate = delta_v[1:]

            n = min(
                len(candidate),
                len(true_dv)
            )

            err = np.abs(
                candidate[:n]
                - true_dv[:n]
            )

            dv_errors.extend(
                err.tolist()
            )


if dv_errors:

    dv_errors = np.asarray(
        dv_errors
    )

    print(
        f"  Median |Δv error|: "
        f"{np.median(dv_errors):.5f} m/s"
    )

    print(
        f"  Mean |Δv error|: "
        f"{dv_errors.mean():.5f} m/s"
    )

    print(
        f"  95th percentile: "
        f"{np.percentile(dv_errors, 95):.5f} m/s"
    )

else:

    print(
        "  Could not perform Δv consistency check."
    )


# ============================================================================
# 10. START/END GPS SANITY
# ============================================================================

print("\n" + "=" * 80)
print("10. GPS START/END SANITY")
print("=" * 80)

gps_rows = []

for i, w in enumerate(windows):

    context = w["context"]
    gt = w["ground_truth"]

    start_lat = context.get(
        "last_lat",
        np.nan
    )

    start_lon = context.get(
        "last_lon",
        np.nan
    )

    end_lat = gt.get(
        "end_lat",
        np.nan
    )

    end_lon = gt.get(
        "end_lon",
        np.nan
    )

    gps_rows.append({
        "window": i,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
    })


gps_df = pd.DataFrame(
    gps_rows
)

print(
    "\nLatitude range:"
)

print(
    gps_df[
        ["start_lat", "end_lat"]
    ].describe()
)

print(
    "\nLongitude range:"
)

print(
    gps_df[
        ["start_lon", "end_lon"]
    ].describe()
)


# ============================================================================
# 11. OVERLAP / DUPLICATE CHECK
# ============================================================================

print("\n" + "=" * 80)
print("11. WINDOW OVERLAP CHECK")
print("=" * 80)

intervals = []

for i, w in enumerate(windows):

    start = float(
        w["window_start_sec"]
    )

    duration = float(
        w["blackout_dur_sec"]
    )

    intervals.append(
        (
            start,
            start + duration,
            i
        )
    )

intervals.sort()

overlap_count = 0

for j in range(1, len(intervals)):

    prev_end = intervals[j - 1][1]
    curr_start = intervals[j][0]

    if curr_start < prev_end:

        overlap_count += 1


print(
    f"  Overlapping consecutive windows: "
    f"{overlap_count}"
)


# ============================================================================
# 12. SAMPLE WINDOW REPORT
# ============================================================================

print("\n" + "=" * 80)
print("12. SAMPLE WINDOWS")
print("=" * 80)

sample_indices = np.linspace(
    0,
    len(windows) - 1,
    min(10, len(windows)),
    dtype=int
)

for i in sample_indices:

    w = windows[i]

    gt = w["ground_truth"]

    print(
        f"\nWindow {i}"
    )

    print(
        f"  duration       : "
        f"{w['blackout_dur_sec']} s"
    )

    print(
        f"  start time     : "
        f"{w['window_start_sec']:.2f} s"
    )

    print(
        f"  total distance : "
        f"{gt['total_distance_m']:.2f} m"
    )

    print(
        f"  endpoint       : "
        f"({gt['end_lat']:.6f}, "
        f"{gt['end_lon']:.6f})"
    )

    print(
        f"  start speed    : "
        f"{gt['speeds_ms'][0]:.2f} m/s"
    )

    print(
        f"  end speed      : "
        f"{gt['speeds_ms'][-1]:.2f} m/s"
    )


# ============================================================================
# 13. PLOT 60-SECOND TRAJECTORIES
# ============================================================================

print("\n" + "=" * 80)
print("13. TRAJECTORY PLOTS")
print("=" * 80)


def displacement_to_latlon(
    start_lat,
    start_lon,
    north_m,
    east_m
):

    R = 6371000.0

    lat = (
        start_lat
        + np.degrees(
            north_m / R
        )
    )

    lon = (
        start_lon
        + np.degrees(
            east_m
            /
            (
                R
                * np.cos(
                    np.radians(
                        start_lat
                    )
                )
            )
        )
    )

    return lat, lon


# -------------------------------------------------------------------------
# A. Plain trajectory plot
# -------------------------------------------------------------------------

if windows_60:

    # Pick spatially diverse examples.
    selected = [
        windows_60[
            int(i * len(windows_60) / min(6, len(windows_60)))
        ][0]
        for i in range(
            min(6, len(windows_60))
        )
    ]

    selected = list(
        dict.fromkeys(selected)
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(111)

    for idx in selected:

        w = windows[idx]

        gt = w["ground_truth"]

        disp = np.asarray(
            gt["cum_disp_m"]
        )

        ax.plot(
            disp[:, 1],
            disp[:, 0],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=f"Window {idx}"
        )

        ax.scatter(
            disp[0, 1],
            disp[0, 0],
            marker="o",
            s=50
        )

        ax.scatter(
            disp[-1, 1],
            disp[-1, 0],
            marker="x",
            s=60
        )

    ax.set_xlabel(
        "East displacement (m)"
    )

    ax.set_ylabel(
        "North displacement (m)"
    )

    ax.set_title(
        "Sample 60-second blackout trajectories"
    )

    ax.axis("equal")

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        OUT_DIR,
        "sample_60s_trajectories.png"
    )

    fig.savefig(
        path,
        dpi=180
    )

    plt.close(fig)

    print(
        f"  Saved: {path}"
    )


# -------------------------------------------------------------------------
# B. Geographic map using Folium
# -------------------------------------------------------------------------

if HAS_FOLIUM and windows_60:

    selected = [
        windows_60[
            int(i * len(windows_60) / min(5, len(windows_60)))
        ][0]
        for i in range(
            min(5, len(windows_60))
        )
    ]

    selected = list(
        dict.fromkeys(selected)
    )

    # Map center from selected starts
    centers = []

    for idx in selected:

        w = windows[idx]

        centers.append(
            [
                float(w["context"]["last_lat"]),
                float(w["context"]["last_lon"])
            ]
        )

    center_lat = np.mean(
        [x[0] for x in centers]
    )

    center_lon = np.mean(
        [x[1] for x in centers]
    )

    fmap = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=13,
        tiles="OpenStreetMap"
    )


    for idx in selected:

        w = windows[idx]

        gt = w["ground_truth"]

        start_lat = float(
            w["context"]["last_lat"]
        )

        start_lon = float(
            w["context"]["last_lon"]
        )

        disp = np.asarray(
            gt["cum_disp_m"],
            dtype=float
        )

        north = disp[:, 0]
        east = disp[:, 1]

        lat, lon = displacement_to_latlon(
            start_lat,
            start_lon,
            north,
            east
        )

        coords = [
            [
                float(a),
                float(b)
            ]
            for a, b in zip(
                lat,
                lon
            )
        ]

        folium.PolyLine(
            coords,
            weight=4,
            opacity=0.8,
            tooltip=f"Window {idx}"
        ).add_to(fmap)

        folium.Marker(
            coords[0],
            popup=(
                f"Window {idx}<br>"
                f"Start"
            )
        ).add_to(fmap)

        folium.Marker(
            coords[-1],
            popup=(
                f"Window {idx}<br>"
                f"End<br>"
                f"Distance: "
                f"{gt['total_distance_m']:.1f} m"
            )
        ).add_to(fmap)


    map_path = os.path.join(
        OUT_DIR,
        "sample_60s_trajectories_map.html"
    )

    fmap.save(
        map_path
    )

    print(
        f"  Saved: {map_path}"
    )

elif not HAS_FOLIUM:

    print(
        "\n  folium is not installed."
    )

    print(
        "  Install with:"
    )

    print(
        "    pip install folium"
    )


# ============================================================================
# 14. FINAL DATASET VERDICT
# ============================================================================

print("\n" + "=" * 80)
print("14. DATASET VERDICT")
print("=" * 80)


problems = []

if len(windows) == 0:
    problems.append(
        "zero windows"
    )

if shape_errors:
    problems.append(
        f"{len(shape_errors)} shape errors"
    )

if numeric_bad:
    problems.append(
        f"{len(numeric_bad)} arrays with NaN/Inf"
    )

if overlap_count:
    problems.append(
        f"{overlap_count} overlapping windows"
    )


if problems:

    print("\n  ✗ DATASET NEEDS INVESTIGATION")

    for p in problems:
        print(
            f"    - {p}"
        )

else:

    print(
        "\n  ✓ STRUCTURAL DATASET CHECK PASSED"
    )

    print(
        "    No missing arrays"
    )

    print(
        "    No shape errors"
    )

    print(
        "    No NaN/Inf"
    )

    print(
        "    No overlapping consecutive windows"
    )


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nOutputs written to: {OUT_DIR}/"
)