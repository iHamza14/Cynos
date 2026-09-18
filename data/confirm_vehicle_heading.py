
"""
Plot vehicle Heading against available independent direction-related signals.

Diagnostic only. No heading convention is assumed and no axis/sign/offset
correction or lag correction is applied.

Outputs:
  data/heading_diagnostic/heading_signals.csv
  data/heading_diagnostic/heading_overlay.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

S_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv"
)
V_PATH = (
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv"
)
OUT_DIR = "data/heading_diagnostic"
HZ = 10
MERGE_TOLERANCE_S = 0.2


def find_col(df, *needles):
    """Prefer exact case-insensitive matches; otherwise require unique match."""
    normalized_needles = tuple(
        n.strip().casefold() for n in needles
    )

    exact_hits = [
        c for c in df.columns
        if c.strip().casefold() in normalized_needles
    ]

    if len(exact_hits) == 1:
        return exact_hits[0]

    if len(exact_hits) > 1:
        raise ValueError(
            f"Multiple exact matches for {needles}: {exact_hits}"
        )

    hits = [
        c for c in df.columns
        if all(
            n in c.casefold()
            for n in normalized_needles
        )
    ]

    if len(hits) != 1:
        raise ValueError(
            f"Expected one match for {needles}, got {hits}.\n"
            f"Available columns: {list(df.columns)}"
        )

    return hits[0]


def parse_clock(value):
    """
    Parse either:
      12:15:27:004
      YYYY-MM-DD 12:15:27:004

    Returns seconds since midnight.
    """
    try:
        time_part = str(value).strip().split()[-1]
        parts = time_part.split(":")

        if len(parts) != 4:
            return np.nan

        hh, mm, ss, ms = parts

        return (
            int(hh) * 3600
            + int(mm) * 60
            + int(ss)
            + int(ms) / 1000.0
        )

    except (ValueError, TypeError, IndexError):
        return np.nan


def wrap_deg(x):
    """Wrap angular differences to [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180) % 360 - 180


def numeric_column(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    s = pd.read_csv(S_PATH, encoding="latin-1")
    v = pd.read_csv(V_PATH, encoding="latin-1")

    s.columns = s.columns.str.strip()
    v.columns = v.columns.str.strip()

    # Parse clocks. Mobile timestamps are seconds since midnight.
    mobile_date_col = find_col(s, "DATE")
    vehicle_time_col = find_col(
        v, "Time Since Start of Day (seconds)"
    )

    s["abs_time"] = s[mobile_date_col].map(parse_clock)
    v["abs_time"] = pd.to_numeric(
        v[vehicle_time_col], errors="coerce"
    )

    s = (
        s.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["abs_time"])
        .sort_values("abs_time")
    )
    v = (
        v.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["abs_time"])
        .sort_values("abs_time")
    )

    if s.empty or v.empty:
        raise ValueError(
            "No valid timestamps remain after parsing. "
            "Inspect the source timestamp columns."
        )

    print(
        f"Mobile time range: "
        f"{s.abs_time.min():.3f} .. {s.abs_time.max():.3f} s"
    )
    print(
        f"Vehicle time range: "
        f"{v.abs_time.min():.3f} .. {v.abs_time.max():.3f} s"
    )

    # Resolve sensor columns.
    s_cols = {
        "gps_orientation_deg": find_col(s, "GPS ORIENTATION"),
        "phone_yaw_deg": find_col(s, "ORIENTATION (Yaw)"),
    }

    v_cols = {
        "vehicle_heading_deg": find_col(v, "Heading (degrees)"),
        "steering_deg": find_col(v, "Steering Angle (degrees)"),
        "yaw_rate_dps": find_col(v, "Yaw Rate (deg/sec)"),
        "vehicle_speed_kmh": find_col(v, "Velocity (km/hr)"),
    }

    sm = s[["abs_time", *s_cols.values()]].rename(
        columns={col: name for name, col in s_cols.items()}
    )

    vm = v[["abs_time", *v_cols.values()]].rename(
        columns={col: name for name, col in v_cols.items()}
    )

    # No lag correction: nearest timestamp match only.
    merged = pd.merge_asof(
        sm.sort_values("abs_time"),
        vm.sort_values("abs_time"),
        on="abs_time",
        direction="nearest",
        tolerance=MERGE_TOLERANCE_S,
    )

    merged = merged.dropna(
        subset=["vehicle_heading_deg"]
    ).copy()

    if merged.empty:
        raise ValueError(
            "No vehicle heading rows matched within "
            f"{MERGE_TOLERANCE_S}s. Check clock alignment."
        )

    # Ensure all sensor signals are numeric.
    for col in merged.columns:
        if col != "abs_time":
            merged[col] = pd.to_numeric(
                merged[col], errors="coerce"
            )

    # Remove invalid timestamps and sort before differentiation.
    merged = (
        merged.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["abs_time"])
        .sort_values("abs_time")
        .reset_index(drop=True)
    )

    # Duplicate timestamps make numerical differentiation ambiguous.
    # Keep the first observation at each timestamp.
    merged = merged.drop_duplicates(
        subset=["abs_time"], keep="first"
    ).reset_index(drop=True)

    if len(merged) < 2:
        raise ValueError(
            "Need at least two distinct timestamps to compute "
            "heading changes."
        )

    # Wrapped heading change.
    heading = merged["vehicle_heading_deg"].to_numpy(
        dtype=float, copy=True
    )
    times = merged["abs_time"].to_numpy(
        dtype=float, copy=True
    )

    heading_delta = np.full(len(merged), np.nan)
    heading_rate = np.full(len(merged), np.nan)

    dt = np.diff(times)
    d_heading = wrap_deg(np.diff(heading))

    valid_dt = np.isfinite(dt) & (dt > 0)
    valid_delta = np.isfinite(d_heading)
    valid = valid_dt & valid_delta

    heading_delta[1:][valid] = d_heading[valid]
    heading_rate[1:][valid] = d_heading[valid] / dt[valid]

    merged["vehicle_heading_wrapped_delta_deg"] = heading_delta
    merged["vehicle_heading_rate_dps"] = heading_rate

    output_csv = os.path.join(
        OUT_DIR, "heading_signals.csv"
    )
    merged.to_csv(output_csv, index=False)

    # Plot raw angles. Different reference frames may make direct
    # numerical comparison invalid; no normalization is applied.
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True
    )

    axes[0].plot(
        merged.abs_time,
        merged.vehicle_heading_deg,
        label="Vehicle Heading (raw)",
    )
    axes[0].plot(
        merged.abs_time,
        merged.gps_orientation_deg,
        label="Mobile GPS Orientation",
        alpha=0.75,
    )
    axes[0].plot(
        merged.abs_time,
        merged.phone_yaw_deg,
        label="Phone Orientation Yaw",
        alpha=0.75,
    )
    axes[0].set_ylabel("Degrees (raw conventions)")
    axes[0].legend()

    axes[1].plot(
        merged.abs_time,
        merged.vehicle_heading_rate_dps,
        label="Wrapped vehicle heading rate",
    )
    axes[1].plot(
        merged.abs_time,
        merged.yaw_rate_dps,
        label="Vehicle yaw rate",
    )
    axes[1].set_ylabel("deg/s")
    axes[1].legend()

    axes[2].plot(
        merged.abs_time,
        merged.steering_deg,
        label="Steering angle",
    )
    axes[2].plot(
        merged.abs_time,
        merged.vehicle_speed_kmh,
        label="Vehicle speed (km/h)",
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    fig.tight_layout()

    output_plot = os.path.join(
        OUT_DIR, "heading_overlay.png"
    )
    fig.savefig(output_plot, dpi=160)
    plt.close(fig)

    # Summary statistics.
    d = merged[
        "vehicle_heading_wrapped_delta_deg"
    ].dropna()

    r = merged[
        "vehicle_heading_rate_dps"
    ].dropna()

    print(f"\nRows after merge: {len(merged)}")

    print(
        "Vehicle heading raw range: "
        f"{merged.vehicle_heading_deg.min():.3f} .. "
        f"{merged.vehicle_heading_deg.max():.3f} deg"
    )

    if not d.empty:
        print(
            "Wrapped heading step change: "
            f"mean={d.mean():.4f}, "
            f"std={d.std():.4f}, "
            f"min={d.min():.4f}, "
            f"max={d.max():.4f} deg"
        )

    if not r.empty:
        print(
            "Heading rate: "
            f"mean={r.mean():.4f}, "
            f"std={r.std():.4f}, "
            f"min={r.min():.4f}, "
            f"max={r.max():.4f} deg/s"
        )

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved plot: {output_plot}")
    print(
        "\nNo lag, sign, axis, or offset correction was applied. "
        "Visual agreement does not establish equivalent heading frames."
    )

    # --- Diagnose unusually large vehicle heading jumps ---

    JUMP_THRESHOLD_DEG = 30.0

    heading = merged["vehicle_heading_deg"].to_numpy(
        dtype=float, copy=True
    )
    times = merged["abs_time"].to_numpy(
        dtype=float, copy=True
    )

    delta = wrap_deg(np.diff(heading))
    dt = np.diff(times)

    jump_mask = (
        np.isfinite(delta)
        & np.isfinite(dt)
        & (dt > 0)
        & (np.abs(delta) >= JUMP_THRESHOLD_DEG)
    )

    jump_indices = np.flatnonzero(jump_mask) + 1

    print("\n--- Large vehicle heading jumps ---")
    print(f"Threshold: {JUMP_THRESHOLD_DEG} degrees")
    print(f"Number of jumps: {len(jump_indices)}")

    if len(jump_indices) > 0:
        jump_rows = []

        for i in jump_indices:
            prev_i = i - 1

            jump_rows.append({
                "time_s": times[i],
                "dt_s": times[i] - times[prev_i],
                "heading_before_deg": heading[prev_i],
                "heading_after_deg": heading[i],
                "wrapped_delta_deg": delta[i - 1],
                "heading_rate_dps": (
                    delta[i - 1] / (times[i] - times[prev_i])
                ),
                "gps_orientation_deg": merged[
                    "gps_orientation_deg"
                ].iloc[i],
                "phone_yaw_deg": merged[
                    "phone_yaw_deg"
                ].iloc[i],
                "vehicle_yaw_rate_dps": merged[
                    "yaw_rate_dps"
                ].iloc[i],
                "steering_deg": merged[
                    "steering_deg"
                ].iloc[i],
                "vehicle_speed_kmh": merged[
                    "vehicle_speed_kmh"
                ].iloc[i],
            })

        jumps = pd.DataFrame(jump_rows)

        print(
            jumps.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        jumps.to_csv(
            os.path.join(OUT_DIR, "heading_jumps.csv"),
            index=False,
        )

        # Plot each jump with a small surrounding time window.
        for jump_number, i in enumerate(jump_indices):
            center_time = times[i]
            window = 5.0

            local = merged[
                (merged["abs_time"] >= center_time - window)
                & (merged["abs_time"] <= center_time + window)
            ].copy()

            if len(local) < 2:
                continue

            fig, axes = plt.subplots(
                3, 1, figsize=(12, 9), sharex=True
            )

            axes[0].plot(
                local.abs_time,
                local.vehicle_heading_deg,
                label="Vehicle heading (target)",
            )
            axes[0].plot(
                local.abs_time,
                local.gps_orientation_deg,
                label="Mobile GPS orientation",
                alpha=0.8,
            )
            axes[0].axvline(
                center_time,
                linestyle="--",
                label="Heading jump",
            )
            axes[0].set_ylabel("Raw degrees")
            axes[0].legend()

            axes[1].plot(
                local.abs_time,
                local.vehicle_heading_rate_dps,
                label="Vehicle heading rate",
            )
            axes[1].plot(
                local.abs_time,
                local.yaw_rate_dps,
                label="Vehicle yaw rate",
            )
            axes[1].axvline(center_time, linestyle="--")
            axes[1].set_ylabel("deg/s")
            axes[1].legend()

            axes[2].plot(
                local.abs_time,
                local.steering_deg,
                label="Steering angle",
            )
            axes[2].plot(
                local.abs_time,
                local.vehicle_speed_kmh,
                label="Vehicle speed (km/h)",
            )
            axes[2].axvline(center_time, linestyle="--")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()

            fig.suptitle(
                f"Heading jump {jump_number + 1}: "
                f"{delta[i - 1]:.1f} degrees"
            )
            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    OUT_DIR,
                    f"heading_jump_{jump_number + 1:03d}.png",
                ),
                dpi=150,
            )
            plt.close(fig)

    else:
        print("No jumps exceeded the threshold.")


if __name__ == "__main__":
    main()

