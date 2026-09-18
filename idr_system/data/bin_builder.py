"""
Enhanced bin builder.

Delegates raw parsing to helper.bin_dataset, then adds IDR-specific fields:
  - GPS age tracking (F1: phone GPS is stale ~98.9% of the time while moving)
  - Per-step yaw rate from vehicle heading (rad/s)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from helper import bin_dataset as helper_bin_dataset
from helper import cumulative_displacement, total_distance_m


# GPS speed change threshold (km/h). Below this, treat as the same fix.
GPS_SPEED_CHANGE_THRESHOLD = 0.05


def bin_dataset(df_s, df_v):
    bins = helper_bin_dataset(df_s, df_v)

    last_gps_speed = None
    time_since_last_gps = 0.0

    for i, b in enumerate(bins):
        # ── GPS age tracking (F1) ────────────────────────────────────────
        current_gps_speed = float(b.get('mobile_gps_speed', 0.0))
        if (last_gps_speed is None
                or abs(current_gps_speed - last_gps_speed) > GPS_SPEED_CHANGE_THRESHOLD):
            time_since_last_gps = 0.0
            last_gps_speed = current_gps_speed
        else:
            time_since_last_gps += 0.1  # 10 Hz → 0.1 s step
        b['mobile_gps_age_sec'] = time_since_last_gps

        # ── Per-step yaw rate from vehicle heading (rad/s) ───────────────
        if i == 0:
            b['v_yaw_rate_rad_s'] = 0.0
            b['v_yaw_rate_deg_s'] = 0.0
        else:
            prev_hdg = bins[i - 1]['v_heading_rad']
            curr_hdg = b['v_heading_rad']
            yaw_rate = (curr_hdg - prev_hdg) / 0.1
            b['v_yaw_rate_rad_s'] = yaw_rate
            b['v_yaw_rate_deg_s'] = np.rad2deg(yaw_rate)

    # ── Sanity check — one bin, one time ──────────────────────────────────
    if not hasattr(bin_dataset, '_printed') and len(bins) > 100:
        b0 = bins[len(bins) // 2]
        gt_deg = np.rad2deg(b0['v_heading_rad']) % 360
        print(f"Sanity — bin {len(bins) // 2}:")
        print(f"  GT heading (deg, compass):  {gt_deg:.1f}")
        print(f"  GPS age at that bin (s):    {b0['mobile_gps_age_sec']:.1f}")
        print(f"  yaw rate (deg/s):           {b0['v_yaw_rate_deg_s']:.2f}")
        bin_dataset._printed = True

    return bins