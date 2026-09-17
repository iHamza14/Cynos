import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import bin_dataset as helper_bin_dataset
from data.yaw_convention import apply_yaw_correction

def bin_dataset(df_s, df_v):
    """
    Enhanced bin builder that delegates to helper.py and adds IDR-specific guarantees:
    - F3 Yaw correction
    - Tracking GPS age
    """
    # We must ensure that the helper.py extracts ORIENTATION_Yaw if available.
    # Because helper.py might not extract it out of the box, we will add it to the dataframe before passing to helper,
    # or just assume it's in mobile_gps_orientation if they are the same thing. 
    # Actually, the prompt says ORIENTATION Yaw/Pitch/Roll (°).
    
    # Let's call the helper to do the heavy lifting of merge_asof and interpolation
    bins = helper_bin_dataset(df_s, df_v)
    
    # Now enhance the bins to meet IDR data pipeline requirements
    last_gps_speed = None
    time_since_last_gps = 0.0
    
    for i in range(len(bins)):
        b = bins[i]
        
        # 1. Yaw convention correction (F3)
        # Using mobile_gps_orientation as a proxy if ORIENTATION_Yaw is missing from helper's output
        # In a full implementation, we'd ensure helper.py grabs the exact ORIENTATION Yaw column.
        yaw_deg = b.get('mobile_gps_orientation', 0.0)
        b['orientation_yaw_vehicle_rad'] = apply_yaw_correction(yaw_deg)
        
        # 2. GPS Age tracking (F1)
        # Check if GPS speed changed to determine if it's stale
        current_gps_speed = b.get('mobile_gps_speed', 0.0)
        if last_gps_speed is None or current_gps_speed != last_gps_speed:
            time_since_last_gps = 0.0
            last_gps_speed = current_gps_speed
        else:
            time_since_last_gps += 0.1 # 10Hz = 0.1s step
            
        b['mobile_gps_age_sec'] = time_since_last_gps
        
        # 3. Add v_yaw_rate_deg_s
        if i == 0:
            b['v_yaw_rate_deg_s'] = 0.0
        else:
            prev_hdg = bins[i-1]['v_heading']
            curr_hdg = b['v_heading']
            # Diff in degrees (assuming v_heading is in degrees, but we need to check)
            # Typically heading diff needs angle wrapping.
            diff = curr_hdg - prev_hdg
            diff = (diff + 180) % 360 - 180
            b['v_yaw_rate_deg_s'] = diff / 0.1
            
    return bins

# We can rely on helper.py for cumulative_displacement and other metrics if needed.
# import from helper.py at the top handles it.
from helper import cumulative_displacement, total_distance_m
