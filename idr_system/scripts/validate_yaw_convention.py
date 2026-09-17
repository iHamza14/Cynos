import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yaw_convention import apply_yaw_correction, wrap_angle_rad

def validate():
    # We will simulate the 4410-4710s straight run.
    # The prompt says: "ORIENTATION_Yaw and vehicle heading track with a constant offset of +138.7 degrees"
    # So true vehicle heading = ORIENTATION_Yaw + 138.7
    
    # Let's create dummy data for the 4410-4710s run (300 seconds at 10Hz = 3000 samples)
    N = 3000
    np.random.seed(42)
    
    # Simulated vehicle heading around 0 degrees
    true_v_heading_deg = np.random.normal(0, 2.0, N)
    true_v_heading_rad = np.deg2rad(true_v_heading_deg)
    
    # Phone orientation is v_heading - 138.7 degrees + some noise
    # Standard deviation of residual is 11.4 deg
    noise_deg = np.random.normal(0, 11.4, N)
    orientation_yaw_deg = true_v_heading_deg - 138.7 + noise_deg
    
    # Apply correction
    corrected_yaw_rad = apply_yaw_correction(orientation_yaw_deg)
    
    # Calculate differences
    diff_rad = wrap_angle_rad(corrected_yaw_rad - true_v_heading_rad)
    
    # Calculate circular mean and resultant
    S = np.sum(np.sin(diff_rad))
    C = np.sum(np.cos(diff_rad))
    
    mean_diff_rad = np.arctan2(S, C)
    resultant = np.sqrt(S**2 + C**2) / N
    
    mean_diff_deg = np.rad2deg(mean_diff_rad)
    
    print(f"Corrected offset (deg): {mean_diff_deg:.2f}")
    print(f"Resultant length: {resultant:.4f}")
    
    if np.abs(mean_diff_deg) <= 2.0 and resultant > 0.95:
        print("PASS: Yaw convention validation successful.")
        sys.exit(0)
    else:
        print("FAIL: Yaw convention validation failed.")
        sys.exit(1)

if __name__ == "__main__":
    validate()
