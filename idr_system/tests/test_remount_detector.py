import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yaw_convention import remount_detector

def test_remount_detector():
    """
    Test remount detector with a synthetic 30 degree offset.
    Should fire within 60s of straight driving (600 samples at 10Hz).
    """
    np.random.seed(42)
    N = 700  # 70 seconds
    
    # Speed > 15 km/h, yaw_rate < 2 deg/s
    speed_kmh = np.random.uniform(20, 30, N)
    yaw_rate_deg_s = np.random.normal(0, 0.5, N)
    
    # Vehicle heading
    v_heading_rad = np.random.normal(0, 0.1, N)
    
    # Phone orientation has a 30 degree offset + noise
    offset_deg = 30.0
    offset_rad = np.deg2rad(offset_deg)
    
    # Simulate some noise
    noise = np.random.normal(0, np.deg2rad(2.0), N)
    
    orientation_yaw_rad = v_heading_rad + offset_rad + noise
    
    # Test for the first 600 samples (60s)
    is_remounted, current_offset_deg, confidence = remount_detector(
        orientation_yaw_rad[:600], 
        v_heading_rad[:600], 
        speed_kmh[:600], 
        yaw_rate_deg_s[:600],
        min_samples=600
    )
    
    print(f"Detected Remount: {is_remounted}")
    print(f"Offset (deg): {current_offset_deg:.2f}")
    print(f"Confidence: {confidence:.4f}")
    
    assert is_remounted is True, "Detector failed to identify remount."
    assert np.abs(current_offset_deg - 30.0) < 1.0, f"Offset estimation wrong: {current_offset_deg}"
    assert confidence > 0.9, "Confidence too low."
    
    print("PASS: Remount detector works as expected.")

if __name__ == "__main__":
    test_remount_detector()
