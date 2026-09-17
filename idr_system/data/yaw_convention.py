import numpy as np

def wrap_angle_rad(angle_rad):
    """Wraps an angle in radians to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def wrap_angle_deg(angle_deg):
    """Wraps an angle in degrees to [-180, 180]."""
    return (angle_deg + 180) % 360 - 180

def apply_yaw_correction(orientation_yaw_deg):
    """
    Applies the F3 constant offset to the Android ORIENTATION_Yaw channel.
    offset = 138.7 degrees.
    Returns:
        corrected_yaw_rad (np.ndarray): The corrected yaw in radians, wrapped to [-pi, pi].
    """
    offset_deg = 138.7
    corrected_deg = wrap_angle_deg(orientation_yaw_deg + offset_deg)
    return np.deg2rad(corrected_deg)

def remount_detector(orientation_yaw_rad, v_heading_rad, speed_kmh, yaw_rate_deg_s, 
                     speed_thresh_kmh=15.0, yaw_rate_thresh_deg_s=2.0, min_samples=600):
    """
    Detects if the phone has been remounted by checking the alignment between 
    the corrected phone orientation and the vehicle heading during straight driving.
    
    Args:
        orientation_yaw_rad (np.ndarray): Phone's corrected orientation yaw in rad.
        v_heading_rad (np.ndarray): Vehicle's true heading in rad.
        speed_kmh (np.ndarray): Vehicle speed in km/h.
        yaw_rate_deg_s (np.ndarray): Vehicle yaw rate in deg/s.
        speed_thresh_kmh (float): Minimum speed to consider.
        yaw_rate_thresh_deg_s (float): Maximum yaw rate to consider 'straight'.
        min_samples (int): Minimum number of valid samples to return a confident estimate.
        
    Returns:
        is_remounted (bool): True if remount is detected.
        current_offset_deg (float): The detected offset in degrees.
        confidence (float): Confidence in the offset estimate (0 to 1).
    """
    valid_idx = (speed_kmh > speed_thresh_kmh) & (np.abs(yaw_rate_deg_s) < yaw_rate_thresh_deg_s)
    
    if np.sum(valid_idx) < min_samples:
        return False, 0.0, 0.0
        
    diff_rad = wrap_angle_rad(orientation_yaw_rad[valid_idx] - v_heading_rad[valid_idx])
    
    # Circular mean
    S = np.sum(np.sin(diff_rad))
    C = np.sum(np.cos(diff_rad))
    
    N = len(diff_rad)
    
    mean_diff_rad = np.arctan2(S, C)
    resultant_length = np.sqrt(S**2 + C**2) / N
    
    current_offset_deg = np.rad2deg(mean_diff_rad)
    confidence = resultant_length
    
    # We consider it remounted if the new offset is significant (> 10 degrees)
    # Note: since orientation_yaw_rad is already F3 corrected, the diff should be near 0.
    is_remounted = bool(np.abs(current_offset_deg) > 10.0 and confidence > 0.9)
    
    return is_remounted, current_offset_deg, confidence

