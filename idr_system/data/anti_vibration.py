import numpy as np
from scipy.signal import butter, filtfilt

def apply_anti_vibration_filter(accel_data, sample_rate_hz=10, cutoff_hz=2.5, order=4):
    """
    Applies a zero-phase low-pass Butterworth filter to remove engine vibration.
    
    Args:
        accel_data (np.ndarray): (N, 3) or (N,) array of acceleration data.
        sample_rate_hz (float): Sampling frequency in Hz.
        cutoff_hz (float): Cutoff frequency in Hz.
        order (int): Butterworth filter order.
        
    Returns:
        np.ndarray: Filtered acceleration data of the same shape.
    """
    nyq = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Need to handle potential edge cases where sequence is too short
    padlen = min(15, len(accel_data) - 1) 
    
    if accel_data.ndim == 1:
        return filtfilt(b, a, accel_data, padlen=padlen)
    else:
        filtered = np.zeros_like(accel_data)
        for i in range(accel_data.shape[1]):
            filtered[:, i] = filtfilt(b, a, accel_data[:, i], padlen=padlen)
        return filtered

def compute_shock_mask(accel_lin, window_size=10, sigma_multiplier=3.0):
    """
    Computes a shock mask where linear acceleration exceeds running mean + 3*sigma.
    
    Args:
        accel_lin (np.ndarray): (N, 3) linear acceleration.
        window_size (int): Running window size.
        sigma_multiplier (float): Multiplier for standard deviation.
        
    Returns:
        np.ndarray: (N,) binary shock mask (1 if shock, 0 otherwise).
    """
    accel_norm = np.linalg.norm(accel_lin, axis=1)
    N = len(accel_norm)
    mask = np.zeros(N, dtype=np.int32)
    
    # We use a simple moving window calculation
    for i in range(N):
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)
        window = accel_norm[start:end]
        
        mean_val = np.mean(window)
        std_val = np.std(window)
        
        if accel_norm[i] > mean_val + sigma_multiplier * std_val:
            mask[i] = 1
            
    return mask

