"""
Anti-vibration low-pass filter and shock detection.

F2 finding: smartphone accel has a strong aliased band at 4.2–4.7 Hz.
We low-pass accel at 2.5 Hz before feeding physics integration.

FFT features for the network use UNFILTERED accel (see dataset.py) so the
network can still see the vibration band and learn to reject it.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_anti_vibration_filter(accel_data, sample_rate_hz=10,
                                cutoff_hz=2.5, order=4):
    """
    Zero-phase Butterworth low-pass on accel. Applies per-axis.

    Args:
        accel_data: (N,) or (N, 3) numpy array.
        sample_rate_hz: input sample rate.
        cutoff_hz: -3 dB cutoff.
        order: filter order.

    Returns:
        Filtered array, same shape as input.
    """
    accel_data = np.asarray(accel_data)

    # Not enough samples to filter reliably — return input unchanged.
    if accel_data.shape[0] < 3 * order:
        return accel_data.copy()

    nyq = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Default padlen used by filtfilt is 3 * max(len(a), len(b)).
    # Use something smaller for short signals but still safe.
    default_padlen = 3 * max(len(a), len(b))
    padlen = min(default_padlen, accel_data.shape[0] - 1)

    if accel_data.ndim == 1:
        return filtfilt(b, a, accel_data, padlen=padlen)

    filtered = np.empty_like(accel_data)
    for i in range(accel_data.shape[1]):
        filtered[:, i] = filtfilt(b, a, accel_data[:, i], padlen=padlen)
    return filtered


def compute_shock_mask(accel_lin, window_size=10, sigma_multiplier=3.0):
    """
    Binary mask where |accel_lin| exceeds local mean + k·std.

    The reference window EXCLUDES the current sample to avoid the sample
    inflating its own threshold.

    Args:
        accel_lin: (N, 3) linear acceleration (gravity removed).
        window_size: half-window size; total window = 2*window_size samples.
        sigma_multiplier: k in mean + k·std.

    Returns:
        (N,) float32 array, 1.0 where shock detected, 0.0 otherwise.
    """
    accel_lin = np.asarray(accel_lin)
    accel_norm = np.linalg.norm(accel_lin, axis=1)
    N = accel_norm.shape[0]
    mask = np.zeros(N, dtype=np.float32)

    if N < 3:
        return mask

    for i in range(N):
        # Reference window excludes index i
        lo = max(0, i - window_size)
        hi = min(N, i + window_size + 1)
        if i == lo:
            window = accel_norm[lo + 1:hi]
        elif i == hi - 1:
            window = accel_norm[lo:hi - 1]
        else:
            window = np.concatenate([accel_norm[lo:i], accel_norm[i + 1:hi]])

        if window.size < 3:
            continue

        mean_val = window.mean()
        std_val = window.std()

        if accel_norm[i] > mean_val + sigma_multiplier * std_val:
            mask[i] = 1.0

    return mask