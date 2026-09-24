"""
Extracts 1-second physical features and pre-integrates IMU data.
"""

import torch


def extract_1sec_features(window, hz=10):
    """
    Extracts 1-second statistical features (std, max, min, rms, skewness, kurtosis)
    for a given sensor window.

    Args:
        window: Tensor of shape [B, L, 3] representing L samples of hz-rate data.
        hz: Sampling rate.

    Returns:
        Tensor of shape [B, T, 18] containing the 6 statistics for each of the 3 axes, where T = L // hz.
    """
    B, L, _ = window.shape
    T = L // hz
    # Reshape into [B, T, hz, 3] to isolate 1-second (hz samples) blocks
    windows = window[:, : T * hz, :].view(B, T, hz, 3)

    mean = windows.mean(dim=2, keepdim=True)
    std = windows.std(dim=2, unbiased=False, keepdim=True)

    max_val = windows.max(dim=2, keepdim=True)[0]
    min_val = windows.min(dim=2, keepdim=True)[0]

    rms = torch.sqrt(torch.mean(windows**2, dim=2, keepdim=True))

    eps = 1e-8
    skew = torch.mean((windows - mean) ** 3, dim=2, keepdim=True) / (std**3 + eps)
    kurt = torch.mean((windows - mean) ** 4, dim=2, keepdim=True) / (std**4 + eps) - 3.0

    # [B, T, 1, 3]
    features = torch.cat([std, max_val, min_val, rms, skew, kurt], dim=3)  # [B, T, 1, 18]
    return features.squeeze(2)


def preintegrate_acceleration(acc_window, hz=10):
    """
    Preintegrates raw acceleration over 1-second intervals.

    Args:
        acc_window: Tensor of shape [B, L, 3]
        hz: Sampling rate.

    Returns:
        Tensor of shape [B, T, 3]
    """
    dt = 1.0 / hz
    B, L, _ = acc_window.shape
    T = L // hz
    windows = acc_window[:, : T * hz, :].view(B, T, hz, 3)
    I = torch.sum(windows * dt, dim=2)
    return I
