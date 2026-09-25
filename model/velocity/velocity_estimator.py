"""
Core DVSE (Deep Velocity Speed Estimator) model architecture.
"""

import torch
import torch.nn as nn

from .velocity_components import MTN, NCN
from .feature_extractor import extract_1sec_features, preintegrate_acceleration
from .physics_layer import euler_to_rotation_matrix, physics_velocity_update

class DVSEModel(nn.Module):
    """Enterprise class definition for DVSEModel."""

    def __init__(self,scalers):
        """Initializes the instance."""
        super().__init__()
        self.ncn = NCN()
        self.mtn = MTN()
        self.register_buffer(
            "acc_feat_mean",
            torch.tensor(scalers["acc_features"].mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "acc_feat_scale",
            torch.tensor(scalers["acc_features"].scale_, dtype=torch.float32),
        )
        self.register_buffer(
            "gyro_feat_mean",
            torch.tensor(scalers["gyro_features"].mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "gyro_feat_scale",
            torch.tensor(scalers["gyro_features"].scale_, dtype=torch.float32),
        )
    def scale_features(self, x, mean, scale):
        return (x - mean) / scale

    def forward(self, acc_window, gyro_window, v_r, v_0=None, hz=10):
        """
        Args:
            acc_window: [B, T*hz, 3] raw acceleration
            gyro_window: [B, T*hz, 3] raw gyro
            v_r: [B, T, 1] reference velocity sequence (now constantly vr_seed)
            v_0: [B, 1] Initial velocity at the start of the window.
            hz: Sampling rate.
        """
        # Feature Extraction
        acc_feat = extract_1sec_features(acc_window, hz=hz)  # [B, T, 18]
        gyro_feat = extract_1sec_features(gyro_window, hz=hz)  # [B, T, 18]
        acc_feat = self.scale_features(
            acc_feat, self.acc_feat_mean, self.acc_feat_scale
        )
        gyro_feat = self.scale_features(
            gyro_feat, self.gyro_feat_mean, self.gyro_feat_scale
        )
        # NCN (Fast Parallel since v_r is known)
        N_dist = self.ncn(acc_feat, gyro_feat, v_r)  # [B, T, 1]

        # MTN
        I = preintegrate_acceleration(acc_window, hz=hz)  # [B, T, 3]
        g_w = torch.tensor([0.0, 0.0, 9.81], dtype=I.dtype, device=I.device)
        g_w_expand = g_w.view(1, 1, 3).expand(I.shape[0], I.shape[1], 3)
        i_feat = torch.cat([I, g_w_expand], dim=-1)  # [B, T, 6]
        angles = self.mtn(i_feat)  # [B, T, 3]

        # Physics Base
        R_vp = euler_to_rotation_matrix(angles)  # [B, T, 3, 3]
        delta_v, v, _ = physics_velocity_update(
            acc_window=acc_window, R_vp=R_vp, N=N_dist, v_0=v_0, hz=hz
        )

        return delta_v, v, N_dist, angles
