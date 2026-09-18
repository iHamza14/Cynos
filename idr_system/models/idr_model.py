"""
IDRModel — neural network container.

This class owns the two neural components:
  MTN          → phone-to-vehicle rotation per 1s window
  NoiseNet     → Δv, Δψ, psi_rate_corr per 1s window

The EKF rollout happens in the training loop (scripts/train.py), not here.
Rationale: keeping the EKF loop at the top level makes it easier to change
numerical details (detaching P, autocast boundaries, per-step diagnostics)
without touching the model definition.

If you later want a self-contained inference module, add a separate
IDRInference wrapper that composes this model with the EKF.
"""

import torch
import torch.nn as nn

from models.mtn import MTN
from models.noise_net import NoiseCompensationNet


class IDRModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        mtn_cfg = config['model']['mtn']
        noise_cfg = config['model']['noise_net']

        self.mtn = MTN(
            input_channels=mtn_cfg['input_channels'],
            hidden_dim=mtn_cfg['hidden_dim'],
            dropout=mtn_cfg['dropout'],
        )

        self.noise_net = NoiseCompensationNet(
            input_dim=noise_cfg['input_channels'],
            hidden_dim=noise_cfg['hidden_dim'],
        )

    def forward(self, mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat):
        """
        Args:
            mtn_input       (B, T_1hz, 7)
            accel_feat      (B, T_1hz, 18)
            gyro_feat       (B, T_1hz, 18)
            fft_feat        (B, T_1hz, 15)
            broadcast_feat  (B, T_1hz, 3)

        Returns dict:
            roll             (B, T_1hz)
            pitch            (B, T_1hz)
            yaw_residual     (B, T_1hz)
            dv               (B, T_1hz)
            dpsi             (B, T_1hz)
            psi_rate_corr    (B, T_1hz)
        """
        roll, pitch, yaw_residual = self.mtn(mtn_input)

        dv, dpsi, psi_rate_corr = self.noise_net(
            accel_feat, gyro_feat, fft_feat, broadcast_feat,
        )

        return {
            'roll': roll,
            'pitch': pitch,
            'yaw_residual': yaw_residual,
            'dv': dv,
            'dpsi': dpsi,
            'psi_rate_corr': psi_rate_corr,
        }