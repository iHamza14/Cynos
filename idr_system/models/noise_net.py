"""
NoiseCompensationNet — GRU-based velocity / heading correction.

Inputs (all per timestep):
    accel_feat      (B, T, 18)   time-domain stats (std/max/min/RMS/skew/kurt)
    gyro_feat       (B, T, 18)   same
    fft_feat        (B, T, 15)   FFT magnitudes, 3 axes × 5 bands
    broadcast_feat  (B, T, 3)    vr_seed, gps_age, shock_mask_sum

Outputs (all per timestep):
    dv              (B, T)   forward velocity correction (m/s)
    dpsi            (B, T)   heading correction (rad)
    psi_rate_corr   (B, T)   yaw gyro-rate correction (rad/s)
"""

import torch
import torch.nn as nn


class NoiseCompensationNet(nn.Module):

    def __init__(self, input_dim=147, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ── Embedding branches ────────────────────────────────────────
        self.accel_mlp = nn.Sequential(
            nn.Linear(18, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        self.gyro_mlp = nn.Sequential(
            nn.Linear(18, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        self.fft_mlp = nn.Sequential(
            nn.Linear(15, 16), nn.ReLU(),
        )

        # 64 (accel) + 64 (gyro) + 16 (fft) + 3 (broadcast) = 147
        expected_dim = 64 + 64 + 16 + 3
        assert input_dim == expected_dim, (
            f"input_dim={input_dim} does not match concatenated feature "
            f"dim {expected_dim}. Update MLP widths or broadcast_feat."
        )

        # ── Temporal core ─────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ── Shared trunk ──────────────────────────────────────────────
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )

        # ── Output heads ──────────────────────────────────────────────
        self.head_dv            = nn.Linear(16, 1)
        self.head_dpsi          = nn.Linear(16, 1)
        self.head_psi_rate_corr = nn.Linear(16, 1)

        # Init heads near zero so corrections don't destabilize the EKF
        # at the start of training.
        for head in (self.head_dv, self.head_dpsi, self.head_psi_rate_corr):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, accel_feat, gyro_feat, fft_feat, broadcast_feat):
        a_out = self.accel_mlp(accel_feat)          # (B, T, 64)
        g_out = self.gyro_mlp(gyro_feat)            # (B, T, 64)
        f_out = self.fft_mlp(fft_feat)              # (B, T, 16)

        x = torch.cat([a_out, g_out, f_out, broadcast_feat], dim=-1)

        assert x.shape[-1] == self.input_dim, (
            f"Concatenated feature dim {x.shape[-1]} != "
            f"input_dim {self.input_dim}"
        )

        gru_out, _ = self.gru(x)                    # (B, T, hidden_dim)
        trunk = self.shared_trunk(gru_out)          # (B, T, 16)

        dv            = self.head_dv(trunk).squeeze(-1)
        dpsi          = self.head_dpsi(trunk).squeeze(-1)
        psi_rate_corr = self.head_psi_rate_corr(trunk).squeeze(-1)

        return dv, dpsi, psi_rate_corr