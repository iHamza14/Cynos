"""
MTN — Motion Transformation Network.

Estimates the phone-to-vehicle rotation from IMU context.
Used by IDRModel to rotate body-frame accel into vehicle frame.

Input:  (B, T, input_channels)  with T ~ 30s at 1 Hz
Output: roll, pitch, yaw_offset — each (B, T)
        These are the Euler angles of R_pv (phone → vehicle).
"""

import torch
import torch.nn as nn

from models.temporal_block import TemporalBlock


class MTNBranch(nn.Module):
    """
    A TCN branch: one TemporalBlock per dilation value in `dilations`.
    Receptive field ≈ 1 + 6·Σ(dilations)  (kernel_size=3, 3 convs per block).
    """

    def __init__(self, in_channels, out_channels, dilations,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, d in enumerate(dilations):
            in_ch = in_channels if i == 0 else out_channels
            padding = (kernel_size - 1) * d
            layers.append(TemporalBlock(
                in_ch, out_channels,
                kernel_size=kernel_size,
                dilation=d, dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        return self.network(x)


class MTN(nn.Module):
    """
    Multi-scale TCN estimating phone→vehicle rotation.

    Two parallel branches:
      - Fast (short RF ~ 19 samples): captures roll/pitch dynamics
      - Slow (RF ~ 37 samples): captures slow yaw drift + context

    Both RFs are tuned for a 30-sample (30s) context.
    """

    def __init__(self, input_channels=7, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_channels, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, hidden_dim)
        self.relu2 = nn.ReLU()

        # Fast branch: RF = 1 + 6·(1+2) = 19 samples
        self.branch_a = MTNBranch(
            hidden_dim, hidden_dim,
            dilations=[1, 2], kernel_size=3, dropout=dropout,
        )

        # Slow branch: RF = 1 + 6·(2+4) = 37 samples
        self.branch_b = MTNBranch(
            hidden_dim, hidden_dim,
            dilations=[2, 4], kernel_size=3, dropout=dropout,
        )

        self.fc3 = nn.Linear(hidden_dim * 2, 64)
        self.relu3 = nn.ReLU()

        # Output heads
        self.yaw_head = nn.Linear(64, 1)
        self.pitch_roll_head = nn.Linear(64, 2)

        # Initialize all three heads near zero so R_pv starts near identity.
        # (Training then learns whatever offset the phone mount requires.)
        nn.init.normal_(self.yaw_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.yaw_head.bias, 0.0)
        nn.init.normal_(self.pitch_roll_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.pitch_roll_head.bias, 0.0)

    def forward(self, x):
        # x: (B, T, input_channels)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        # TCN expects (B, C, T)
        x_t = x.transpose(1, 2)

        out_a = self.branch_a(x_t)     # (B, hidden_dim, T)
        out_b = self.branch_b(x_t)     # (B, hidden_dim, T)

        out = torch.cat([out_a, out_b], dim=1)   # (B, 2*hidden_dim, T)
        out = out.transpose(1, 2)                # (B, T, 2*hidden_dim)

        out = self.relu3(self.fc3(out))          # (B, T, 64)

        yaw_offset = self.yaw_head(out).squeeze(-1)         # (B, T)
        pitch_roll = self.pitch_roll_head(out)              # (B, T, 2)
        roll  = pitch_roll[:, :, 0]
        pitch = pitch_roll[:, :, 1]

        return roll, pitch, yaw_offset