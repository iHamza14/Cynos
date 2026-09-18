#!/usr/bin/env python3
"""GRU + causal TCN velocity head for blackout velocity estimation.

Input: per-second features (B, 60, 36), seed speed (B, 1).
Output: predicted velocity (B, 60) and per-second delta-v (B, 60).
"""
import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class VelocityDVSE(nn.Module):
    """GRU sequence encoder -> causal TCN velocity head.

    TCN receives GRU states concatenated with previous integrated speed.
    During training it uses the same autoregressive rollout as inference.
    """
    def __init__(self, input_dim=36, gru_hidden=96, tcn_channels=64,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.gru = nn.GRU(64, gru_hidden, batch_first=True)
        self.tcn_in = nn.Conv1d(gru_hidden + 1, tcn_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            *(ResidualTCNBlock(tcn_channels, d, dropout) for d in dilations)
        )
        self.dv_out = nn.Conv1d(tcn_channels, 1, kernel_size=1)

    def forward(self, features, speed_seed):
        # features: (B,T,36), seed: (B,1)
        encoded = self.encoder(features)
        states, _ = self.gru(encoded)                 # (B,T,H)

        # Build causal previous-speed sequence; step 0 sees seed.
        # Each next input uses prior predicted speed, not ground truth.
        b, steps, _ = states.shape
        prev_speed = speed_seed
        prev_speeds = []
        # First pass uses seed repeated to obtain a causal Δv proposal.
        # Re-roll TCN each step so later outputs can condition on prior predictions.
        speeds, dvs = [], []
        history = []
        for t in range(steps):
            history.append(torch.cat([states[:, t:t+1, :], prev_speed[:, None, :]], dim=-1))
            seq = torch.cat(history, dim=1).transpose(1, 2)
            hidden = self.tcn(self.tcn_in(seq))
            # hidden: (batch, channels, sequence_length)
            last_hidden = hidden[:, :, -1:]  # (batch, 64, 1)
            dv = self.dv_out(last_hidden).squeeze(-1)  # (batch, 1)
            prev_speed = prev_speed + dv
            speeds.append(prev_speed.squeeze(-1))
            dvs.append(dv.squeeze(-1))
        return torch.stack(speeds, dim=1), torch.stack(dvs, dim=1)
