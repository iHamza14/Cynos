"""
TCN building blocks — causal, dilated, residual.

Each TemporalBlock consists of 3 dilated causal Conv1d layers
with a residual connection. Receptive field contribution:
    RF_block = 3 · (kernel_size − 1) · dilation

Stacked blocks (dilations d_1, ..., d_N):
    RF_total = 1 + 3 · (kernel_size − 1) · Σ d_i
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove `chomp_size` elements from the right to enforce causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN block: 3 dilated causal Conv1d layers + residual.

    `padding` is computed internally from kernel_size and dilation.
    Dropout is applied once per block (after the last conv).
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp3 = Chomp1d(padding)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.final_relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(conv.weight, mode='fan_in',
                                    nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in',
                                    nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x);  out = self.chomp1(out);  out = self.relu1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out)
        out = self.conv3(out); out = self.chomp3(out); out = self.relu3(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)