import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseNetwork(nn.Module):
    """Stage 1: Predict delta-v from raw IMU and velocity."""

    def __init__(self):
        super().__init__()

        self.fc_acc = nn.Linear(3, 64)
        self.fc_gyro = nn.Linear(3, 64)
        self.fc_vr = nn.Linear(1, 16)

        self.gru = nn.GRU(
            input_size=144,
            hidden_size=128,
            batch_first=True,
        )

        self.fc_out = nn.Linear(128, 1)

    def forward(self, acc_feat, gyro_feat, vr, hidden=None):
        # acc_feat: (B, T, 3)
        # gyro_feat: (B, T, 3)
        # vr: (B, T, 1)
        # hidden: (1, B, 128) or None

        a = F.relu(self.fc_acc(acc_feat))
        g = F.relu(self.fc_gyro(gyro_feat))
        v = F.relu(self.fc_vr(vr))

        merged = torch.cat([a, g, v], dim=-1)

        out, hidden = self.gru(merged, hidden)

        delta_v = self.fc_out(out)

        return delta_v, hidden
class MTN(nn.Module):
    """Stage 2: Predicts Euler angles (pose) directly from mtn_input."""
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        
    def forward(self, mtn_input):
        # mtn_input is (B, T, 6). TCN expects (B, C, T)
        x = mtn_input.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) # (B, 3, T)
        return x.permute(0, 2, 1) # (B, T, 3)

class DVSE(nn.Module):
    """Combined model for inference wrapper"""
    def __init__(self):
        super().__init__()
        self.noise_net = NoiseNetwork()
        self.mtn = MTN()
        
    def forward(self, acc_feat, gyro_feat, vr, mtn_input, raw_accel, gravity):
        delta_v = self.noise_net(acc_feat, gyro_feat, vr)
        euler = self.mtn(mtn_input)
        return delta_v, euler