import torch
import torch.nn as nn
import torch.nn.functional as F

class NCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.acc_embed = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.gyro_embed = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.regression = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, acc_feat, gyro_feat, v_r):
        """
        Args:
            acc_feat: [B, 10, 18]
            gyro_feat: [B, 10, 18]
            v_r: [B, 10, 1]
        Returns:
            N: [B, 10, 1]
        """
        a_emb = self.acc_embed(acc_feat)
        g_emb = self.gyro_embed(gyro_feat)
        x = torch.cat([a_emb, g_emb], dim=-1) # [B, 10, 128]
        gru_out, _ = self.gru(x) # [B, 10, 128]
        x_reg = torch.cat([gru_out, v_r], dim=-1) # [B, 10, 129]
        n = self.regression(x_reg) # [B, 10, 1]
        return n

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, 
            padding=(kernel_size - 1) * dilation, 
            dilation=dilation, **kwargs
        )
    
    def forward(self, x):
        return super().forward(x)[..., :-self.padding[0]]

class MTNTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv3 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(x + res)

class MTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.tcn = nn.Sequential(
            MTNTCNBlock(64, kernel_size=2, dilation=1),
            MTNTCNBlock(64, kernel_size=2, dilation=2)
        )
        self.head = nn.Linear(64, 3)

    def forward(self, i_feat):
        """
        Args:
            i_feat: [B, 10, 6] (preintegration + gravity)
        Returns:
            angles: [B, 10, 3] (alpha, beta, gamma)
        """
        x = self.embed(i_feat) # [B, 10, 64]
        x = x.transpose(1, 2) # [B, 64, 10]
        x = self.tcn(x) # [B, 64, 10]
        x = x.transpose(1, 2) # [B, 10, 64]
        angles = self.head(x) # [B, 10, 3]
        return angles

class CausalBlock(nn.Module):
    """Used by GyroTCN"""
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)
        z = F.pad(x, (self.pad, 0))
        z = self.dropout(torch.relu(self.conv1(z)))
        z = F.pad(z, (self.pad, 0))
        z = self.dropout(torch.relu(self.conv2(z)))
        return torch.relu(z + residual)

class GyroTCN(nn.Module):
    """The global heading estimator"""
    def __init__(self, in_features=3, channels=48, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            CausalBlock(in_features, channels, dilation=1, dropout=dropout),
            CausalBlock(channels, channels, dilation=2, dropout=dropout),
            CausalBlock(channels, channels, dilation=4, dropout=dropout),
            CausalBlock(channels, channels, dilation=8, dropout=dropout),
        )
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # [batch, time, features] -> [batch, features, time]
        z = self.blocks(x.transpose(1, 2))
        return self.head(z).squeeze(1)
