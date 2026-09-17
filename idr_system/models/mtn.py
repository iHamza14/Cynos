import torch
import torch.nn as nn
from models.temporal_block import TemporalBlock

class MTNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, kernel_size=3, dropout=0.2):
        super(MTNBranch, self).__init__()
        layers = []
        num_levels = len(dilations)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            padding = (kernel_size - 1) * dilation_size
            
            # According to prompt: each block could be multiple convs, but typical TCN has 2 convs per block.
            # We'll adapt TemporalBlock to take dilation and pad.
            # We use a simplified version here wrapping the provided TemporalBlock.
            # The TemporalBlock as defined has 3 convs with the same dilation.
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout
            ))
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MTN(nn.Module):
    def __init__(self, input_channels=7, hidden_dim=64, dropout=0.2):
        super(MTN, self).__init__()
        
        self.fc1 = nn.Linear(input_channels, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, hidden_dim)
        self.relu2 = nn.ReLU()
        
        # Parallel TCN branches
        # Note: MTNBranch expects input shape (batch, channels, seq_len)
        self.branch_a = MTNBranch(hidden_dim, hidden_dim, dilations=[1, 2, 4], kernel_size=3, dropout=dropout)
        self.branch_b = MTNBranch(hidden_dim, hidden_dim, dilations=[4, 8, 16, 32], kernel_size=3, dropout=dropout)
        
        self.fc3 = nn.Linear(hidden_dim * 2, 64)
        self.relu3 = nn.ReLU()
        
        self.yaw_head = nn.Linear(64, 1)
        self.pitch_roll_head = nn.Linear(64, 2)
        
        # Initialize yaw residual to near 0
        nn.init.normal_(self.yaw_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.yaw_head.bias, 0.0)

    def forward(self, x):
        # x shape: (batch, seq_len, input_channels)
        batch_size, seq_len, _ = x.shape
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        # Convert to (batch, channels, seq_len) for TCN Conv1d
        x_t = x.transpose(1, 2)
        
        out_a = self.branch_a(x_t)
        out_b = self.branch_b(x_t)
        
        # Concatenate along channel axis
        out = torch.cat([out_a, out_b], dim=1)
        
        # Convert back to (batch, seq_len, channels)
        out = out.transpose(1, 2)
        
        out = self.relu3(self.fc3(out))
        
        yaw_res = self.yaw_head(out)
        pitch_roll = self.pitch_roll_head(out)
        
        # return (roll, pitch, yaw_residual)
        # pitch_roll has shape (batch, seq_len, 2)
        # yaw_res has shape (batch, seq_len, 1)
        roll = pitch_roll[:, :, 0]
        pitch = pitch_roll[:, :, 1]
        yaw_residual = yaw_res[:, :, 0]
        
        return roll, pitch, yaw_residual
