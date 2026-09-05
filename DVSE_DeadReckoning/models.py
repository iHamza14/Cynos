import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x is (B, C, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x # (B, out_channels, T)

class NoiseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_acc = nn.Linear(18, 64)
        self.fc_gyro = nn.Linear(18, 64)
        self.fc_vr = nn.Linear(1, 16)
        
        self.gru = nn.GRU(input_size=144, hidden_size=128, batch_first=True)
        self.fc_out = nn.Linear(128, 1)
        
    def forward(self, acc, gyro, vr):
        # acc/gyro: (B, T, 18), vr: (B, T, 1)
        a = F.relu(self.fc_acc(acc))
        g = F.relu(self.fc_gyro(gyro))
        v = F.relu(self.fc_vr(vr))
        
        merged = torch.cat([a, g, v], dim=-1) # (B, T, 144)
        out, _ = self.gru(merged)
        noise = self.fc_out(out) # (B, T, 1)
        return noise

def euler_to_rotation_matrix(euler):
    B, T, _ = euler.shape
    alpha = euler[..., 0]
    beta = euler[..., 1]
    gamma = euler[..., 2]
    
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    
    R = torch.zeros((B, T, 3, 3), device=euler.device)
    
    R[..., 0, 0] = ca * cb
    R[..., 0, 1] = ca * sb * sg - sa * cg
    R[..., 0, 2] = ca * sb * cg + sa * sg
    
    R[..., 1, 0] = sa * cb
    R[..., 1, 1] = sa * sb * sg + ca * cg
    R[..., 1, 2] = sa * sb * cg - ca * sg
    
    R[..., 2, 0] = -sb
    R[..., 2, 1] = cb * sg
    R[..., 2, 2] = cb * cg
    
    return R

class DVSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_net = NoiseNetwork()
        self.mtn = TCN(in_channels=6, out_channels=3) # Outputs Euler angles (3)
        
    def forward(self, acc_feat, gyro_feat, vr, mtn_input, raw_accel, gravity):
        """
        acc_feat: (B, T, 18)
        gyro_feat: (B, T, 18)
        vr: (B, T, 1)
        mtn_input: (B, T, 6)
        raw_accel: (B, T, 3)
        gravity: (B, T, 3)
        """
        # Noise (B, T, 1)
        noise = self.noise_net(acc_feat, gyro_feat, vr)
        
        # MTN expects (B, C, T)
        mtn_in_t = mtn_input.permute(0, 2, 1)
        euler = self.mtn(mtn_in_t) # (B, 3, T)
        euler = euler.permute(0, 2, 1) # (B, T, 3)
        
        # Physics Integration
        R = euler_to_rotation_matrix(euler) # (B, T, 3, 3)
        
        # a_p - g_p
        a_minus_g = raw_accel - gravity # (B, T, 3)
        a_minus_g = a_minus_g.unsqueeze(-1) # (B, T, 3, 1)
        
        # Rotate to vehicle frame
        a_v = torch.matmul(R, a_minus_g).squeeze(-1) # (B, T, 3)
        
        # Forward acceleration (Y axis)
        a_fwd = a_v[..., 1].unsqueeze(-1) # (B, T, 1)
        
        # Integrate (dt=1)
        v_physics = a_fwd * 1.0 
        
        # Final Delta V = physics - noise
        delta_v = v_physics - noise
        
        return delta_v, euler
