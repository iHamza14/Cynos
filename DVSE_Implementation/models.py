import torch
import torch.nn as nn

class NoiseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.acc_fc1 = nn.Linear(18, 32)
        self.acc_fc2 = nn.Linear(32, 64)
        self.gyro_fc1 = nn.Linear(18, 32)
        self.gyro_fc2 = nn.Linear(32, 64)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, acc_feat, gyro_feat, vr):
        a = torch.relu(self.acc_fc1(acc_feat))
        a = torch.relu(self.acc_fc2(a))
        g = torch.relu(self.gyro_fc1(gyro_feat))
        g = torch.relu(self.gyro_fc2(g))
        x = torch.cat([a, g], dim=-1)
        out, _ = self.gru(x)
        out = torch.cat([out, vr], dim=-1)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        n = self.fc4(out)
        return n

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x.transpose(1, 2)

class MTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 64)
        self.tcn = TCN(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tcn(x)
        euler = self.fc3(x)
        return euler

def euler_to_rotation_matrix(euler_angles):
    B, Seq, _ = euler_angles.shape
    R = torch.zeros(B, Seq, 3, 3, device=euler_angles.device)
    
    alpha = euler_angles[..., 0]
    beta = euler_angles[..., 1]
    gamma = euler_angles[..., 2]
    
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    
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
        self.mtn = MTN()

    def forward(self, acc_feat, gyro_feat, vr, mtn_input, raw_accel, gravity):
        N = self.noise_net(acc_feat, gyro_feat, vr)
        euler = self.mtn(mtn_input)
        R_pv = euler_to_rotation_matrix(euler)
        
        a_diff = (raw_accel - gravity).unsqueeze(-1)
        a_v = torch.matmul(R_pv, a_diff).squeeze(-1)
        a_forward = a_v[..., 1:2]
        
        delta_v = a_forward + N
        return delta_v, euler
