import torch
import torch.nn as nn

class NoiseCompensationNet(nn.Module):
    def __init__(self, input_dim=147, hidden_dim=128):
        super(NoiseCompensationNet, self).__init__()
        
        # Accel branch (18 features)
        self.accel_mlp = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Gyro branch (18 features)
        self.gyro_mlp = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # FFT branch (15 features)
        self.fft_mlp = nn.Sequential(
            nn.Linear(15, 16),
            nn.ReLU()
        )
        
        # GRU
        self.gru = nn.GRU(input_size=147, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Shared trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Heads
        self.head_dv = nn.Linear(16, 1)            # Head 1 (Δv)
        self.head_dpsi = nn.Linear(16, 1)          # Head 2 (Δψ)
        self.head_psi_rate_corr = nn.Linear(16, 1) # Head 3 (Δψ_rate_correction)

    def forward(self, accel_feat, gyro_feat, fft_feat, broadcast_feat):
        """
        Args:
            accel_feat: (batch, seq_len, 18)
            gyro_feat: (batch, seq_len, 18)
            fft_feat: (batch, seq_len, 15)
            broadcast_feat: (batch, seq_len, 3) - vr_seed, gps_age, shock_mask
        Returns:
            dv, dpsi, psi_rate_corr
        """
        a_out = self.accel_mlp(accel_feat)
        g_out = self.gyro_mlp(gyro_feat)
        f_out = self.fft_mlp(fft_feat)
        
        # Concatenate features: 64 + 64 + 16 + 3 = 147
        x = torch.cat([a_out, g_out, f_out, broadcast_feat], dim=-1)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Shared trunk
        trunk_out = self.shared_trunk(gru_out)
        
        # Output heads
        dv = self.head_dv(trunk_out).squeeze(-1)
        dpsi = self.head_dpsi(trunk_out).squeeze(-1)
        psi_rate_corr = self.head_psi_rate_corr(trunk_out).squeeze(-1)
        
        return dv, dpsi, psi_rate_corr
