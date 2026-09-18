import torch
import torch.nn as nn
from .dvse_components import NCN, MTN
from .dvse_features import extract_1sec_features, preintegrate_acceleration
from .dvse_physics import euler_to_rotation_matrix, physics_velocity_update

class DVSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ncn = NCN()
        self.mtn = MTN()

    def forward(self, acc_window, gyro_window, v_r, v_0=None, hz=10):
        """
        Args:
            acc_window: [B, 100, 3] raw acceleration
            gyro_window: [B, 100, 3] raw gyro
            v_r: [B, 10, 1] reference velocity sequence
            v_0: [B, 1] Initial velocity at the start of the 10-second window.
            hz: Sampling rate.
            
        Returns:
            delta_v: [B, 10] total 1-second velocity increments
            v: [B, 10] integrated velocity
            N: [B, 10, 1] learned disturbance
            angles: [B, 10, 3] Euler angles (alpha, beta, gamma)
        """
        # Feature Extraction
        acc_feat = extract_1sec_features(acc_window, hz=hz) # [B, 10, 18]
        gyro_feat = extract_1sec_features(gyro_window, hz=hz) # [B, 10, 18]
        
        # NCN
        N = self.ncn(acc_feat, gyro_feat, v_r) # [B, 10, 1]
        
        # MTN
        I = preintegrate_acceleration(acc_window, hz=hz) # [B, 10, 3]
        g_w = torch.tensor([0.0, 0.0, 9.81], dtype=I.dtype, device=I.device)
        g_w_expand = g_w.view(1, 1, 3).expand(I.shape[0], I.shape[1], 3)
        i_feat = torch.cat([I, g_w_expand], dim=-1) # [B, 10, 6]
        
        angles = self.mtn(i_feat) # [B, 10, 3]
        
        # Physics
        R_vp = euler_to_rotation_matrix(angles) # [B, 10, 3, 3]
        delta_v, v, delta_v_physics = physics_velocity_update(
            acc_window=acc_window, 
            R_vp=R_vp, 
            N=N, 
            v_0=v_0, 
            hz=hz
        )
        
        return delta_v, v, N, angles
