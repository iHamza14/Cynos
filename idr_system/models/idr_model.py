import torch
import torch.nn as nn
from models.mtn import MTN
from models.noise_net import NoiseCompensationNet

class IDRModel(nn.Module):
    def __init__(self, mtn_config=None, noise_net_config=None):
        super(IDRModel, self).__init__()
        
        self.mtn = MTN()
        self.noise_net = NoiseCompensationNet()
        
    def forward(self, mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat):
        # MTN forward
        roll, pitch, yaw_residual = self.mtn(mtn_input)
        
        # Noise net forward
        dv, dpsi, psi_rate_corr = self.noise_net(accel_feat, gyro_feat, fft_feat, broadcast_feat)
        
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw_residual': yaw_residual,
            'dv': dv,
            'dpsi': dpsi,
            'psi_rate_corr': psi_rate_corr
        }
