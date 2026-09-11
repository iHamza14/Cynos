import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch.nn.functional as F

# ---------------------------
# 1. Dataset & Preprocessing
# ---------------------------
class BlackoutDataset(Dataset):
    def __init__(self, data_path, scalers_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        window = self.data[idx]
        
        # Unpack components
        context = window['context']
        blackout = window['blackout']
        ground_truth = window['ground_truth']
        
        # Standardize using pre-fit scalers
        # Assuming shape (SeqLen, Channels)
        ctx_accel = self.scalers['raw_accel'].transform(context['raw_accel'])
        ctx_gyro = self.scalers['raw_gyro'].transform(context['raw_gyro'])
        blk_accel = self.scalers['raw_accel'].transform(blackout['raw_accel'])
        blk_gyro = self.scalers['raw_gyro'].transform(blackout['raw_gyro'])
        
        return {
            # Concatenate Normalized IMU for TCN
            'context_norm_imu': torch.tensor(np.concatenate([ctx_accel, ctx_gyro], axis=-1), dtype=torch.float32),
            'blackout_norm_imu': torch.tensor(np.concatenate([blk_accel, blk_gyro], axis=-1), dtype=torch.float32),
            
            # Raw unnormalized sequences for EKF Physics
            'context_raw_accel': torch.tensor(context['raw_accel'], dtype=torch.float32),
            'context_raw_gyro': torch.tensor(context['raw_gyro'], dtype=torch.float32),
            'blackout_raw_accel': torch.tensor(blackout['raw_accel'], dtype=torch.float32),
            'blackout_raw_gyro': torch.tensor(blackout['raw_gyro'], dtype=torch.float32),
            
            # Seed scalar / vector
            'vr_seed': torch.tensor(context['vr_seed'], dtype=torch.float32),
            
            # Ground truth targets for EKF outputs
            'speeds_ms': torch.tensor(ground_truth['speeds_ms'], dtype=torch.float32),
            'headings_rad': torch.tensor(ground_truth['headings_rad'], dtype=torch.float32),
            'cum_disp_m': torch.tensor(ground_truth['cum_disp_m'], dtype=torch.float32) # (N, 2)
        }

# ---------------------------
# 2. TCN Architecture
# ---------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size=6, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Causal padding = (kernel_size - 1) * dilation
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
            
        self.network = nn.Sequential(*layers)
        
        # Outputs at each timestep
        self.vel_out = nn.Conv1d(num_channels[-1], 3, 1) # 3D Velocity Pseudo-Measurement
        self.r_scale_out = nn.Conv1d(num_channels[-1], 3, 1) # 3x Scaling factors for Measurement Noise

    def forward(self, x):
        # x is expected to be (Batch, Channels, SeqLen)
        features = self.network(x)
        pseudo_vel = self.vel_out(features)
        
        # Softplus to ensure strictly positive variance multipliers
        r_scale = F.softplus(self.r_scale_out(features)) + 1e-4 
        return pseudo_vel, r_scale

# ---------------------------
# 3. Differentiable EKF
# ---------------------------
def rot_matrix_from_euler(roll, pitch, yaw):
    """ Differentiable Batch Rotation Matrix (Z-Y-X formulation) """
    B = roll.size(0)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    
    R = torch.zeros(B, 3, 3, device=roll.device)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R

class DifferentiableEKF(nn.Module):
    def __init__(self, static_x, static_P, process_noise_std=1e-3, base_measurement_noise_std=0.1):
        super().__init__()
        self.dt = 0.1  # 10 Hz
        
        # Define process noise covariance (Q) and baseline measurement noise (R)
        self.register_buffer('Q', torch.eye(15) * (process_noise_std ** 2))
        self.register_buffer('base_R', torch.eye(3) * (base_measurement_noise_std ** 2))
        
        # Load initialized parameters
        self.register_buffer('static_x', torch.tensor(static_x, dtype=torch.float32))
        self.register_buffer('static_P', torch.tensor(static_P, dtype=torch.float32))
        
        # Measurement matrix H maps 15 states to 3 Velocity states (indices 3, 4, 5)
        H = torch.zeros(3, 15)
        H[0, 3], H[1, 4], H[2, 5] = 1.0, 1.0, 1.0
        self.register_buffer('H', H)

    def predict_step(self, x, P, accel, gyro):
        B = x.size(0)
        
        # Unpack State
        pos = x[:, 0:3]
        vel = x[:, 3:6]
        roll, pitch, yaw = x[:, 6], x[:, 7], x[:, 8]
        b_a = x[:, 9:12]
        b_g = x[:, 12:15]
        
        # Subtract biases
        a_corr = accel - b_a
        g_corr = gyro - b_g
        
        # Update orientation
        new_roll = roll + g_corr[:, 0] * self.dt
        new_pitch = pitch + g_corr[:, 1] * self.dt
        new_yaw = yaw + g_corr[:, 2] * self.dt
        
        # Transform acceleration to global frame
        R_mat = rot_matrix_from_euler(roll, pitch, yaw)
        gravity = torch.zeros(B, 3, device=x.device)
        gravity[:, 2] = 9.80665
        
        # a_global = R * a_body - g
        a_global = torch.bmm(R_mat, a_corr.unsqueeze(2)).squeeze(2) - gravity
        
        # Update kinematic state
        new_vel = vel + a_global * self.dt
        new_pos = pos + vel * self.dt + 0.5 * a_global * (self.dt ** 2)
        
        # Formulate new state vector
        new_x = torch.cat([new_pos, new_vel, new_roll.unsqueeze(1), new_pitch.unsqueeze(1), new_yaw.unsqueeze(1), b_a, b_g], dim=1)
        
        # State Jacobian F
        # Using approximated identity mapping with velocity-position propagation for stability
        F = torch.eye(15, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        F[:, 0, 3], F[:, 1, 4], F[:, 2, 5] = self.dt, self.dt, self.dt
        
        # P = F * P * F^T + Q
        new_P = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + self.Q.unsqueeze(0)
        return new_x, new_P

    def update_step(self, x, P, z, r_scale):
        B = x.size(0)
        
        # Dynamic Covariance Matrix
        # Element-wise scaling on the diagonal Base R matrix
        R_dynamic = self.base_R.unsqueeze(0) * r_scale.unsqueeze(2) 
        
        H = self.H.unsqueeze(0).repeat(B, 1, 1)
        H_t = H.transpose(1, 2)
        
        # Innovation y = z - Hx
        y = z - torch.bmm(H, x.unsqueeze(2)).squeeze(2)
        
        # Innovation Covariance S = HPH^T + R
        S = torch.bmm(torch.bmm(H, P), H_t) + R_dynamic
        
        # Kalman Gain K = PH^TS^{-1}
        S_inv = torch.linalg.inv(S)
        K = torch.bmm(torch.bmm(P, H_t), S_inv)
        
        # A posteriori state update
        new_x = x + torch.bmm(K, y.unsqueeze(2)).squeeze(2)
        
        # A posteriori covariance update: P = (I - KH)P
        I = torch.eye(15, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        new_P = torch.bmm(I - torch.bmm(K, H), P)
        
        return new_x, new_P

    def forward(self, vr_seed, ctx_accel, ctx_gyro, blk_accel, blk_gyro, tcn_pseudo_vel, tcn_r_scale):
        B = vr_seed.size(0)
        
        x = self.static_x.unsqueeze(0).repeat(B, 1)
        P = self.static_P.unsqueeze(0).repeat(B, 1, 1)
        
        # Setup initial speed logic
        # Initialize X-velocity with the scalar seed speed
        x[:, 3] = vr_seed.squeeze()
        
        # Phase 1: Context Warmup (Predict Only)
        ctx_len = ctx_accel.size(1)
        for t in range(ctx_len):
            x, P = self.predict_step(x, P, ctx_accel[:, t, :], ctx_gyro[:, t, :])
            
        # Phase 2: Blackout Rollout (Predict + Update via TCN)
        blk_len = blk_accel.size(1)
        pred_traj = []
        pred_speeds = []
        
        # Ensure TCN tensors align by taking the transpose for step-wise extraction
        # pseudo_vel shape from TCN: (B, 3, L). Change to (B, L, 3)
        tcn_pseudo_vel = tcn_pseudo_vel.transpose(1, 2)
        tcn_r_scale = tcn_r_scale.transpose(1, 2)
        
        for t in range(blk_len):
            # EKF Predict Physics
            x, P = self.predict_step(x, P, blk_accel[:, t, :], blk_gyro[:, t, :])
            
            # EKF Update using Neural Measurement
            z = tcn_pseudo_vel[:, t, :]
            r = tcn_r_scale[:, t, :]
            
            x, P = self.update_step(x, P, z, r)
            
            # Track states (2D Cumulative Displacement (X,Y) and Norm Speed)
            pred_traj.append(x[:, 0:2].unsqueeze(1)) 
            pred_speeds.append(x[:, 3:6].norm(dim=1).unsqueeze(1))
            
        return torch.cat(pred_traj, dim=1), torch.cat(pred_speeds, dim=1)

# ---------------------------
# 4. Training Loop
# ---------------------------
def train_ekf_tcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Attempt dynamic initialization using the previous script
    try:
        from ekf_initialization import process_and_initialize_ekf
        static_file = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw01/S-Vw1.csv'
        static_x, static_P, _ = process_and_initialize_ekf(static_file)
        print("Loaded initial states from S-Vw1.csv.")
    except Exception as e:
        print(f"Warning: Falling back to zero-init. Could not load static config: {e}")
        static_x = np.zeros(15)
        static_P = np.eye(15) * 1e-3

    # Instantiate
    tcn = TCN().to(device)
    ekf = DifferentiableEKF(static_x, static_P).to(device)
    
    # Differentiable tracking - optimize TCN parameters via EKF outputs
    optimizer = optim.Adam(tcn.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    import os
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    dataset = BlackoutDataset(os.path.join(data_dir, 'train_blackout_windows.pkl'), 
                              os.path.join(data_dir, 'scalers.pkl'))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 10
    
    print("Setup complete. Starting training loop...")
    
    for epoch in range(epochs):
        tcn.train()
        ekf.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Unpack and push to device
            vr_seed = batch['vr_seed'].to(device)
            ctx_accel = batch['context_raw_accel'].to(device)
            ctx_gyro = batch['context_raw_gyro'].to(device)
            blk_accel = batch['blackout_raw_accel'].to(device)
            blk_gyro = batch['blackout_raw_gyro'].to(device)
            
            # TCN Input formatting: shape (B, C, L)
            blk_norm = batch['blackout_norm_imu'].transpose(1, 2).to(device)
            
            # Ground truth
            gt_disp = batch['cum_disp_m'].to(device)
            gt_speeds = batch['speeds_ms'].to(device)
            
            # 1. TCN yields Pseudo Velocity and Dynamic R scales over blackout
            pseudo_vel, r_scale = tcn(blk_norm)
            
            # 2. Differentiable Rollout
            pred_disp, pred_speeds = ekf(
                vr_seed, ctx_accel, ctx_gyro, 
                blk_accel, blk_gyro, 
                pseudo_vel, r_scale
            )
            
            # 3. Loss Comparison
            loss_disp = mse(pred_disp, gt_disp)
            loss_speed = mse(pred_speeds, gt_speeds)
            loss = loss_disp + loss_speed
            
            # 4. Backpropagate trajectory loss through the unrolled Kalman Filter loops
            loss.backward()
            
            # Apply Gradient Clipping to stop matrix inversion explosion during early epochs
            torch.nn.utils.clip_grad_norm_(tcn.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

if __name__ == '__main__':
    train_ekf_tcn()
