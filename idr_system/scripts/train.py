import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.idr_model import IDRModel
from ekf.differentiable_ekf import DifferentiableEKF
from data.dataset import IDRDataset
from models.rotation_utils import euler_to_rotation_matrix, apply_rotation

def upsample_1hz_to_10hz(tensor, target_len):
    """Upsamples (Batch, Steps, Dim) from 1Hz to 10Hz."""
    # Repeat each element 10 times along the sequence dimension
    tensor_10hz = torch.repeat_interleave(tensor, repeats=10, dim=1)
    # Trim or pad to match exact target_len
    if tensor_10hz.shape[1] > target_len:
        tensor_10hz = tensor_10hz[:, :target_len]
    elif tensor_10hz.shape[1] < target_len:
        pad_len = target_len - tensor_10hz.shape[1]
        pad_tensor = tensor_10hz[:, -1:].repeat(1, pad_len, 1) if tensor_10hz.dim() == 3 else tensor_10hz[:, -1:].repeat(1, pad_len)
        tensor_10hz = torch.cat([tensor_10hz, pad_tensor], dim=1)
    return tensor_10hz

def run_ekf_forward(model, ekf, batch, config):
    device = batch['raw_a_10hz'].device
    bs = batch['raw_a_10hz'].shape[0]
    seq_len = batch['raw_a_10hz'].shape[1]
    dt = config['ekf']['dt']
    
    # Run Neural Nets
    net_out = model(
        batch['mtn_input'],
        batch['accel_feat'],
        batch['gyro_feat'],
        batch['fft_feat'],
        batch['broadcast_feat']
    )
    
    # 1Hz network outputs
    psi_rate_corr = net_out['psi_rate_corr']
    dv = net_out['dv']
    dpsi = net_out['dpsi']
    roll_mtn = net_out['roll']
    pitch_mtn = net_out['pitch']
    yaw_res = net_out['yaw_residual']
    
    # Upsample to 10Hz
    psi_rate_corr_10hz = upsample_1hz_to_10hz(psi_rate_corr, seq_len)
    dv_10hz = upsample_1hz_to_10hz(dv, seq_len)
    dpsi_10hz = upsample_1hz_to_10hz(dpsi, seq_len)
    roll_10hz = upsample_1hz_to_10hz(roll_mtn, seq_len)
    pitch_10hz = upsample_1hz_to_10hz(pitch_mtn, seq_len)
    yaw_res_10hz = upsample_1hz_to_10hz(yaw_res, seq_len)
    
    # Initialize EKF State (18 dims)
    state = torch.zeros(bs, 18, device=device)
    
    initial_yaw = batch['initial_yaw'].squeeze(1)
    vr_seed = batch['vr_seed'].squeeze(1)
    
    # Set initial vx, vy based on vr_seed and initial_yaw
    state[:, 3] = vr_seed * torch.cos(initial_yaw)
    state[:, 4] = vr_seed * torch.sin(initial_yaw)
    state[:, 8] = initial_yaw
    state[:, 17] = vr_seed
    
    P = torch.eye(18, device=device).unsqueeze(0).repeat(bs, 1, 1) * 0.01
    
    # Pre-allocate measurement matrices
    H_yaw = torch.zeros(bs, 1, 18, device=device)
    H_yaw[:, 0, 8] = 1.0
    R_yaw = torch.eye(1, device=device).unsqueeze(0).repeat(bs, 1, 1) * config['ekf']['r_yaw_straight']
    
    H_vel = torch.zeros(bs, 2, 18, device=device)
    H_vel[:, 0, 3] = 1.0
    H_vel[:, 1, 4] = 1.0
    R_vel = torch.eye(2, device=device).unsqueeze(0).repeat(bs, 1, 1) * 0.1
    
    preds_speed = []
    preds_traj = []
    
    raw_a = batch['raw_a_10hz']
    raw_g = batch['raw_g_10hz']
    
    for t in range(seq_len):
        state = state.clone()
        
        # Apply neural corrections to state before predict
        state[:, 6] = roll_10hz[:, t]
        state[:, 7] = pitch_10hz[:, t]
        state[:, 15] = psi_rate_corr_10hz[:, t]
        
        # Predict step
        state, P = ekf.predict(state, P, raw_a[:, t], raw_g[:, t])
        
        # Update step: Yaw soft measurement
        z_yaw = (initial_yaw + yaw_res_10hz[:, t] + dpsi_10hz[:, t]).unsqueeze(1)
        state, P = ekf.update(state, P, z_yaw, H_yaw, R_yaw)
        
        # Update step: TCN velocity measurement
        v_mag = vr_seed + dv_10hz[:, t]
        curr_yaw = state[:, 8]
        z_vel_x = v_mag * torch.cos(curr_yaw)
        z_vel_y = v_mag * torch.sin(curr_yaw)
        z_vel = torch.stack([z_vel_x, z_vel_y], dim=1)
        
        state, P = ekf.update(state, P, z_vel, H_vel, R_vel)
        
        # Save predictions
        vx = state[:, 3]
        vy = state[:, 4]
        vz = state[:, 5]
        speed = torch.sqrt(vx**2 + vy**2 + vz**2 + 1e-8)
        
        preds_speed.append(speed)
        preds_traj.append(state[:, 0:2])
        
    preds_speed = torch.stack(preds_speed, dim=1) # (B, seq_len)
    preds_traj = torch.stack(preds_traj, dim=1)   # (B, seq_len, 2)
    
    return preds_speed, preds_traj

def train():
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    train_cfg = config['train']
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Datasets
    train_dataset = IDRDataset('data/train_blackout_windows.pkl', scalers_path='data/scalers.pkl', augment=True)
    val_dataset = IDRDataset('data/test_blackout_windows.pkl', scalers_path='data/scalers.pkl', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = IDRModel(config).to(device)
    ekf = DifferentiableEKF(config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    criterion = nn.SmoothL1Loss(beta=train_cfg['smooth_l1_beta'])
    
    best_drift = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(train_cfg['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    
            optimizer.zero_grad()
            
            preds_speed, preds_traj = run_ekf_forward(model, ekf, batch, config)
            
            gt_speed = batch['gt_speeds']
            gt_traj = batch['gt_cum_disp']
            
            loss_speed = criterion(preds_speed, gt_speed)
            loss_traj = criterion(preds_traj, gt_traj)
            
            loss = loss_speed + loss_traj
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        total_drift_pct = 0.0
        n_windows = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                        
                preds_speed, preds_traj = run_ekf_forward(model, ekf, batch, config)
                
                gt_speed = batch['gt_speeds']
                gt_traj = batch['gt_cum_disp']
                
                loss_speed = criterion(preds_speed, gt_speed)
                loss_traj = criterion(preds_traj, gt_traj)
                val_loss += (loss_speed + loss_traj).item()
                
                # Calculate Drift %
                # end_position_error = Euclidean distance between predicted final position and GPS final position
                # total_distance_traveled = cumulative path length during blackout
                
                pred_end_pos = preds_traj[:, -1, :]
                gt_end_pos = gt_traj[:, -1, :]
                
                end_pos_error = torch.sqrt(torch.sum((pred_end_pos - gt_end_pos)**2, dim=1))
                dist_traveled = batch['total_dist']
                
                # To handle edge cases where dist=0 (should be rejected by MIN_SPEED)
                drift_pct = (end_pos_error / (dist_traveled + 1e-8)) * 100.0
                
                total_drift_pct += torch.sum(drift_pct).item()
                n_windows += drift_pct.shape[0]
                
        val_loss /= len(val_loader)
        avg_drift = total_drift_pct / n_windows
        
        print(f"Epoch {epoch+1:03d}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Drift: {avg_drift:.2f}%")
        
        if avg_drift < best_drift:
            best_drift = avg_drift
            patience_counter = 0
            torch.save(model.state_dict(), 'data/best_model.pth')
            print("  -> Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['early_stop_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train()
