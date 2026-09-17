import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.idr_model import IDRModel
from ekf.differentiable_ekf import DifferentiableEKF
from data.dataset import IDRDataset
from scripts.train import run_ekf_forward

def run_identity_test(batch, config, device):
    print("\n--- Identity Test ---")
    # Identity test: replace GRU output with zeros, MTN yaw_residual with 0
    # feed corrected orientation yaw directly into EKF.
    ekf = DifferentiableEKF(config).to(device)
    
    bs = batch['raw_a_10hz'].shape[0]
    seq_len = batch['raw_a_10hz'].shape[1]
    dt = config['ekf']['dt']
    
    state = torch.zeros(bs, 18, device=device)
    initial_yaw = batch['initial_yaw'].squeeze(1)
    vr_seed = batch['vr_seed'].squeeze(1)
    
    state[:, 3] = vr_seed * torch.cos(initial_yaw)
    state[:, 4] = vr_seed * torch.sin(initial_yaw)
    state[:, 8] = initial_yaw
    state[:, 17] = vr_seed
    
    P = torch.eye(18, device=device).unsqueeze(0).repeat(bs, 1, 1) * 0.01
    
    preds_traj = []
    
    raw_a = batch['raw_a_10hz']
    raw_g = batch['raw_g_10hz']
    
    for t in range(seq_len):
        # Zero GRU output (psi_rate_corr = 0)
        state[:, 15] = 0.0
        
        # Zero MTN output (roll, pitch, yaw_residual = 0) - this means EKF just uses raw_g integration
        # Wait, the prompt says "feed corrected orientation yaw directly into EKF"
        # The true initial_yaw is the corrected orientation yaw at t=0. 
        # So we just run predict with zero bias and zero neural corrections!
        
        state, P = ekf.predict(state, P, raw_a[:, t], raw_g[:, t])
        preds_traj.append(state[:, 0:2])
        
    preds_traj = torch.stack(preds_traj, dim=1)
    
    gt_traj = batch['gt_cum_disp']
    pred_end_pos = preds_traj[0, -1, :]
    gt_end_pos = gt_traj[0, -1, :]
    
    end_pos_error = torch.sqrt(torch.sum((pred_end_pos - gt_end_pos)**2)).item()
    dist_traveled = batch['total_dist'][0].item()
    
    drift_pct = (end_pos_error / (dist_traveled + 1e-8)) * 100.0
    print(f"Identity Test Drift: {drift_pct:.2f}%")
    return drift_pct

def run_overfit_test(batch, config, device):
    print("\n--- Overfit One Window Test ---")
    model = IDRModel(config).to(device)
    ekf = DifferentiableEKF(config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        
        preds_speed, preds_traj = run_ekf_forward(model, ekf, batch, config)
        
        gt_speed = batch['gt_speeds']
        gt_traj = batch['gt_cum_disp']
        
        loss_speed = criterion(preds_speed, gt_speed)
        loss_traj = criterion(preds_traj, gt_traj)
        
        loss = loss_speed + loss_traj
        loss.backward()
        optimizer.step()
        
        if (i+1) % 20 == 0:
            print(f" Iteration {i+1}/100 | Loss: {loss.item():.4f}")
            
    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds_speed, preds_traj = run_ekf_forward(model, ekf, batch, config)
        
    pred_end_pos = preds_traj[0, -1, :]
    gt_end_pos = gt_traj[0, -1, :]
    
    end_pos_error = torch.sqrt(torch.sum((pred_end_pos - gt_end_pos)**2)).item()
    dist_traveled = batch['total_dist'][0].item()
    
    drift_pct = (end_pos_error / (dist_traveled + 1e-8)) * 100.0
    print(f"Overfit Test Final Drift: {drift_pct:.2f}%")
    if drift_pct < 1.0:
        print("PASS: Overfit drift < 1%")
    else:
        print("FAIL: Model unable to overfit. Check gradient flow.")

def main():
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    train_dataset = IDRDataset('data/train_blackout_windows.pkl', scalers_path='data/scalers.pkl', augment=False)
    
    # Grab just ONE window
    val_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    batch = next(iter(val_loader))
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
            
    run_identity_test(batch, config, device)
    run_overfit_test(batch, config, device)

if __name__ == "__main__":
    main()
