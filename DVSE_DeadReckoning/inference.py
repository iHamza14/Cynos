import torch
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
import pickle
import numpy as np
import matplotlib.pyplot as plt

def inference_step(model, acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_seed):
    """
    Autoregressive inference: vr_seed is provided from mobile GPS.
    Subsequent vr values are vr_seed + cumsum(predicted delta_v).
    """
    B, T, _ = acc_t.shape
    delta_v_preds = []
    eulers = []
    
    # Process autoregressively step-by-step
    for t in range(T):
        acc_sub = acc_t[:, :t+1, :]
        gyro_sub = gyro_t[:, :t+1, :]
        mtn_sub = mtn_t[:, :t+1, :]
        raw_sub = raw_t[:, :t+1, :]
        grav_sub = grav_t[:, :t+1, :]
        
        if t == 0:
            vr_seq = vr_seed.unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        else:
            dv_tensor = torch.cat(delta_v_preds, dim=1) # (B, t)
            v_cum = vr_seed.unsqueeze(1) + torch.cumsum(dv_tensor, dim=1) # (B, t)
            vr_seq = torch.cat([vr_seed.unsqueeze(1).unsqueeze(2), v_cum.unsqueeze(-1)], dim=1) # (B, t+1, 1)
            
        # Forward pass up to current step
        dv, eu = model(acc_sub, gyro_sub, vr_seq, mtn_sub, raw_sub, grav_sub)
        
        # Only take the prediction for the final timestep 't'
        delta_v_preds.append(dv[:, -1, :])
        eulers.append(eu[:, -1, :])
        
    delta_v_pred = torch.stack(delta_v_preds, dim=1) # (B, T, 1)
    euler_pred = torch.stack(eulers, dim=1) # (B, T, 3)
    return delta_v_pred, euler_pred

def test():
    with open('checkpoints/scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
        
    test_dataset = DeadReckoningDataset('data/test_windows.pkl', scaler=scalers, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = DVSE()
    model.load_state_dict(torch.load('checkpoints/dvse_dead_reckoning.pth', weights_only=True))
    model.eval()
    
    print("Running Autoregressive Inference on Test Set...")
    total_disp_err = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            acc_t, gyro_t, mtn_t, raw_t, grav_t, _, vr_seed, _, target_disp, _ = batch
            
            # Predict
            delta_v_pred, euler_pred = inference_step(model, acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_seed)
            
            # Reconstruct displacement using predicted yaw
            heading_rad = euler_pred[..., 2]
            v_cum = vr_seed.unsqueeze(1) + torch.cumsum(delta_v_pred.squeeze(-1), dim=1)
            
            delta_north = (v_cum * torch.cos(heading_rad)).sum(dim=1)
            delta_east = (v_cum * torch.sin(heading_rad)).sum(dim=1)
            pred_disp = torch.stack([delta_north, delta_east], dim=-1)
            
            # Error in meters
            err = torch.norm(pred_disp - target_disp, dim=-1).mean().item()
            total_disp_err += err
            
            if i % 10 == 0:
                print(f"Window {i} | Disp Error: {err:.2f} meters")
                
    avg_err = total_disp_err / len(test_loader)
    print(f"\nFinal Test Set Average Haversine Error (60s Windows): {avg_err:.2f} meters")

if __name__ == '__main__':
    test()
