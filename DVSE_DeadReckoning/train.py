import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
import os
import pickle

def loss_fn(delta_v_pred, euler_pred, delta_v_target, vr_seed, target_disp, heading_target_rad, lambda1=1.0, lambda2=0.5, lambda3=0.1):
    # 1. Primary: per-step delta_v regression
    loss_dv = F.mse_loss(delta_v_pred, delta_v_target)
    
    # 2. End-to-end displacement
    heading_rad = euler_pred[..., 2] # (B, T)
    v_cum = vr_seed.unsqueeze(1) + torch.cumsum(delta_v_pred.squeeze(-1), dim=1) # (B, T)
    
    delta_north = (v_cum * torch.cos(heading_rad)).sum(dim=1) # (B,)
    delta_east = (v_cum * torch.sin(heading_rad)).sum(dim=1) # (B,)
    pred_disp = torch.stack([delta_north, delta_east], dim=-1) # (B, 2)
    
    loss_disp = F.mse_loss(pred_disp, target_disp)
    
    # 3. Euler regularization (yaw)
    loss_euler = F.mse_loss(heading_rad, heading_target_rad)
    
    return lambda1 * loss_dv + lambda2 * loss_disp + lambda3 * loss_euler, loss_dv, loss_disp, loss_euler

def train():
    os.makedirs('checkpoints', exist_ok=True)
    
    train_dataset = DeadReckoningDataset('data/train_windows.pkl', is_train=True)
    test_dataset = DeadReckoningDataset('data/test_windows.pkl', scaler=train_dataset.get_scalers(), is_train=False)
    
    # Save scalers
    with open('checkpoints/scalers.pkl', 'wb') as f:
        pickle.dump(train_dataset.get_scalers(), f)
        
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = DVSE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_train, vr_seed, delta_v_seq, target_disp, heading_target = batch
            
            # Add Gaussian Noise (sigma=0.5 m/s) to vr_train for augmentation
            vr_noisy = vr_train + torch.randn_like(vr_train) * 0.5
            
            optimizer.zero_grad()
            delta_v_pred, euler_pred = model(acc_t, gyro_t, vr_noisy, mtn_t, raw_t, grav_t)
            
            loss, l_dv, l_disp, l_eu = loss_fn(
                delta_v_pred, euler_pred, delta_v_seq, vr_seed, 
                target_disp, heading_target
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'checkpoints/dvse_dead_reckoning.pth')
    print("Training Complete. Model saved.")

if __name__ == '__main__':
    train()
