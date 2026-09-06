import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import NoiseNetwork
import os

def sliding_loss(pred_dv, target_dv):
    # Offset 0
    l_0_dv = F.smooth_l1_loss(pred_dv, target_dv)
    l_0_cum = F.smooth_l1_loss(torch.cumsum(pred_dv, dim=1), torch.cumsum(target_dv, dim=1))
    l_0 = 0.7 * l_0_dv + 0.3 * l_0_cum
    
    # Offset 1 (IMU leads GPS by 1 second)
    if pred_dv.size(1) > 1:
        pred_shifted = pred_dv[:, 1:]
        target_shifted = target_dv[:, :-1]
        l_1_dv = F.smooth_l1_loss(pred_shifted, target_shifted)
        l_1_cum = F.smooth_l1_loss(torch.cumsum(pred_shifted, dim=1), torch.cumsum(target_shifted, dim=1))
        l_1 = 0.7 * l_1_dv + 0.3 * l_1_cum
        
        return torch.min(l_0, l_1)
    return l_0

def train_stage1():
    os.makedirs('checkpoints', exist_ok=True)
    
    scaler_path = 'data/scalers.pkl'
    train_dataset = DeadReckoningDataset('data/train_blackout_windows.pkl', scaler_path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    model = NoiseNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 40
    print("=== Starting Stage 1: Noise Network (GRU) ===")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            (acc_t, gyro_t, mtn_t, raw_t, grav_t, 
             vr_train, vr_seed, delta_v_seq, target_disp, 
             heading_target, raw_gyro, start_heading) = batch
            
            # Prevent data leakage: The GRU must receive the PREVIOUS step's velocity to predict the current delta_v
            # vr_input[0] = vr_seed
            # vr_input[t] = vr_train[t-1]
            vr_input = torch.zeros_like(vr_train)
            vr_input[:, 0, :] = vr_seed
            if vr_train.size(1) > 1:
                vr_input[:, 1:, :] = vr_train[:, :-1, :]
            
            # Augmentation: Add Gaussian Noise (sigma=0.5 m/s) to the shifted vr sequence
            vr_noisy = vr_input + torch.randn_like(vr_input) * 0.5
            
            optimizer.zero_grad()
            delta_v_pred = model(acc_t, gyro_t, vr_noisy)
            
            loss = sliding_loss(delta_v_pred, delta_v_seq)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Stage 1 Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'checkpoints/stage1_noise_net.pth')
    print("Stage 1 Complete. Saved checkpoints/stage1_noise_net.pth")

if __name__ == '__main__':
    train_stage1()
