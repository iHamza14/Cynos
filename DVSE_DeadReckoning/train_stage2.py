import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import MTN, DVSE
import os

def angular_loss(pred_yaw, target_yaw):
    # Ensures smooth loss wrapping around 360 degrees (-pi to pi)
    diff = torch.atan2(torch.sin(pred_yaw - target_yaw), torch.cos(pred_yaw - target_yaw))
    return torch.mean(diff ** 2)

def train_stage2():
    os.makedirs('checkpoints', exist_ok=True)
    
    scaler_path = 'data/scalers.pkl'
    train_dataset = DeadReckoningDataset('data/train_blackout_windows.pkl', scaler_path)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    mtn = MTN()
    optimizer = optim.Adam(mtn.parameters(), lr=1e-3)
    
    epochs = 40
    print("=== Starting Stage 2: Motion Transformation Network (TCN) ===")
    
    for epoch in range(epochs):
        mtn.train()
        total_loss = 0.0
        
        for batch in train_loader:
            (acc_t, gyro_t, mtn_t, raw_t, grav_t, 
             vr_train, vr_seed, delta_v_seq, target_disp, 
             heading_target, raw_gyro, start_heading) = batch
            
            optimizer.zero_grad()
            euler_pred = mtn(mtn_t) # (B, T, 3)
            
            # Predict yaw (heading) which is euler[..., 2]
            yaw_pred = euler_pred[..., 2]
            
            # Direct supervision on Vehicle Heading (Ground Truth Pose)
            loss = angular_loss(yaw_pred, heading_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mtn.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Stage 2 Yaw Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(mtn.state_dict(), 'checkpoints/stage2_mtn.pth')
    print("Stage 2 Complete. Saved checkpoints/stage2_mtn.pth")
    
    # Bundle into final DVSE checkpoint for inference scripts
    dvse = DVSE()
    dvse.noise_net.load_state_dict(torch.load('checkpoints/stage1_noise_net.pth', weights_only=True))
    dvse.mtn.load_state_dict(torch.load('checkpoints/stage2_mtn.pth', weights_only=True))
    
    torch.save(dvse.state_dict(), 'checkpoints/dvse_dead_reckoning.pth')
    print("Bundled complete model to checkpoints/dvse_dead_reckoning.pth")

if __name__ == '__main__':
    train_stage2()
