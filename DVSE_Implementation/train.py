import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import DVSE
from dataset import DVSEDataset
import os

def loss_matching(pred_dv, target_dv, pred_v, target_v, criterion):
    # Normal alignment
    L1_dv = criterion(pred_dv[:, 1:], target_dv[:, 1:])
    L1_v = criterion(pred_v[:, 1:], target_v[:, 1:])
    L1 = 0.7 * L1_dv + 0.3 * L1_v
    
    # Shifted alignment (prediction shifted back 1 step relative to target)
    L2_dv = criterion(pred_dv[:, :-1], target_dv[:, 1:])
    L2_v = criterion(pred_v[:, :-1], target_v[:, 1:])
    L2 = 0.7 * L2_dv + 0.3 * L2_v
    
    return torch.min(L1, L2).mean()

def train():
    train_path = 'data/train_windows.csv'
    test_path = 'data/test_windows.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data not found. Please run data_sync.py first.")
        return

    train_dataset = DVSEDataset(train_path, seq_len=10)
    test_dataset = DVSEDataset(test_path, seq_len=10)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = DVSE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(reduction='none')
    
    epochs = 20
    print(f"Starting Training: {len(train_dataset)} train sequences, {len(test_dataset)} test sequences...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_dv, target_v = batch
            
            optimizer.zero_grad()
            pred_dv, euler = model(acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav)
            
            pred_v = torch.zeros_like(pred_dv)
            pred_v[:, 0] = target_v[:, 0]
            for t in range(1, pred_dv.size(1)):
                pred_v[:, t] = pred_v[:, t-1] + pred_dv[:, t]
                
            loss = loss_matching(pred_dv, target_dv, pred_v, target_v, criterion)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_dv, target_v = batch
                pred_dv, euler = model(acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav)
                
                pred_v = torch.zeros_like(pred_dv)
                pred_v[:, 0] = target_v[:, 0]
                for t in range(1, pred_dv.size(1)):
                    pred_v[:, t] = pred_v[:, t-1] + pred_dv[:, t]
                    
                loss = loss_matching(pred_dv, target_dv, pred_v, target_v, criterion)
                test_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
        
    torch.save(model.state_dict(), 'dvse_model.pth')
    print("Training complete, model saved to dvse_model.pth.")

if __name__ == '__main__':
    train()
