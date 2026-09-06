import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

class DeadReckoningDataset(Dataset):
    def __init__(self, pkl_path, scaler_path):
        with open(pkl_path, 'rb') as f:
            self.windows = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        self.acc_scaler = scaler['acc']
        self.gyro_scaler = scaler['gyro']
        self.mtn_scaler = scaler['mtn']
        self.raw_scaler = scaler['raw']
        self.grav_scaler = scaler['grav']

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        
        # Scale inputs
        acc_feat = self.acc_scaler.transform(w['acc_feat'])
        gyro_feat = self.gyro_scaler.transform(w['gyro_feat'])
        mtn_input = self.mtn_scaler.transform(w['mtn_input'])
        raw_accel = self.raw_scaler.transform(w['raw_accel'])
        gravity = self.grav_scaler.transform(w['gravity'])
        
        # Convert to tensors
        acc_t = torch.FloatTensor(acc_feat)
        gyro_t = torch.FloatTensor(gyro_feat)
        mtn_t = torch.FloatTensor(mtn_input)
        raw_t = torch.FloatTensor(raw_accel)
        grav_t = torch.FloatTensor(gravity)
        
        vr_train = torch.FloatTensor(w['vr_train'])
        delta_v_seq = torch.FloatTensor(w['delta_v_seq'])
        target_disp = torch.FloatTensor(w['target_disp'])
        heading_target = torch.FloatTensor(w['heading_target_rad'])
        
        vr_seed = torch.FloatTensor([w['vr_seed']])
        
        return (acc_t, gyro_t, mtn_t, raw_t, grav_t, 
                vr_train, vr_seed, delta_v_seq, target_disp, heading_target)
