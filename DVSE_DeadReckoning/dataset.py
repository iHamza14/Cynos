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
        
        # Scale inputs (from the blackout section)
        acc_feat = self.acc_scaler.transform(w['blackout']['acc_feat'])
        gyro_feat = self.gyro_scaler.transform(w['blackout']['gyro_feat'])
        mtn_input = self.mtn_scaler.transform(w['blackout']['mtn_input'])
        raw_accel = self.raw_scaler.transform(w['blackout']['raw_accel'])
        gravity = self.grav_scaler.transform(w['blackout']['gravity'])
        
        # Convert to tensors
        acc_t = torch.FloatTensor(acc_feat)
        gyro_t = torch.FloatTensor(gyro_feat)
        mtn_t = torch.FloatTensor(mtn_input)
        raw_t = torch.FloatTensor(raw_accel)
        grav_t = torch.FloatTensor(gravity)
        
        # Target vectors from ground truth
        vr_train = torch.FloatTensor(w['ground_truth']['speeds_ms']).unsqueeze(1)
        delta_v_seq = torch.FloatTensor(w['ground_truth']['delta_v']).unsqueeze(1)
        
        # Disp target (just endpoint for now, or the N-step sequence)
        target_disp = torch.FloatTensor([w['ground_truth']['cum_disp_m'][-1]])
        
        # Heading target
        heading_target = torch.FloatTensor(w['ground_truth']['headings_rad'])
        
        # Initialization seed from context
        vr_seed = torch.FloatTensor([w['context']['vr_seed']])
        
        # Raw Gyro for integration
        raw_gyro = torch.FloatTensor(w['blackout'].get('raw_gyro', np.zeros((len(acc_feat), 3))))
        
        # We also pass the start heading so map_trajectory.py can use it.
        start_heading = torch.FloatTensor([w['ground_truth']['headings_rad'][0]])
        
        return (acc_t, gyro_t, mtn_t, raw_t, grav_t, 
                vr_train, vr_seed, delta_v_seq, target_disp, heading_target, raw_gyro, start_heading)
