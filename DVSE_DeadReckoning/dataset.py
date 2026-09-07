import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class DeadReckoningDataset(Dataset):
    def __init__(self, pkl_path, scaler_path):
        with open(pkl_path, 'rb') as f:
            self.windows = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        self.raw_accel_scaler = scaler['raw_accel']
        self.raw_gyro_scaler = scaler['raw_gyro']
        self.gravity_scaler = scaler['gravity']

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        
        raw_accel = self.raw_accel_scaler.transform(w['blackout']['raw_accel'])
        raw_gyro = self.raw_gyro_scaler.transform(w['blackout']['raw_gyro'])
        gravity = self.gravity_scaler.transform(w['blackout']['gravity'])
        
        raw_accel_t = torch.FloatTensor(raw_accel)
        raw_gyro_t = torch.FloatTensor(raw_gyro)
        gravity_t = torch.FloatTensor(gravity)
        
        vr_train = torch.FloatTensor(w['ground_truth']['speeds_ms']).unsqueeze(1)
        delta_v_seq = torch.FloatTensor(w['ground_truth']['delta_v']).unsqueeze(1)
        target_disp = torch.FloatTensor(w['ground_truth']['cum_disp_m'])
        heading_target = torch.FloatTensor(w['ground_truth']['headings_rad'])
        
        vr_seed = torch.FloatTensor([w['context']['vr_seed']])
        start_heading = torch.FloatTensor([w['ground_truth']['headings_rad'][0]])
        
        return (raw_accel_t, raw_gyro_t, gravity_t, 
                vr_train, vr_seed, delta_v_seq, target_disp, heading_target, start_heading)