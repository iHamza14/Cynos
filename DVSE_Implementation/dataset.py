import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast

class DVSEDataset(Dataset):
    def __init__(self, csv_path, seq_len=10):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        
        for col in ['acc_feat', 'gyro_feat', 'mtn_in', 'raw_acc', 'grav']:
            self.df[col] = self.df[col].apply(ast.literal_eval)
            
    def __len__(self):
        return len(self.df) - self.seq_len
        
    def __getitem__(self, idx):
        window = self.df.iloc[idx : idx + self.seq_len]
        
        acc_feat = torch.tensor(np.vstack(window['acc_feat'].values), dtype=torch.float32)
        gyro_feat = torch.tensor(np.vstack(window['gyro_feat'].values), dtype=torch.float32)
        mtn_in = torch.tensor(np.vstack(window['mtn_in'].values), dtype=torch.float32)
        raw_acc = torch.tensor(np.vstack(window['raw_acc'].values), dtype=torch.float32)
        grav = torch.tensor(np.vstack(window['grav'].values), dtype=torch.float32)
        
        target_delta_v = torch.tensor(window['delta_v_target'].values, dtype=torch.float32).unsqueeze(-1)
        target_v = torch.tensor(window['gps_speed'].values, dtype=torch.float32).unsqueeze(-1)
        
        vr = torch.zeros_like(target_v)
        vr[0] = target_v[0]
        vr[1:] = target_v[:-1]
        
        return acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_delta_v, target_v
