import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from data.augmentation import RotationAugmentation

class IDRDataset(Dataset):
    def __init__(self, pkl_path, scalers_path=None, augment=False):
        with open(pkl_path, 'rb') as f:
            self.windows = pickle.load(f)
            
        self.scalers = None
        if scalers_path is not None:
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
                
        self.augment = augment
        if self.augment:
            self.rot_aug = RotationAugmentation(max_roll=30, max_pitch=30, max_yaw=180)

    def __len__(self):
        return len(self.windows)
        
    def _extract_1hz_features(self, accel, gyro, gravity, yaw, vr_seed, gps_age, shock_mask):
        # Data is at 10Hz. We reshape into 10-step blocks to get 1s windows.
        # Shape: (N, 3) -> (N//10, 10, 3)
        N = accel.shape[0]
        steps = N // 10
        
        # In case N is not perfectly divisible (should be), we truncate
        a_blocks = accel[:steps*10].reshape(steps, 10, 3)
        g_blocks = gyro[:steps*10].reshape(steps, 10, 3)
        grv_blocks = gravity[:steps*10].reshape(steps, 10, 3)
        yaw_blocks = yaw[:steps*10].reshape(steps, 10)
        shock_blocks = shock_mask[:steps*10].reshape(steps, 10)
        
        # -- MTN Features (7 channels) --
        # Pre-integrated accel over 1s (sum * dt)
        mtn_accel = np.sum(a_blocks, axis=1) * 0.1
        mtn_grav = np.mean(grv_blocks, axis=1) # Or just use [0,0,9.81]
        mtn_yaw = np.mean(yaw_blocks, axis=1).reshape(-1, 1)
        mtn_input = np.concatenate([mtn_accel, mtn_grav, mtn_yaw], axis=1) # (steps, 7)
        
        # -- NoiseNet Features (18 accel + 18 gyro + 15 fft + 3 broadcast = 54) --
        # 18 = 3 axes * 6 stats (std, max, min, rms, skewness, kurtosis)
        def calc_stats(blocks):
            # blocks: (steps, 10, 3)
            mean_val = np.mean(blocks, axis=1, keepdims=True)
            std_val = np.std(blocks, axis=1)
            max_val = np.max(blocks, axis=1)
            min_val = np.min(blocks, axis=1)
            rms_val = np.sqrt(np.mean(blocks**2, axis=1))
            
            # Simplified skewness/kurtosis to keep it fast and avoid scipy dependency per step
            # skew: E[(x-mu)^3] / std^3
            diff = blocks - mean_val
            skew = np.mean(diff**3, axis=1) / (std_val**3 + 1e-8)
            kurt = np.mean(diff**4, axis=1) / (std_val**4 + 1e-8)
            
            return np.concatenate([std_val, max_val, min_val, rms_val, skew, kurt], axis=1) # (steps, 18)
            
        accel_feat = calc_stats(a_blocks)
        gyro_feat = calc_stats(g_blocks)
        
        # FFT features (5 bands * 3 axes = 15)
        # 10Hz sampling. Nyquist is 5Hz. 1s = 10 points. 
        # Frequencies: 0, 1, 2, 3, 4, 5 Hz.
        fft_vals = np.abs(np.fft.rfft(a_blocks, axis=1)) # (steps, 6, 3)
        # Ignore DC (0Hz), keep 1-5Hz (5 bands)
        fft_feat = fft_vals[:, 1:, :].reshape(steps, 15)
        
        # Broadcast features
        shock_sum = np.sum(shock_blocks, axis=1)
        vr_seed_arr = np.full((steps,), vr_seed)
        
        # Note: if gps_age is an array, take the mean. If float, use as is.
        if isinstance(gps_age, np.ndarray):
            gps_age_arr = np.mean(gps_age[:steps*10].reshape(steps, 10), axis=1)
        else:
            gps_age_arr = np.full((steps,), gps_age)
            
        broadcast_feat = np.stack([vr_seed_arr, gps_age_arr, shock_sum], axis=1)
        
        return mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat

    def __getitem__(self, idx):
        win = self.windows[idx]
        
        # Get raw data
        raw_a = win['blackout']['raw_accel'].copy()
        raw_g = win['blackout']['raw_gyro'].copy()
        
        if self.augment:
            raw_a, raw_g = self.rot_aug(raw_a, raw_g)
            
        # Standardize if scalers exist
        if self.scalers is not None:
            raw_a = self.scalers['raw_accel'].transform(raw_a)
            raw_g = self.scalers['raw_gyro'].transform(raw_g)
            
        grav = win['blackout']['gravity'].copy()
        if self.scalers is not None:
            grav = self.scalers['gravity'].transform(grav)
            
        yaw = win['blackout']['orientation_yaw_vehicle_rad']
        vr_seed = win['context']['vr_seed']
        gps_age = win['ground_truth'].get('gps_age_seq', win['context']['gps_age_at_entry'])
        shock_mask = win['blackout']['shock_mask']
        
        # Compute 1Hz features
        mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat = self._extract_1hz_features(
            raw_a, raw_g, grav, yaw, vr_seed, gps_age, shock_mask
        )
        
        # Prepare 10Hz tensors for EKF
        raw_a_10hz = torch.FloatTensor(raw_a)
        raw_g_10hz = torch.FloatTensor(raw_g)
        
        # Ground truth
        gt_speeds = torch.FloatTensor(win['ground_truth']['speeds_ms'].astype(np.float32))
        gt_cum_disp = torch.FloatTensor(win['ground_truth']['cum_disp_m'].astype(np.float32))
        
        # Context extraction for EKF initialization
        initial_yaw = win['context']['initial_yaw']
        
        out = {
            'mtn_input': torch.FloatTensor(mtn_input.astype(np.float32)),
            'accel_feat': torch.FloatTensor(accel_feat.astype(np.float32)),
            'gyro_feat': torch.FloatTensor(gyro_feat.astype(np.float32)),
            'fft_feat': torch.FloatTensor(fft_feat.astype(np.float32)),
            'broadcast_feat': torch.FloatTensor(broadcast_feat.astype(np.float32)),
            'raw_a_10hz': raw_a_10hz,
            'raw_g_10hz': raw_g_10hz,
            'vr_seed': torch.FloatTensor([float(vr_seed)]),
            'initial_yaw': torch.FloatTensor([float(initial_yaw)]),
            'gt_speeds': gt_speeds,
            'gt_cum_disp': gt_cum_disp,
            'total_dist': torch.FloatTensor([float(win['ground_truth']['total_distance_m'])]),
            'blackout_dur': torch.IntTensor([int(win['blackout_dur_sec'])])
        }
        return out
