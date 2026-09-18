"""
PyTorch Dataset for IDR training.

Reads pre-built window pickles and produces the batch dict expected by
IDRModel.forward(batch).
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import RotationAugmentation, SpeedScaling


def compass_to_math(yaw_c):
    """Compass heading (0=North, CW) → math yaw (0=+X, CCW)."""
    return (np.pi / 2 - yaw_c + np.pi) % (2 * np.pi) - np.pi


class IDRDataset(Dataset):
    """
    Output keys (matching IDRModel.forward):
        mtn_input         (T_1hz, 7)
        accel_feat        (T_1hz, 18)
        gyro_feat         (T_1hz, 18)
        fft_feat          (T_1hz, 15)
        broadcast_feat    (T_1hz, 3)
        raw_accel_10hz    (T_10hz, 3)   — filtered, standardized
        raw_gyro_10hz     (T_10hz, 3)   — filtered, standardized
        initial_state     (18,)
        gt_speeds         (T_10hz,)
        gt_cum_disp       (T_10hz, 2)
        total_dist        (1,)
        blackout_dur      (1,)
    """

    def __init__(self, pkl_path, scalers_path=None, augment=False,
                 durations=None):
        """
        Args:
            durations: iterable of blackout_dur_sec to include.
                       None = all durations.
        """
        with open(pkl_path, 'rb') as f:
            all_windows = pickle.load(f)

        if durations is not None:
            durations = set(int(d) for d in durations)
            self.windows = [w for w in all_windows
                            if int(w['blackout_dur_sec']) in durations]
        else:
            self.windows = all_windows

        self.scalers = None
        if scalers_path is not None:
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)

        self.augment = augment
        if self.augment:
            self.rot_aug = RotationAugmentation(
                max_roll_deg=30, max_pitch_deg=30, max_yaw_deg=180
            )
            self.speed_aug = SpeedScaling(max_scale=0.10)

    def __len__(self):
        return len(self.windows)

    # ─────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────
    def _extract_1hz_features(self, accel_raw, gyro_raw, gravity_raw,
                              vr_seed_scalar, gps_age_scalar, shock_mask):
        """
        All inputs are numpy, at 10 Hz.
        accel_raw, gyro_raw, gravity_raw: (N, 3) — UNSTANDARDIZED
        vr_seed_scalar, gps_age_scalar: scalars
        shock_mask: (N,) float
        """
        N = accel_raw.shape[0]
        steps = N // 10
        n_use = steps * 10

        a_blocks     = accel_raw[:n_use].reshape(steps, 10, 3)
        g_blocks     = gyro_raw[:n_use].reshape(steps, 10, 3)
        shock_blocks = shock_mask[:n_use].reshape(steps, 10)

        # ── MTN input: 7 channels ─────────────────────────────────────
        mtn_accel = np.sum(a_blocks, axis=1) * 0.1              # (steps, 3)
        mtn_grav  = np.tile(np.array([[0.0, 0.0, 9.81]]), (steps, 1))
        # Per design: no yaw channel (F3 channel is unreliable).
        # The 7th channel is a constant 0 placeholder for API compatibility.
        mtn_extra = np.zeros((steps, 1))
        mtn_input = np.concatenate([mtn_accel, mtn_grav, mtn_extra], axis=1)

        # ── Time-domain stats for accel & gyro ────────────────────────
        def calc_stats(blocks):
            mean_val = np.mean(blocks, axis=1, keepdims=True)
            std_val  = np.std(blocks, axis=1)
            max_val  = np.max(blocks, axis=1)
            min_val  = np.min(blocks, axis=1)
            rms_val  = np.sqrt(np.mean(blocks ** 2, axis=1))
            diff     = blocks - mean_val
            skew     = np.mean(diff ** 3, axis=1) / (std_val ** 3 + 1e-8)
            kurt     = np.mean(diff ** 4, axis=1) / (std_val ** 4 + 1e-8)
            return np.concatenate(
                [std_val, max_val, min_val, rms_val, skew, kurt], axis=1
            )

        accel_feat = calc_stats(a_blocks)              # (steps, 18)
        gyro_feat  = calc_stats(g_blocks)              # (steps, 18)

        # ── FFT bands — computed on RAW accel (not standardized) ──────
        fft_vals = np.abs(np.fft.rfft(a_blocks, axis=1))    # (steps, 6, 3)
        fft_feat = fft_vals[:, 1:, :].reshape(steps, 15)    # drop DC

        # ── Broadcast scalars ─────────────────────────────────────────
        shock_sum   = np.sum(shock_blocks, axis=1)          # (steps,)
        vr_arr      = np.full((steps,), vr_seed_scalar)
        gps_age_arr = np.full((steps,), gps_age_scalar)
        broadcast_feat = np.stack([vr_arr, gps_age_arr, shock_sum], axis=1)

        return mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat

    # ─────────────────────────────────────────────────────────────────
    # __getitem__
    # ─────────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        win = self.windows[idx]

        # ── Raw, unstandardized signals ───────────────────────────────
        accel_raw = win['blackout']['raw_accel'].copy()
        gyro_raw  = win['blackout']['raw_gyro'].copy()
        gravity_raw = win['blackout']['gravity'].copy()
        shock_mask  = win['blackout']['shock_mask'].copy()

        # ── Augmentation: rotate first (in physical units) ────────────
        # Rotating accel + gyro together by the same matrix simulates
        # a different phone-mount orientation.
        if self.augment:
            accel_raw, gyro_raw = self.rot_aug(accel_raw, gyro_raw)
            # Note: gravity also needs the same rotation for MTN input
            # consistency, but we pass fixed [0,0,9.81] to MTN anyway.

        # ── Standardize for EKF input (after augmentation) ────────────
        if self.scalers is not None:
            accel_std = self.scalers['raw_accel'].transform(accel_raw)
            gyro_std  = self.scalers['raw_gyro'].transform(gyro_raw)
        else:
            accel_std = accel_raw
            gyro_std  = gyro_raw

        # ── Context scalars ───────────────────────────────────────────
        vr_seed       = float(win['context']['vr_seed'])
        gps_age_entry = float(win['context']['gps_age_at_entry'])

        if self.augment:
            scale = self.speed_aug.sample()
            vr_seed *= scale

        # ── 1 Hz features (uses RAW accel for FFT) ────────────────────
        mtn_input, accel_feat, gyro_feat, fft_feat, broadcast_feat = \
            self._extract_1hz_features(
                accel_raw, gyro_raw, gravity_raw,
                vr_seed, gps_age_entry, shock_mask,
            )

        # ── Initial state for EKF (18-dim) ────────────────────────────
        # Seed yaw from last context heading (compass → math).
        yaw0_compass = float(win['context']['heading_seq'][-1])
        yaw0_math    = compass_to_math(yaw0_compass)

        # Seed roll/pitch from first 0.5s of blackout accel.
        a_init = accel_raw[:5].mean(axis=0)
        roll0  = float(np.arctan2(a_init[1], a_init[2]))
        pitch0 = float(np.arctan2(-a_init[0],
                                  np.sqrt(a_init[1] ** 2 + a_init[2] ** 2)))

        initial_state = np.zeros(18, dtype=np.float32)
        initial_state[3] = vr_seed * np.cos(yaw0_math)
        initial_state[4] = vr_seed * np.sin(yaw0_math)
        initial_state[6] = roll0
        initial_state[7] = pitch0
        initial_state[8] = yaw0_math

        # ── Ground truth ──────────────────────────────────────────────
        gt_speeds = win['ground_truth']['speeds_ms'].astype(np.float32)
        if self.augment:
            gt_speeds = gt_speeds * scale

        gt_cum_disp = win['ground_truth']['cum_disp_m'].astype(np.float32)

        return {
            'mtn_input':      torch.from_numpy(mtn_input.astype(np.float32)),
            'accel_feat':     torch.from_numpy(accel_feat.astype(np.float32)),
            'gyro_feat':      torch.from_numpy(gyro_feat.astype(np.float32)),
            'fft_feat':       torch.from_numpy(fft_feat.astype(np.float32)),
            'broadcast_feat': torch.from_numpy(broadcast_feat.astype(np.float32)),
            'raw_accel_10hz': torch.from_numpy(accel_std.astype(np.float32)),
            'raw_gyro_10hz':  torch.from_numpy(gyro_std.astype(np.float32)),
            'initial_state':  torch.from_numpy(initial_state),
            'gt_speeds':      torch.from_numpy(gt_speeds),
            'gt_cum_disp':    torch.from_numpy(gt_cum_disp),
            'total_dist':     torch.tensor([float(win['ground_truth']['total_distance_m'])]),
            'blackout_dur':   torch.tensor([int(win['blackout_dur_sec'])]),
        }