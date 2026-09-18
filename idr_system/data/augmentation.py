"""
Data augmentation for the IDR pipeline.

Applied during training to reduce overfitting to the specific phone/mount/
vehicle in the training data. Not applied at validation or test time.

Augmentations:
  RotationAugmentation — simulates different phone mount orientations
  VibrationInjection   — adds band-limited noise simulating engine vibration
  TimeShift            — jitters IMU vs GPS alignment
  SpeedScaling         — scales speeds to simulate different road types
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


# ─────────────────────────────────────────────────────────────────────────
# 1. Phone-to-vehicle rotation augmentation (Paper 2's key augmentation)
# ─────────────────────────────────────────────────────────────────────────

class RotationAugmentation:
    """
    Rotate accel and gyro by a random 3-axis rotation.

    Mathematically equivalent to having mounted the phone at a different
    angle relative to the vehicle. Same rotation applied to accel AND gyro
    AND across the entire sequence — preserves the physical relationship
    between the two sensors.
    """

    def __init__(self, max_roll_deg=30, max_pitch_deg=30, max_yaw_deg=180):
        self.max_roll = max_roll_deg
        self.max_pitch = max_pitch_deg
        self.max_yaw = max_yaw_deg

    def __call__(self, accel, gyro):
        """
        accel, gyro: (N, 3) numpy arrays.
        Returns rotated copies.
        """
        roll  = np.random.uniform(-self.max_roll,  self.max_roll)
        pitch = np.random.uniform(-self.max_pitch, self.max_pitch)
        yaw   = np.random.uniform(-self.max_yaw,   self.max_yaw)

        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        accel_rot = rot.apply(accel)
        gyro_rot  = rot.apply(gyro)

        return accel_rot, gyro_rot

    def sample_rotation_matrix(self):
        """
        Return the 3x3 rotation matrix for callers who need to apply it
        consistently to other signals (e.g., orientation_yaw_raw_rad).
        """
        roll  = np.random.uniform(-self.max_roll,  self.max_roll)
        pitch = np.random.uniform(-self.max_pitch, self.max_pitch)
        yaw   = np.random.uniform(-self.max_yaw,   self.max_yaw)
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        return rot.as_matrix()


# ─────────────────────────────────────────────────────────────────────────
# 2. Band-limited vibration injection
# ─────────────────────────────────────────────────────────────────────────

class VibrationInjection:
    """
    Add band-limited noise to accel, simulating engine vibration at 5–20 Hz.

    Because the smartphone samples at 10 Hz (Nyquist = 5 Hz), a 5–20 Hz
    band aliases into the 0–5 Hz band. To simulate this correctly, generate
    noise at a higher rate, band-pass to 5–20 Hz, then downsample to 10 Hz.
    """

    def __init__(self, sample_rate=10, min_freq=5.0, max_freq=20.0,
                 amplitude=0.1, upsample_factor=10):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.amplitude = amplitude
        self.upsample_factor = upsample_factor

    def __call__(self, accel):
        """
        accel: (N, 3) numpy array.
        Returns accel + band-limited noise.
        """
        N = accel.shape[0]
        fs_hi = self.sample_rate * self.upsample_factor
        nyq_hi = 0.5 * fs_hi

        # Generate white noise at high rate
        noise_hi = np.random.normal(0, 1.0, (N * self.upsample_factor, 3))

        # Band-pass to [min_freq, max_freq]
        low  = self.min_freq  / nyq_hi
        high = self.max_freq  / nyq_hi
        # Clamp to valid band
        low  = max(low,  1e-4)
        high = min(high, 0.999)
        b, a = butter(4, [low, high], btype='band')
        noise_hi = filtfilt(b, a, noise_hi, axis=0)

        # Downsample to 10 Hz by taking every Nth sample
        noise = noise_hi[::self.upsample_factor]
        noise = noise[:N]                      # safety trim

        # Normalize to target amplitude
        noise = noise / (noise.std() + 1e-8) * self.amplitude
        return accel + noise


# ─────────────────────────────────────────────────────────────────────────
# 3. Time shift (jitter IMU vs GPS alignment)
# ─────────────────────────────────────────────────────────────────────────

class TimeShift:
    """
    Shift the IMU stream by ±max_shift samples.

    Uses edge-padding instead of np.roll so no discontinuous jump appears
    in the middle of the signal.
    """

    def __init__(self, max_shift_sec=0.5, sample_rate=10):
        self.max_shift_samples = int(max_shift_sec * sample_rate)

    def __call__(self, data):
        """
        data: (N, C) or (N,) numpy array.
        Returns shifted array of the same shape.
        """
        if self.max_shift_samples == 0:
            return data

        shift = np.random.randint(-self.max_shift_samples,
                                   self.max_shift_samples + 1)
        if shift == 0:
            return data

        N = data.shape[0]
        out = np.empty_like(data)

        if shift > 0:
            # Move data forward; pad the beginning with the first sample
            out[:shift] = data[0]
            out[shift:] = data[:-shift]
        else:
            shift = -shift
            # Move data backward; pad the end with the last sample
            out[-shift:] = data[-1]
            out[:-shift] = data[shift:]

        return out


# ─────────────────────────────────────────────────────────────────────────
# 4. Speed scaling (simulate different road types)
# ─────────────────────────────────────────────────────────────────────────

class SpeedScaling:
    """
    Multiply speeds by a random scalar in [1-max, 1+max].

    Should be applied consistently to:
      - context.vr_seed
      - context.speed_seq
      - ground_truth.speeds_ms

    Otherwise the model sees inconsistent targets.
    """

    def __init__(self, max_scale=0.10):
        self.max_scale = max_scale
        self.scale = 1.0

    def sample(self):
        """Pick a new scale factor. Call once per window."""
        self.scale = np.random.uniform(1.0 - self.max_scale,
                                        1.0 + self.max_scale)
        return self.scale

    def __call__(self, speed):
        """Apply the sampled scale. Call after sample() on each speed array."""
        return speed * self.scale