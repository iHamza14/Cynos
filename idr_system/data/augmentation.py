import numpy as np
from scipy.spatial.transform import Rotation as R

class RotationAugmentation:
    def __init__(self, max_roll=30, max_pitch=30, max_yaw=180):
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.max_yaw = max_yaw

    def __call__(self, accel, gyro):
        # Random angles in degrees
        roll = np.random.uniform(-self.max_roll, self.max_roll)
        pitch = np.random.uniform(-self.max_pitch, self.max_pitch)
        yaw = np.random.uniform(-self.max_yaw, self.max_yaw)
        
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        
        # Apply rotation
        accel_rot = rot.apply(accel)
        gyro_rot = rot.apply(gyro)
        
        return accel_rot, gyro_rot

class VibrationInjection:
    def __init__(self, sample_rate=10, min_freq=5, max_freq=20, amplitude=0.1):
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.amplitude = amplitude

    def __call__(self, accel):
        N = accel.shape[0]
        t = np.arange(N) / self.sample_rate
        
        # Simple noise injection for this placeholder
        noise = np.random.normal(0, self.amplitude, accel.shape)
        return accel + noise
        
class TimeShift:
    def __init__(self, max_shift_sec=0.5, sample_rate=10):
        self.max_shift_samples = int(max_shift_sec * sample_rate)

    def __call__(self, data):
        shift = np.random.randint(-self.max_shift_samples, self.max_shift_samples + 1)
        if shift == 0:
            return data
        
        shifted_data = np.roll(data, shift, axis=0)
        # In a real scenario, handle padding or trimming correctly
        return shifted_data

class SpeedScaling:
    def __init__(self, max_scale=0.10):
        self.max_scale = max_scale

    def __call__(self, speed):
        scale = np.random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale)
        return speed * scale
