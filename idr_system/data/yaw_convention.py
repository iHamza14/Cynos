import numpy as np

def wrap_angle_rad(angle_rad):
    """Wraps an angle in radians to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
