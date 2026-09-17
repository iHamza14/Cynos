import numpy as np

def bin_dataset(df_s, df_v):
    """
    Dummy bin builder function for the context of IDR System generation.
    It expects the smartphone (10Hz) and vehicle (100Hz) dataframes.
    
    Returns a list of 10Hz bins containing raw_accel, raw_gyro, gravity, etc.
    This acts as a placeholder for the actual preprocessing logic that aligns
    the two dataframes.
    """
    # In a real scenario, this would use pandas merge_asof with 200ms tolerance
    # and enforce the 500ms IMU-to-GNSS lag.
    bins = []
    # Dummy logic to fulfill the import signature for now
    return bins

def cumulative_displacement(speeds, headings_rad, dt=0.1):
    """
    Computes cumulative displacement (x, y) given speeds and headings.
    
    Args:
        speeds (np.ndarray): (N,) speeds in m/s
        headings_rad (np.ndarray): (N,) unwrapped headings in radians
        dt (float): Timestep in seconds
        
    Returns:
        np.ndarray: (N, 2) cumulative displacements
    """
    vx = speeds * np.cos(headings_rad)
    vy = speeds * np.sin(headings_rad)
    
    dx = vx * dt
    dy = vy * dt
    
    disp = np.zeros((len(speeds), 2))
    disp[:, 0] = np.cumsum(dx)
    disp[:, 1] = np.cumsum(dy)
    
    return disp
