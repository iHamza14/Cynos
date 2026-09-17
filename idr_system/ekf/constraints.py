import torch

def nhc_measurement_jacobian(state):
    """
    Non-Holonomic Constraints (NHC) assume lateral and vertical velocity in vehicle frame are 0.
    v_body = R^T * v_world
    We want v_body_y = 0 and v_body_z = 0.
    
    Returns H matrix of shape (batch, 2, 18) for the NHC measurement.
    """
    batch = state.shape[0]
    device = state.device
    
    # H = d(v_body_y, v_body_z) / dx
    H = torch.zeros(batch, 2, 18, device=device)
    
    roll = state[:, 6]
    pitch = state[:, 7]
    yaw = state[:, 8]
    
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    
    # We are differentiating v_body = R^T * v_world.
    # R^T elements:
    # R^T_10 = sin(y)cos(p)
    # R^T_11 = sin(y)sin(p)sin(r) + cos(y)cos(r)
    # R^T_12 = sin(y)sin(p)cos(r) - cos(y)sin(r)
    
    # Just numeric or analytic for the velocity part:
    # v_body_y = R^T_10 * v_x + R^T_11 * v_y + R^T_12 * v_z
    
    # Since NHC is highly non-linear w.r.t attitude, a full analytic jacobian is complex.
    # For many implementations, we just use the velocity part of H, assuming attitude is decoupled.
    # Velocity part:
    
    H[:, 0, 3] = sin_y * cos_p
    H[:, 0, 4] = sin_y * sin_p * sin_r + cos_y * cos_r
    H[:, 0, 5] = sin_y * sin_p * cos_r - cos_y * sin_r
    
    H[:, 1, 3] = -sin_p
    H[:, 1, 4] = cos_p * sin_r
    H[:, 1, 5] = cos_p * cos_r
    
    return H
