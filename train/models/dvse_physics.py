import torch
import math

def euler_to_rotation_matrix(angles):
    """
    Converts alpha (x), beta (y), gamma (z) Euler angles to a rotation matrix R_v^p.
    R_v^p = Rx(alpha) * Ry(beta) * Rz(gamma)

    Args:
        angles: Tensor of shape [..., 3] containing (alpha, beta, gamma) in radians.
    
    Returns:
        Tensor of shape [..., 3, 3] representing the rotation matrix.
    """
    alpha = angles[..., 0]
    beta = angles[..., 1]
    gamma = angles[..., 2]
    
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)
    
    # Rx
    Rx = torch.stack([
        torch.stack([one, zero, zero], dim=-1),
        torch.stack([zero, torch.cos(alpha), -torch.sin(alpha)], dim=-1),
        torch.stack([zero, torch.sin(alpha), torch.cos(alpha)], dim=-1)
    ], dim=-2)
    
    # Ry
    Ry = torch.stack([
        torch.stack([torch.cos(beta), zero, torch.sin(beta)], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-torch.sin(beta), zero, torch.cos(beta)], dim=-1)
    ], dim=-2)
    
    # Rz
    Rz = torch.stack([
        torch.stack([torch.cos(gamma), -torch.sin(gamma), zero], dim=-1),
        torch.stack([torch.sin(gamma), torch.cos(gamma), zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1)
    ], dim=-2)
    
    return Rx @ Ry @ Rz

def physics_velocity_update(acc_window, R_vp, N, v_0=None, hz=10):
    """
    Applies the physics-based velocity update.
    
    Args:
        acc_window: [B, L, 3] raw acceleration in phone frame.
        R_vp: [B, T, 3, 3] phone to vehicle rotation matrices (from MTN).
        N: [B, T, 1] learned velocity disturbance (from NCN).
        v_0: [B, 1] Initial velocity at the start of the 10-second window. If None, assumes 0.
        hz: Sampling rate.
        
    Returns:
        delta_v: [B, T] total 1-second velocity increments.
        v: [B, T] integrated absolute velocity.
        delta_v_physics: [B, T] physical 1-second velocity increments.
    """
    B, L, _ = acc_window.shape
    T = L // hz
    dt = 1.0 / hz
    
    g_w = torch.tensor([0.0, 0.0, 9.81], dtype=acc_window.dtype, device=acc_window.device)
    
    # [B, T, hz, 3]
    acc_grouped = acc_window[:, :T*hz, :].view(B, T, hz, 3)
    
    # R_p^w = (R_v^p)^T. g_p = (R_v^p)^T g_w
    R_vp_T = R_vp.transpose(-1, -2) # [B, T, 3, 3]
    g_p = torch.einsum('b...ij,j->b...i', R_vp_T, g_w) # [B, T, 3]
    
    # Expand to hz samples per second
    g_p_expand = g_p.unsqueeze(2) # [B, T, 1, 3]
    R_vp_expand = R_vp.unsqueeze(2) # [B, T, 1, 3, 3]
    
    # a_p - g_p
    a_p_no_g = acc_grouped - g_p_expand # [B, T, hz, 3]
    
    # a_v = R_v^p (a_p - g_p)
    a_v = torch.einsum('b...ij,b...j->b...i', R_vp_expand, a_p_no_g) # [B, T, hz, 3]
    
    # Vehicle +Y forward projection
    a_f = a_v[..., 1] # [B, T, hz]
    
    # 1-second integration
    delta_v_physics = torch.sum(a_f * dt, dim=2) # [B, T]
    
    # Delta v = Delta v_physics + N
    delta_v = delta_v_physics + N.squeeze(-1) # [B, T]
    
    # Recursive velocity integration: v[k] = v[k-1] + delta_v[k]
    v_diff = torch.cumsum(delta_v, dim=1) # [B, T]
    if v_0 is not None:
        v = v_0 + v_diff
    else:
        v = v_diff
        
    return delta_v, v, delta_v_physics

def apply_random_rotation_augmentation(acc, gyro, max_angle_rad=math.pi):
    """
    Applies a random 3-axis rotation R_r = Rx(alpha) * Ry(beta) * Rz(gamma) to the entire window.
    Section 7 of Cynos DVSE architecture spec.

    Args:
        acc: [B, L, 3] or [L, 3] accelerometer measurements.
        gyro: [B, L, 3] or [L, 3] gyroscope measurements.
        max_angle_rad: maximum rotation angle in radians (default: pi).
    Returns:
        acc_rot, gyro_rot: rotated tensors with identical relative physical transformation.
    """
    is_batch = acc.ndim == 3
    if not is_batch:
        acc = acc.unsqueeze(0)
        gyro = gyro.unsqueeze(0)

    B = acc.shape[0]
    angles = (torch.rand(B, 3, device=acc.device, dtype=acc.dtype) * 2 - 1) * max_angle_rad
    R_r = euler_to_rotation_matrix(angles) # [B, 3, 3]

    acc_rot = torch.einsum('bij,blj->bli', R_r, acc)
    gyro_rot = torch.einsum('bij,blj->bli', R_r, gyro)

    if not is_batch:
        acc_rot = acc_rot.squeeze(0)
        gyro_rot = gyro_rot.squeeze(0)

    return acc_rot, gyro_rot

