import torch

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Creates a 3x3 rotation matrix from euler angles (roll, pitch, yaw).
    Convention: ZYX (yaw, pitch, roll) intrinsic rotations.
    R = R_z(yaw) * R_y(pitch) * R_x(roll)
    
    Args:
        roll: Tensor of shape (...,) in radians
        pitch: Tensor of shape (...,) in radians
        yaw: Tensor of shape (...,) in radians
        
    Returns:
        Tensor of shape (..., 3, 3) representing the rotation matrices.
    """
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    
    # R_x (Roll)
    # [1, 0, 0]
    # [0, cos(r), -sin(r)]
    # [0, sin(r), cos(r)]
    
    # R_y (Pitch)
    # [cos(p), 0, sin(p)]
    # [0, 1, 0]
    # [-sin(p), 0, cos(p)]
    
    # R_z (Yaw)
    # [cos(y), -sin(y), 0]
    # [sin(y), cos(y), 0]
    # [0, 0, 1]
    
    # Combined R = R_z * R_y * R_x
    R = torch.zeros(roll.shape + (3, 3), dtype=roll.dtype, device=roll.device)
    
    R[..., 0, 0] = cos_y * cos_p
    R[..., 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[..., 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    
    R[..., 1, 0] = sin_y * cos_p
    R[..., 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[..., 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    
    R[..., 2, 0] = -sin_p
    R[..., 2, 1] = cos_p * sin_r
    R[..., 2, 2] = cos_p * cos_r
    
    return R

def apply_rotation(vectors, R):
    """
    Applies rotation matrices to vectors.
    
    Args:
        vectors: Tensor of shape (..., 3)
        R: Tensor of shape (..., 3, 3)
        
    Returns:
        Tensor of shape (..., 3) rotated vectors.
    """
    # Using unsqueeze and squeeze to handle batched matrix-vector multiplication
    return torch.matmul(R, vectors.unsqueeze(-1)).squeeze(-1)
