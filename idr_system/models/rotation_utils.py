"""
Rotation utilities for the IDR pipeline.

Convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (ZYX intrinsic / aerospace)
    R is a BODY → WORLD rotation.
    Given v_body, v_world = R @ v_body.

    These matrices are orthonormal: R.T == R^{-1}.
    Angles are in radians.

All callers in the IDR pipeline use this single convention. Do not
introduce a second convention. If you need the opposite direction,
use `apply_inverse_rotation`, not a manual transpose.
"""

import torch


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Build a body → world rotation matrix from ZYX Euler angles.

    Args:
        roll, pitch, yaw: tensors of shape (...,) in radians.

    Returns:
        Tensor of shape (..., 3, 3).
    """
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    R = torch.zeros(roll.shape + (3, 3),
                    dtype=roll.dtype, device=roll.device)

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
    Apply a body → world rotation to vectors.

    Args:
        vectors: (..., 3) in body frame.
        R:       (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) in world frame.
    """
    return torch.matmul(R, vectors.unsqueeze(-1)).squeeze(-1)


def apply_inverse_rotation(vectors, R):
    """
    Apply the inverse of a body → world rotation (i.e., world → body).

    Args:
        vectors: (..., 3) in world frame.
        R:       (..., 3, 3) body→world rotation matrices.

    Returns:
        (..., 3) in body frame.
    """
    return torch.matmul(R.transpose(-1, -2),
                        vectors.unsqueeze(-1)).squeeze(-1)


def rotation_is_orthonormal(R, atol=1e-5):
    """
    Sanity check: verify R @ R.T ≈ I.

    Args:
        R:   (..., 3, 3)
        atol: tolerance

    Returns:
        bool
    """
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    RRt = torch.matmul(R, R.transpose(-1, -2))
    return bool(torch.allclose(RRt, I.expand_as(RRt), atol=atol))


def rotation_matrix_from_euler(roll, pitch, yaw):
    """Alias for euler_to_rotation_matrix — clearer at call sites."""
    return euler_to_rotation_matrix(roll, pitch, yaw)