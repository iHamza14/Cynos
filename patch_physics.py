import re
import math

with open("train/models/dvse_physics.py", "r") as f:
    content = f.read()

# I will append the augmentation function to the file
augmentation_code = """
import math

def apply_random_rotation_augmentation(acc, gyro, max_angle_rad=math.pi):
    \"\"\"
    Applies a random 3-axis rotation R_r = Rx(alpha) * Ry(beta) * Rz(gamma) to the entire window.
    Section 7 of Cynos DVSE architecture spec.

    Args:
        acc: [B, L, 3] or [L, 3] accelerometer measurements.
        gyro: [B, L, 3] or [L, 3] gyroscope measurements.
        max_angle_rad: maximum rotation angle in radians (default: pi).
    Returns:
        acc_rot, gyro_rot: rotated tensors with identical relative physical transformation.
    \"\"\"
    is_batch = acc.ndim == 3
    if not is_batch:
        acc = acc.unsqueeze(0)
        gyro = gyro.unsqueeze(0)

    B = acc.shape[0]
    angles = (torch.rand(B, 3, device=acc.device, dtype=acc.dtype) * 2 - 1) * max_angle_rad
    R_r = euler_to_rotation_matrix(angles)  # [B, 3, 3]

    acc_rot = torch.einsum("bij,blj->bli", R_r, acc)
    gyro_rot = torch.einsum("bij,blj->bli", R_r, gyro)

    if not is_batch:
        acc_rot = acc_rot.squeeze(0)
        gyro_rot = gyro_rot.squeeze(0)

    return acc_rot, gyro_rot
"""

if "apply_random_rotation_augmentation" not in content:
    with open("train/models/dvse_physics.py", "a") as f:
        f.write(augmentation_code)
