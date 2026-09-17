import os
import sys
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rotation_utils import euler_to_rotation_matrix, apply_rotation

def test_rotation():
    pi_half = torch.tensor(math.pi / 2)
    zero = torch.tensor(0.0)
    
    # Test Z-axis rotation (Yaw = 90 deg)
    # Rotating X vector [1, 0, 0] by 90 deg around Z gives [0, 1, 0]
    R_z = euler_to_rotation_matrix(zero, zero, pi_half)
    vec_x = torch.tensor([1.0, 0.0, 0.0])
    res_z = apply_rotation(vec_x, R_z)
    
    assert torch.allclose(res_z, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6), f"Z rotation failed: {res_z}"
    
    # Test Y-axis rotation (Pitch = 90 deg)
    # Rotating X vector [1, 0, 0] by 90 deg around Y gives [0, 0, -1]
    R_y = euler_to_rotation_matrix(zero, pi_half, zero)
    res_y = apply_rotation(vec_x, R_y)
    assert torch.allclose(res_y, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6), f"Y rotation failed: {res_y}"
    
    # Test X-axis rotation (Roll = 90 deg)
    # Rotating Y vector [0, 1, 0] by 90 deg around X gives [0, 0, 1]
    R_x = euler_to_rotation_matrix(pi_half, zero, zero)
    vec_y = torch.tensor([0.0, 1.0, 0.0])
    res_x = apply_rotation(vec_y, R_x)
    assert torch.allclose(res_x, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6), f"X rotation failed: {res_x}"

    print("PASS: Rotation utility tests passed.")

if __name__ == "__main__":
    test_rotation()
