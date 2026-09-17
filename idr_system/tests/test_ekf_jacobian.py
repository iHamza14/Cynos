import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekf.jacobians import compute_F_jacobian
from models.rotation_utils import euler_to_rotation_matrix, apply_rotation

def state_transition(state, accel_raw, gyro_raw, dt):
    """
    Simulates the EKF predict step to compute the true Jacobian via autograd.
    """
    roll = state[:, 6]
    pitch = state[:, 7]
    yaw = state[:, 8]
    
    b_gx = state[:, 12]
    b_gy = state[:, 13]
    b_gz = state[:, 14]
    psi_rate_corr = state[:, 15]
    
    new_roll = roll + (gyro_raw[:, 0] - b_gx) * dt
    new_pitch = pitch + (gyro_raw[:, 1] - b_gy) * dt
    new_yaw = yaw + (gyro_raw[:, 2] - b_gz + psi_rate_corr) * dt
    
    b_ax = state[:, 9]
    b_ay = state[:, 10]
    b_az = state[:, 11]
    
    a_true = accel_raw - torch.stack([b_ax, b_ay, b_az], dim=-1)
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    a_world = apply_rotation(a_true, R)
    
    new_v = state[:, 3:6] + a_world * dt
    new_p = state[:, 0:3] + state[:, 3:6] * dt + 0.5 * a_world * dt**2
    
    new_state = state.clone()
    new_state[:, 0:3] = new_p
    new_state[:, 3:6] = new_v
    new_state[:, 6] = new_roll
    new_state[:, 7] = new_pitch
    new_state[:, 8] = new_yaw
    
    return new_state

def test_jacobian():
    batch_size = 2
    state = torch.randn(batch_size, 18, requires_grad=True, dtype=torch.float64)
    accel_raw = torch.randn(batch_size, 3, dtype=torch.float64)
    gyro_raw = torch.randn(batch_size, 3, dtype=torch.float64)
    dt = 0.1
    
    # 1. Analytic F
    F_analytic = compute_F_jacobian(state, accel_raw, gyro_raw, dt)
    
    # 2. Autograd F
    F_autograd = torch.zeros(batch_size, 18, 18, dtype=torch.float64)
    
    for b in range(batch_size):
        # We need a wrapper function for autograd.jacobian
        def func(s):
            return state_transition(s.unsqueeze(0), accel_raw[b].unsqueeze(0), gyro_raw[b].unsqueeze(0), dt).squeeze(0)
            
        J = torch.autograd.functional.jacobian(func, state[b])
        F_autograd[b] = J
        
    diff = torch.max(torch.abs(F_analytic - F_autograd))
    print(f"Max Jacobian difference: {diff.item():.6e}")
    
    if diff >= 1e-4:
        err = torch.abs(F_analytic - F_autograd)
        idx = torch.where(err > 1e-4)
        for i in range(len(idx[0])):
            b, r, c = idx[0][i], idx[1][i], idx[2][i]
            print(f"Mismatch at Batch {b}, Row {r}, Col {c}: Analytic={F_analytic[b,r,c]:.6e}, Autograd={F_autograd[b,r,c]:.6e}")
    
    if diff < 1e-4:
        print("PASS: Analytic Jacobian matches Autograd.")
    else:
        print("FAIL: Analytic Jacobian diverges from Autograd.")
        sys.exit(1)

if __name__ == "__main__":
    test_jacobian()
