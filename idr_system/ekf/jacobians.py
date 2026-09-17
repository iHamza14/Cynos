import torch
import math
from models.rotation_utils import euler_to_rotation_matrix

def compute_F_jacobian(state, accel_raw, gyro_raw, dt=0.1):
    """
    Computes the analytic Jacobian F = ∂f / ∂x for the EKF predict step.
    
    State (18-dim): [px, py, pz, vx, vy, vz, roll, pitch, yaw, 
                     b_ax, b_ay, b_az, b_gx, b_gy, b_gz, psi_rate_corr, psi_res, v_fwd]
    """
    batch_size = state.shape[0]
    device = state.device
    F = torch.eye(18, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Extract state variables
    roll = state[:, 6]
    pitch = state[:, 7]
    yaw = state[:, 8]
    
    b_ax = state[:, 9]
    b_ay = state[:, 10]
    b_az = state[:, 11]
    
    # Biases
    # gyro inputs are corrected by bias
    # True accel = accel_raw - b_a
    # We need to compute derivative of rotated accel w.r.t roll, pitch, yaw
    # a_world = R * (accel_raw - b_a)
    
    a_x = accel_raw[:, 0] - b_ax
    a_y = accel_raw[:, 1] - b_ay
    a_z = accel_raw[:, 2] - b_az
    
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    
    # Position w.r.t velocity
    F[:, 0, 3] = dt
    F[:, 1, 4] = dt
    F[:, 2, 5] = dt
    
    # Velocity w.r.t bias
    # R * (-I) * dt
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    F[:, 3:6, 9:12] = -R * dt
    
    # Orientation w.r.t gyro bias
    # roll, pitch, yaw = old_r, old_p, old_y + (gyro - b_g) * dt
    # This is a simplification (Euler integration).
    F[:, 6:9, 12:15] = -torch.eye(3, device=device).unsqueeze(0) * dt
    
    # Yaw w.r.t psi_rate_corr
    F[:, 8, 15] = dt
    
    # Velocity w.r.t attitude (analytic derivatives of R w.r.t euler angles)
    # v_new = v_old + R * a * dt
    # dv_new / d(roll, pitch, yaw) = dR / d(roll, pitch, yaw) * a * dt
    
    # dR/droll
    dR_dr = torch.zeros_like(R)
    dR_dr[:, 0, 1] = cos_y * sin_p * cos_r + sin_y * sin_r
    dR_dr[:, 0, 2] = -cos_y * sin_p * sin_r + sin_y * cos_r
    dR_dr[:, 1, 1] = sin_y * sin_p * cos_r - cos_y * sin_r
    dR_dr[:, 1, 2] = -sin_y * sin_p * sin_r - cos_y * cos_r
    dR_dr[:, 2, 1] = cos_p * cos_r
    dR_dr[:, 2, 2] = -cos_p * sin_r
    
    # dR/dpitch
    dR_dp = torch.zeros_like(R)
    dR_dp[:, 0, 0] = -cos_y * sin_p
    dR_dp[:, 0, 1] = cos_y * cos_p * sin_r
    dR_dp[:, 0, 2] = cos_y * cos_p * cos_r
    dR_dp[:, 1, 0] = -sin_y * sin_p
    dR_dp[:, 1, 1] = sin_y * cos_p * sin_r
    dR_dp[:, 1, 2] = sin_y * cos_p * cos_r
    dR_dp[:, 2, 0] = -cos_p
    dR_dp[:, 2, 1] = -sin_p * sin_r
    dR_dp[:, 2, 2] = -sin_p * cos_r
    
    # dR/dyaw
    dR_dy = torch.zeros_like(R)
    dR_dy[:, 0, 0] = -sin_y * cos_p
    dR_dy[:, 0, 1] = -sin_y * sin_p * sin_r - cos_y * cos_r
    dR_dy[:, 0, 2] = -sin_y * sin_p * cos_r + cos_y * sin_r
    dR_dy[:, 1, 0] = cos_y * cos_p
    dR_dy[:, 1, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    dR_dy[:, 1, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    
    a_vec = torch.stack([a_x, a_y, a_z], dim=-1).unsqueeze(-1)
    
    dv_dr = torch.matmul(dR_dr, a_vec).squeeze(-1) * dt
    dv_dp = torch.matmul(dR_dp, a_vec).squeeze(-1) * dt
    dv_dy = torch.matmul(dR_dy, a_vec).squeeze(-1) * dt
    
    F[:, 3:6, 6] = dv_dr
    F[:, 3:6, 7] = dv_dp
    F[:, 3:6, 8] = dv_dy
    
    # Position w.r.t attitude (0.5 * a_world * dt^2)
    F[:, 0:3, 6] = 0.5 * dv_dr * dt
    F[:, 0:3, 7] = 0.5 * dv_dp * dt
    F[:, 0:3, 8] = 0.5 * dv_dy * dt
    
    # Position w.r.t accel bias (0.5 * (-R) * dt^2)
    F[:, 0:3, 9:12] = 0.5 * -R * (dt**2)
    
    return F
