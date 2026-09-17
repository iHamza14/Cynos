import torch
import torch.nn as nn
from models.rotation_utils import euler_to_rotation_matrix, apply_rotation
from ekf.jacobians import compute_F_jacobian

class DifferentiableEKF(nn.Module):
    def __init__(self, config):
        super(DifferentiableEKF, self).__init__()
        self.dt = config['ekf']['dt']
        self.q_diag = config['ekf']['q_diag']
        self.r_gps_base = config['ekf']['r_gps_base']
        self.r_gps_age_exponent = config['ekf']['r_gps_age_exponent']
        self.r_nhc = config['ekf']['r_nhc']
        self.r_yaw_straight = config['ekf']['r_yaw_straight']
        self.r_map_match = config['ekf']['r_map_match']
        self.straight_yaw_rate_thresh = config['ekf']['straight_yaw_rate_thresh_deg_s']
        
        # Q matrix
        self.Q = torch.diag(torch.tensor(self.q_diag, dtype=torch.float32))

    def predict(self, state, P, accel_raw, gyro_raw):
        """
        state: (batch, 18)
        P: (batch, 18, 18)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 1. Compute F Jacobian
        F = compute_F_jacobian(state, accel_raw, gyro_raw, self.dt)
        
        # 2. State Predict
        roll = state[:, 6]
        pitch = state[:, 7]
        yaw = state[:, 8]
        
        b_gx = state[:, 12]
        b_gy = state[:, 13]
        b_gz = state[:, 14]
        
        psi_rate_corr = state[:, 15]
        
        # Update orientation
        new_roll = roll + (gyro_raw[:, 0] - b_gx) * self.dt
        new_pitch = pitch + (gyro_raw[:, 1] - b_gy) * self.dt
        new_yaw = yaw + (gyro_raw[:, 2] - b_gz + psi_rate_corr) * self.dt
        
        # Update velocity
        b_ax = state[:, 9]
        b_ay = state[:, 10]
        b_az = state[:, 11]
        
        a_true = accel_raw - torch.stack([b_ax, b_ay, b_az], dim=-1)
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        a_world = apply_rotation(a_true, R)
        
        # Gravity is already removed in preprocess (linear accel is fed)
        new_v = state[:, 3:6] + a_world * self.dt
        
        # Update position
        new_p = state[:, 0:3] + state[:, 3:6] * self.dt + 0.5 * a_world * self.dt**2
        
        # Re-pack state
        new_state = state.clone()
        new_state[:, 0:3] = new_p
        new_state[:, 3:6] = new_v
        new_state[:, 6] = new_roll
        new_state[:, 7] = new_pitch
        new_state[:, 8] = new_yaw
        # biases and corrections stay constant in prediction
        
        # 3. Covariance Predict
        Q_batch = self.Q.unsqueeze(0).to(device).repeat(batch_size, 1, 1)
        # P = F * P * F^T + Q
        F_T = F.transpose(1, 2)
        new_P = torch.bmm(torch.bmm(F, P), F_T) + Q_batch
        
        return new_state, new_P

    def update(self, state, P, z, H, R):
        """
        Generic EKF update using torch.linalg.solve for stability.
        z: measurement (batch, dim_z)
        H: measurement Jacobian (batch, dim_z, 18)
        R: measurement covariance (batch, dim_z, dim_z)
        """
        device = state.device
        
        # y = z - h(x) (assuming h(x) = Hx for linear measurement, which is true for velocity/pos/NHC)
        h_x = torch.bmm(H, state.unsqueeze(-1)).squeeze(-1)
        y = z - h_x
        
        # S = H P H^T + R
        H_T = H.transpose(1, 2)
        S = torch.bmm(torch.bmm(H, P), H_T) + R
        
        # K = P H^T S^-1
        # To compute S^-1 efficiently and stably, we solve: S * K^T = H * P
        # -> K^T = solve(S, H * P)
        # -> K = K^T^T
        
        # Add jitter to S for numerical stability
        jitter = torch.eye(S.shape[-1], device=device).unsqueeze(0) * 1e-6
        S_stable = S + jitter
        
        # S_stable shape: (batch, dim_z, dim_z)
        # H P shape: (batch, dim_z, 18)
        # solve(A, B) -> X such that AX = B
        # we want X = S^-1 (H P)
        
        try:
            # torch.linalg.solve(A, B) returns X where AX = B
            HP = torch.bmm(H, P)
            # We want K = P H^T S^-1, so K^T = S^-T H P = S^-1 H P (since S is symmetric)
            K_T = torch.linalg.solve(S_stable, HP)
            K = K_T.transpose(1, 2)
            
            # state = state + Ky
            state = state + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
            
            # P = (I - KH)P = P - KHP
            I = torch.eye(18, device=device).unsqueeze(0)
            P = P - torch.bmm(torch.bmm(K, H), P)
            
        except RuntimeError as e:
            # If singular, just skip update
            pass
            
        return state, P
