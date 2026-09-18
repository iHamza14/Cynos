import torch
import torch.nn as nn

from models.rotation_utils import euler_to_rotation_matrix, apply_rotation
from ekf.jacobians import compute_F_jacobian


class DifferentiableEKF(nn.Module):
    """
    Differentiable EKF for the IDR pipeline.

    State (18): [px, py, pz, vx, vy, vz, roll, pitch, yaw,
                 b_ax, b_ay, b_az, b_gx, b_gy, b_gz,
                 psi_rate_corr, psi_res, v_fwd]

    Conventions:
      - yaw is math convention (0 = +X, increases CCW)
      - accel_body is in phone frame; R_pv rotates it to vehicle frame
      - gyro_body is in phone frame; yaw rate at index 1
    """

    def __init__(self, config):
        super().__init__()
        ekf_cfg = config['ekf']

        self.dt = ekf_cfg['dt']
        self.r_gps_base = ekf_cfg['r_gps_base']
        self.r_gps_age_exponent = ekf_cfg['r_gps_age_exponent']
        self.r_nhc = ekf_cfg['r_nhc']
        self.r_yaw_straight = ekf_cfg['r_yaw_straight']
        self.r_map_match = ekf_cfg['r_map_match']
        self.straight_yaw_rate_thresh = ekf_cfg['straight_yaw_rate_thresh_deg_s']

        # Register as buffer so it moves with the module across devices
        self.register_buffer(
            'Q',
            torch.diag(torch.tensor(ekf_cfg['q_diag'], dtype=torch.float32))
        )

    # ─────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────
    def predict(self, state, P, accel_body, gyro_body, R_pv=None):
        """
        Args:
            state:      (batch, 18)
            P:          (batch, 18, 18)
            accel_body: (batch, 3)  raw accel in phone frame (linear accel,
                                    gravity already removed in preprocess)
            gyro_body:  (batch, 3)  raw gyro in phone frame
            R_pv:       (batch, 3, 3) or None
                        phone→vehicle rotation. If None, identity (v1 behavior).
                        MTN output goes here in v2.

        Returns:
            state_new, P_new
        """
        batch_size = state.shape[0]
        device = state.device

        # ── Rotate accel from phone frame to vehicle frame ─────────────
        if R_pv is not None:
            accel_vehicle = apply_rotation(accel_body, R_pv)   # (batch, 3)
        else:
            accel_vehicle = accel_body

        # ── State unpack ───────────────────────────────────────────────
        roll  = state[:, 6]
        pitch = state[:, 7]
        yaw   = state[:, 8]

        b_ax, b_ay, b_az = state[:, 9], state[:, 10], state[:, 11]
        b_gx, b_gy, b_gz = state[:, 12], state[:, 13], state[:, 14]
        psi_rate_corr    = state[:, 15]

        # ── Attitude propagation ──────────────────────────────────────
        # Gyro axes: index 1 = yaw (empirically established)
        new_roll  = roll  + (gyro_body[:, 0] - b_gx) * self.dt
        new_pitch = pitch + (gyro_body[:, 1] - b_gy) * self.dt
        new_yaw   = yaw   + (gyro_body[:, 1] - b_gz + psi_rate_corr) * self.dt
        # ↑ pitch also uses index 1 here for v1. Once axes are confirmed for
        #   pitch/roll, replace with the correct indices (likely 0 and 2).

        # ── Velocity / position propagation ───────────────────────────
        b_a = torch.stack([b_ax, b_ay, b_az], dim=-1)
        a_true_vehicle = accel_vehicle - b_a

        R_vw = euler_to_rotation_matrix(new_roll, new_pitch, new_yaw)
        a_world = apply_rotation(a_true_vehicle, R_vw)

        new_v = state[:, 3:6] + a_world * self.dt
        new_p = state[:, 0:3] + state[:, 3:6] * self.dt + 0.5 * a_world * (self.dt ** 2)

        # ── Pack state ────────────────────────────────────────────────
        new_state = state.clone()
        new_state[:, 0:3] = new_p
        new_state[:, 3:6] = new_v
        new_state[:, 6]   = new_roll
        new_state[:, 7]   = new_pitch
        new_state[:, 8]   = new_yaw
        # biases / corrections unchanged

        # ── Covariance propagation ────────────────────────────────────
        # F is approximate (yaw-axis mismatch, no R_pv in Jacobian). This is
        # acceptable for end-to-end training; gradients flow through new_state.
        # If strict covariance propagation is needed, pass accel_vehicle and
        # compose R_pv inside compute_F_jacobian.
        F = compute_F_jacobian(state, accel_vehicle, gyro_body, self.dt)

        Q_batch = self.Q.unsqueeze(0).expand(batch_size, -1, -1)
        F_T = F.transpose(1, 2)
        new_P = torch.bmm(torch.bmm(F, P), F_T) + Q_batch

        return new_state, new_P

    # ─────────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────────
    def update(self, state, P, z, H, R, h_x=None):
        """
        EKF measurement update.

        Args:
            state:  (batch, 18)
            P:      (batch, 18, 18)
            z:      (batch, dim_z)   measurement
            H:      (batch, dim_z, 18)  measurement Jacobian
            R:      (batch, dim_z, dim_z) or scalar or (dim_z,)  noise cov
            h_x:    (batch, dim_z) or None
                    Nonlinear measurement function h(x).
                    If None, uses the linear approximation h(x) = H @ state.
                    Provide this for NHC, yaw, map-match.

        Returns:
            state_new, P_new
        """
        device = state.device
        dim_z = z.shape[-1]
        batch = state.shape[0]

        # ── Innovation ────────────────────────────────────────────────
        if h_x is None:
            h_x = torch.bmm(H, state.unsqueeze(-1)).squeeze(-1)
        y = z - h_x

        # ── R as (batch, dim_z, dim_z) ────────────────────────────────
        if isinstance(R, torch.Tensor):
            if R.dim() == 0:
                R_mat = torch.eye(dim_z, device=device) * R
                R_mat = R_mat.unsqueeze(0).expand(batch, -1, -1)
            elif R.dim() == 1:
                R_mat = torch.diag_embed(R).expand(batch, -1, -1)
            elif R.dim() == 2:
                R_mat = R.unsqueeze(0).expand(batch, -1, -1)
            else:
                R_mat = R
        else:
            R_mat = torch.eye(dim_z, device=device) * float(R)
            R_mat = R_mat.unsqueeze(0).expand(batch, -1, -1)

        # ── Kalman gain via solve ─────────────────────────────────────
        H_T = H.transpose(1, 2)
        S = torch.bmm(torch.bmm(H, P), H_T) + R_mat

        # Jitter for numerical stability
        S = S + torch.eye(dim_z, device=device).unsqueeze(0) * 1e-6

        # K = P H^T S^{-1}   ⇔   S K^T = H P
        HP = torch.bmm(H, P)                    # (batch, dim_z, 18)
        K_T = torch.linalg.solve(S, HP)         # (batch, dim_z, 18)
        K = K_T.transpose(1, 2)                 # (batch, 18, dim_z)

        # ── State update ──────────────────────────────────────────────
        state_new = state + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)

        # ── Covariance update — Joseph form for symmetry/stability ────
        I = torch.eye(18, device=device).unsqueeze(0).expand(batch, -1, -1)
        IKH = I - torch.bmm(K, H)
        P_new = (torch.bmm(torch.bmm(IKH, P), IKH.transpose(1, 2))
                 + torch.bmm(torch.bmm(K, R_mat), K.transpose(1, 2)))

        return state_new, P_new

    # ─────────────────────────────────────────────────────────────────
    # Convenience wrappers for common measurements
    # ─────────────────────────────────────────────────────────────────
    def update_nhc(self, state, P):
        """
        NHC measurement: lateral and vertical velocity in body frame are 0.
        Uses the correct nonlinear h(x) via nhc_measurement_function.
        """
        from ekf.constraints import nhc_measurement_function, nhc_measurement_jacobian

        v_body = nhc_measurement_function(state)      # (batch, 3)
        h_x = v_body[:, 1:3]                          # lateral, vertical
        H = nhc_measurement_jacobian(state)           # (batch, 2, 18)
        z = torch.zeros(state.shape[0], 2, device=state.device)

        return self.update(state, P, z, H, R=self.r_nhc, h_x=h_x)