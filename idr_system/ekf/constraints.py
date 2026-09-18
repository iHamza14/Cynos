"""
Non-Holonomic Constraints (NHC) for ground-vehicle EKF.

Assumption: in the vehicle body frame, lateral velocity (y) and
vertical velocity (z) are zero. Forward (x) is free.

State (18-dim): [px,py,pz, vx,vy,vz, roll,pitch,yaw,
                 b_ax,b_ay,b_az, b_gx,b_gy,b_gz,
                 psi_rate_corr, psi_res, v_fwd]

NHC measurement function:
    h(x) = R(roll,pitch,yaw)^T @ v_world       (returns 3-vector v_body)
    measured: z = [0, 0]  for (v_body_y, v_body_z)
"""

import torch
from models.rotation_utils import euler_to_rotation_matrix


# ─────────────────────────────────────────────────────────────────────────
# Explicit measurement function h(x) — required for correct innovation
# ─────────────────────────────────────────────────────────────────────────

def nhc_measurement_function(state):
    """
    Compute v_body = R^T @ v_world.

    Args:
        state: (batch, 18)
    Returns:
        v_body: (batch, 3)  — [forward, lateral, vertical] in vehicle frame
    """
    roll  = state[:, 6]
    pitch = state[:, 7]
    yaw   = state[:, 8]

    v_world = state[:, 3:6]                       # (batch, 3)

    R = euler_to_rotation_matrix(roll, pitch, yaw)   # body → world
    v_body = torch.bmm(R.transpose(1, 2), v_world.unsqueeze(-1)).squeeze(-1)

    return v_body


# ─────────────────────────────────────────────────────────────────────────
# Jacobian H = ∂h/∂x for the (y, z) components of v_body
# ─────────────────────────────────────────────────────────────────────────

def nhc_measurement_jacobian(state):
    """
    NHC measurement is (v_body_y, v_body_z), both of which should be 0.

    Returns:
        H: (batch, 2, 18)
           Row 0: ∂v_body_y / ∂x
           Row 1: ∂v_body_z / ∂x

    Non-zero blocks:
        cols 3:6   — velocity
        cols 6:9   — attitude (roll, pitch, yaw)

    Bias, psi_rate_corr, psi_res, v_fwd do not affect the measurement.
    """
    batch  = state.shape[0]
    device = state.device

    H = torch.zeros(batch, 2, 18, device=device)

    roll  = state[:, 6]
    pitch = state[:, 7]
    yaw   = state[:, 8]

    vx = state[:, 3]
    vy = state[:, 4]
    vz = state[:, 5]

    cos_r, sin_r = torch.cos(roll),  torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw),   torch.sin(yaw)

    # ── Row 0: ∂v_body_y / ∂x ─────────────────────────────────────────
    # v_body_y = R[1,:] · v_world
    H[:, 0, 3] =  sin_y * cos_p
    H[:, 0, 4] =  sin_y * sin_p * sin_r + cos_y * cos_r
    H[:, 0, 5] =  sin_y * sin_p * cos_r - cos_y * sin_r

    # ∂v_body_y / ∂roll
    H[:, 0, 6] = ((sin_y * sin_p * cos_r - cos_y * sin_r) * vy
                  + (-sin_y * sin_p * sin_r - cos_y * cos_r) * vz)

    # ∂v_body_y / ∂pitch
    H[:, 0, 7] = (-sin_y * sin_p * vx
                  + sin_y * cos_p * sin_r * vy
                  + sin_y * cos_p * cos_r * vz)

    # ∂v_body_y / ∂yaw
    H[:, 0, 8] = ( cos_y * cos_p * vx
                  + (cos_y * sin_p * sin_r - sin_y * cos_r) * vy
                  + (cos_y * sin_p * cos_r + sin_y * sin_r) * vz)

    # ── Row 1: ∂v_body_z / ∂x ─────────────────────────────────────────
    # v_body_z = R[2,:] · v_world
    H[:, 1, 3] = -sin_p
    H[:, 1, 4] =  cos_p * sin_r
    H[:, 1, 5] =  cos_p * cos_r

    # ∂v_body_z / ∂roll
    H[:, 1, 6] =  cos_p * cos_r * vy - cos_p * sin_r * vz

    # ∂v_body_z / ∂pitch
    H[:, 1, 7] = -cos_p * vx - sin_p * sin_r * vy - sin_p * cos_r * vz

    # ∂v_body_z / ∂yaw is identically zero (v_body_z is yaw-invariant)
    # (already initialized to 0)

    return H


# ─────────────────────────────────────────────────────────────────────────
# NHC update — correct nonlinear innovation
# ─────────────────────────────────────────────────────────────────────────

def nhc_update(state, P, R_nhc, eps=1e-6):
    """
    Apply NHC measurement (v_body_y = 0, v_body_z = 0) to the EKF.

    Args:
        state: (batch, 18)
        P:     (batch, 18, 18)
        R_nhc: scalar or (2,) — measurement noise variance for [y, z]

    Returns:
        state_new, P_new
    """
    batch  = state.shape[0]
    device = state.device

    # Measurement function h(x) → (batch, 3)
    v_body = nhc_measurement_function(state)
    h_x    = v_body[:, 1:3]                        # (batch, 2) — lateral, vertical

    # Measurement target
    z = torch.zeros(batch, 2, device=device)

    # Jacobian
    H = nhc_measurement_jacobian(state)            # (batch, 2, 18)

    # Innovation — correct form: y = z − h(x)
    y = z - h_x                                    # (batch, 2)

    # R as (batch, 2, 2)
    if isinstance(R_nhc, (int, float)):
        R_mat = torch.eye(2, device=device) * float(R_nhc)
        R_mat = R_mat.unsqueeze(0).repeat(batch, 1, 1)
    else:
        R_mat = torch.diag(torch.as_tensor(R_nhc, device=device))
        R_mat = R_mat.unsqueeze(0).repeat(batch, 1, 1)

    # S = H P H^T + R
    H_T = H.transpose(1, 2)
    S = torch.bmm(torch.bmm(H, P), H_T) + R_mat

    # K = P H^T S^{-1}   (via solve, not inverse)
    jitter = torch.eye(2, device=device).unsqueeze(0) * eps
    S_stable = S + jitter
    HP = torch.bmm(H, P)                           # (batch, 2, 18)
    K_T = torch.linalg.solve(S_stable, HP)         # S · K^T = H P
    K = K_T.transpose(1, 2)                        # (batch, 18, 2)

    # state ← state + K y
    state_new = state + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)

    # P ← (I − K H) P
    I = torch.eye(18, device=device).unsqueeze(0)
    P_new = P - torch.bmm(torch.bmm(K, H), P)

    return state_new, P_new