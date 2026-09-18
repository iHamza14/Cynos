import torch
from models.rotation_utils import euler_to_rotation_matrix


def wrap_angle_rad(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


class DeterministicEKF:
    """
    Math convention. yaw: 0 = +X axis, increases CCW.
    State: [px, py, vx, vy, yaw, b_gz, b_ax, b_ay]
    """

    def __init__(self, roll, pitch, dt=0.1, device='cpu', dtype=torch.float32):
        self.dt = dt
        self.roll = torch.tensor(roll, dtype=dtype, device=device)
        self.pitch = torch.tensor(pitch, dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype

        self.x = torch.zeros(8, dtype=dtype, device=device)

        # P: raise yaw and gyro-bias uncertainty to realistic values
        self.P = torch.diag(torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.25, 1e-2, 1.0, 1.0],
            dtype=dtype, device=device,
        ))

        # Q: yaw q reflects gyro noise (σ ≈ 0.28 rad/s → 8e-4/step²).
        # Gyro bias q allows adaptation within ~10s.
        # Accel bias q allows adaptation within ~30s.
        self.Q = torch.diag(torch.tensor(
            [1e-4, 1e-4, 1e-2, 1e-2, 1e-3, 1e-6, 1e-6, 1e-6],
            dtype=dtype, device=device,
        ))

    def predict(self, accel_body, gyro_body):
        dt = self.dt
        a = torch.as_tensor(accel_body, dtype=self.dtype, device=self.device)
        g = torch.as_tensor(gyro_body, dtype=self.dtype, device=self.device)

        # Attitude is frozen at construction. Do NOT update from accel:
        #  - vehicle braking corrupts accel-derived pitch
        #  - gyro axes for roll/pitch are unusable on this dataset
        #  - phone is mounted, so roll/pitch don't change during a 60s window

        px, py, vx, vy, yaw, b_gz, b_ax, b_ay = self.x

        yaw_new = wrap_angle_rad(yaw + (g[1] - b_gz) * dt)

        R = euler_to_rotation_matrix(self.roll, self.pitch, yaw_new)
        gravity = torch.tensor([0.0, 0.0, 9.81],
                                dtype=self.dtype, device=self.device)

        # Bias correction in body frame, then rotate
        b_body = torch.stack([
            b_ax, b_ay,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        ])
        a_corrected = a - b_body
        a_world = R @ a_corrected - gravity

        vx_new = vx + a_world[0] * dt
        vy_new = vy + a_world[1] * dt
        px_new = px + vx * dt + 0.5 * a_world[0] * dt * dt
        py_new = py + vy * dt + 0.5 * a_world[1] * dt * dt

        self.x = torch.stack([px_new, py_new, vx_new, vy_new,
                              yaw_new, b_gz, b_ax, b_ay])

        # Jacobian
        F = torch.eye(8, dtype=self.dtype, device=self.device)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = -dt                                  # ∂yaw/∂b_gz = -dt

        # ∂v/∂b_a: accel bias enters world velocity through R
        F[2, 6] = -R[0, 0] * dt
        F[2, 7] = -R[0, 1] * dt
        F[3, 6] = -R[1, 0] * dt
        F[3, 7] = -R[1, 1] * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_yaw(self, yaw_meas_math, R_yaw):
        H = torch.zeros((1, 8), dtype=self.dtype, device=self.device)
        H[0, 4] = 1.0
        z = torch.tensor([yaw_meas_math],
                         dtype=self.dtype, device=self.device)
        y = z - self.x[4].reshape(1)
        y[0] = wrap_angle_rad(y[0])
        S = H @ self.P @ H.T + R_yaw
        K = self.P @ H.T / S
        self.x = self.x + (K @ y).flatten()
        self.P = (torch.eye(8, dtype=self.dtype, device=self.device)
                  - K @ H) @ self.P

    def update_nhc(self, R_nhc=2.0):
        vx, vy, yaw = self.x[2], self.x[3], self.x[4]
        h_x = -vx * torch.sin(yaw) + vy * torch.cos(yaw)

        H = torch.zeros((1, 8), dtype=self.dtype, device=self.device)
        H[0, 2] = -torch.sin(yaw)
        H[0, 3] = torch.cos(yaw)
        H[0, 4] = -(vx * torch.cos(yaw) + vy * torch.sin(yaw))

        z = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        y = z - h_x.reshape(1)
        S = H @ self.P @ H.T + R_nhc
        K = self.P @ H.T / S
        self.x = self.x + (K @ y).flatten()
        self.P = (torch.eye(8, dtype=self.dtype, device=self.device)
                  - K @ H) @ self.P