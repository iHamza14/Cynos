import torch
import torch.nn as nn


# ============================================================
# Noise Network
# ============================================================

class NoiseNetwork(nn.Module):
    """
    Learns the residual / noise affecting the physics-based
    delta-v estimate.

    Inputs:
        acc_feat  : (B, T, 18)
        gyro_feat : (B, T, 18)
        vr        : (B, T, 1)

    Output:
        noise     : (B, T, 1)
    """

    def __init__(self):
        super().__init__()

        # Accelerometer feature branch
        self.acc_fc1 = nn.Linear(18, 32)
        self.acc_fc2 = nn.Linear(32, 64)

        # Gyroscope feature branch
        self.gyro_fc1 = nn.Linear(18, 32)
        self.gyro_fc2 = nn.Linear(32, 64)

        # Temporal modeling
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        # VR is concatenated AFTER the GRU
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, acc_feat, gyro_feat, vr):

        # ----------------------------------------------------
        # Accelerometer features
        # ----------------------------------------------------

        a = torch.relu(
            self.acc_fc1(acc_feat)
        )

        a = torch.relu(
            self.acc_fc2(a)
        )

        # ----------------------------------------------------
        # Gyroscope features
        # ----------------------------------------------------

        g = torch.relu(
            self.gyro_fc1(gyro_feat)
        )

        g = torch.relu(
            self.gyro_fc2(g)
        )

        # ----------------------------------------------------
        # Combine accelerometer + gyro
        # ----------------------------------------------------

        x = torch.cat(
            [a, g],
            dim=-1
        )
        # (B, T, 128)

        # ----------------------------------------------------
        # Temporal processing
        # ----------------------------------------------------

        out, _ = self.gru(x)
        # (B, T, 128)

        # ----------------------------------------------------
        # Add odometry / velocity seed
        # ----------------------------------------------------

        out = torch.cat(
            [out, vr],
            dim=-1
        )
        # (B, T, 129)

        # ----------------------------------------------------
        # Noise prediction head
        # ----------------------------------------------------

        out = torch.relu(
            self.fc1(out)
        )

        out = torch.relu(
            self.fc2(out)
        )

        out = torch.relu(
            self.fc3(out)
        )

        noise = self.fc4(out)

        return noise


# ============================================================
# TCN
# ============================================================

class TCN(nn.Module):
    """
    Temporal convolution network used by MTN.

    Input:
        (B, T, C)

    Output:
        (B, T, out_channels)
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            64,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # x: (B, T, C)
        # Conv1d expects: (B, C, T)
        # ----------------------------------------------------

        x = x.transpose(1, 2)

        x = torch.relu(
            self.conv1(x)
        )

        x = torch.relu(
            self.conv2(x)
        )

        # Back to (B, T, C)
        x = x.transpose(1, 2)

        return x


# ============================================================
# MTN
# ============================================================

class MTN(nn.Module):
    """
    Motion / orientation estimation network.

    Input:
        mtn_input: (B, T, 6)

    Output:
        Euler angles: (B, T, 3)
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            6,
            32
        )

        self.fc2 = nn.Linear(
            32,
            64
        )

        self.tcn = TCN(
            64,
            64
        )

        self.fc3 = nn.Linear(
            64,
            3
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        # ----------------------------------------------------
        # Temporal convolution
        # ----------------------------------------------------

        x = self.tcn(x)

        # ----------------------------------------------------
        # Euler angles
        # ----------------------------------------------------

        euler = self.fc3(x)

        return euler


# ============================================================
# Euler → Rotation Matrix
# ============================================================

def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrices.

    Input:
        euler_angles: (B, T, 3)

    Output:
        R: (B, T, 3, 3)
    """

    B, T, _ = euler_angles.shape

    R = torch.zeros(
        B,
        T,
        3,
        3,
        device=euler_angles.device,
        dtype=euler_angles.dtype
    )

    alpha = euler_angles[..., 0]
    beta = euler_angles[..., 1]
    gamma = euler_angles[..., 2]

    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    cb = torch.cos(beta)
    sb = torch.sin(beta)

    cg = torch.cos(gamma)
    sg = torch.sin(gamma)

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R[..., 0, 0] = ca * cb

    R[..., 0, 1] = (
        ca * sb * sg
        - sa * cg
    )

    R[..., 0, 2] = (
        ca * sb * cg
        + sa * sg
    )

    R[..., 1, 0] = (
        sa * cb
    )

    R[..., 1, 1] = (
        sa * sb * sg
        + ca * cg
    )

    R[..., 1, 2] = (
        sa * sb * cg
        - ca * sg
    )

    R[..., 2, 0] = -sb

    R[..., 2, 1] = (
        cb * sg
    )

    R[..., 2, 2] = (
        cb * cg
    )

    return R


# ============================================================
# DVSE
# ============================================================

class DVSE(nn.Module):

    def __init__(self):
        super().__init__()

        self.noise_net = NoiseNetwork()

        self.mtn = MTN()

    def forward(
        self,
        acc_feat,
        gyro_feat,
        vr,
        mtn_input,
        raw_accel,
        gravity
    ):
        """
        Inputs:

            acc_feat:
                (B, T, 18)

            gyro_feat:
                (B, T, 18)

            vr:
                (B, T, 1)

            mtn_input:
                (B, T, 6)

            raw_accel:
                (B, T, 3)

            gravity:
                (B, T, 3)

        Outputs:

            delta_v:
                (B, T, 1)

            euler:
                (B, T, 3)
        """

        # ====================================================
        # 1. Learned noise
        # ====================================================

        N = self.noise_net(
            acc_feat,
            gyro_feat,
            vr
        )

        # ====================================================
        # 2. Estimate Euler angles
        # ====================================================

        euler = self.mtn(
            mtn_input
        )

        # ====================================================
        # 3. Euler → rotation matrix
        # ====================================================

        R_pv = euler_to_rotation_matrix(
            euler
        )

        # ====================================================
        # 4. Remove gravity
        # ====================================================

        a_diff = (
            raw_accel - gravity
        )

        # (B, T, 3)
        # →
        # (B, T, 3, 1)

        a_diff = a_diff.unsqueeze(-1)

        # ====================================================
        # 5. Rotate acceleration
        # ====================================================

        a_v = torch.matmul(
            R_pv,
            a_diff
        ).squeeze(-1)

        # (B, T, 3)

        # ====================================================
        # 6. Vehicle forward acceleration
        # ====================================================

        a_forward = a_v[..., 1:2]

        # ====================================================
        # 7. Physics + learned residual
        # ====================================================

        delta_v = (
            a_forward + N
        )

        return delta_v, euler