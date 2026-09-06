import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DeadReckoningDataset
from models import DVSE


# ============================================================
# Loss
# ============================================================

def loss_fn(
    delta_v_pred,
    euler_pred,
    delta_v_target,
    vr_seed,
    target_disp,
    heading_target_rad,
    lambda1=1.0,
    lambda2=0.5,
    lambda3=0.1
):
    # --------------------------------------------------------
    # 1. Primary: per-step delta_v regression
    # --------------------------------------------------------
    loss_dv = F.mse_loss(
        delta_v_pred,
        delta_v_target
    )

    # --------------------------------------------------------
    # 2. End-to-end displacement
    # --------------------------------------------------------
    heading_rad = euler_pred[..., 2]  # (B, T)

    # delta_v_pred: (B, T, 1)
    # squeeze -> (B, T)
    v_cum = (
        vr_seed
        + torch.cumsum(
            delta_v_pred.squeeze(-1),
            dim=1
        )
    )

    delta_north = (
        v_cum * torch.cos(heading_rad)
    ).sum(dim=1)

    delta_east = (
        v_cum * torch.sin(heading_rad)
    ).sum(dim=1)

    pred_disp = torch.stack(
        [delta_north, delta_east],
        dim=-1
    )

    loss_disp = F.mse_loss(
        pred_disp,
        target_disp
    )

    # --------------------------------------------------------
    # 3. Euler / yaw regularization
    # --------------------------------------------------------
    loss_euler = F.mse_loss(
        heading_rad,
        heading_target_rad
    )

    total_loss = (
        lambda1 * loss_dv
        + lambda2 * loss_disp
        + lambda3 * loss_euler
    )

    return (
        total_loss,
        loss_dv,
        loss_disp,
        loss_euler
    )


# ============================================================
# Move batch to device
# ============================================================

def move_batch_to_device(batch, device):
    return tuple(
        x.to(device, non_blocking=True)
        if torch.is_tensor(x)
        else x
        for x in batch
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader,
    device,
    scaler=None
):
    model.eval()

    total_loss = 0.0
    total_dv = 0.0
    total_disp = 0.0
    total_euler = 0.0

    for batch in test_loader:

        (
            acc_t,
            gyro_t,
            mtn_t,
            raw_t,
            grav_t,
            vr_train,
            vr_seed,
            delta_v_seq,
            target_disp,
            heading_target
        ) = move_batch_to_device(batch, device)

        delta_v_pred, euler_pred = model(
            acc_t,
            gyro_t,
            vr_train,
            mtn_t,
            raw_t,
            grav_t
        )

        (
            loss,
            l_dv,
            l_disp,
            l_eu
        ) = loss_fn(
            delta_v_pred,
            euler_pred,
            delta_v_seq,
            vr_seed,
            target_disp,
            heading_target
        )

        total_loss += loss.item()
        total_dv += l_dv.item()
        total_disp += l_disp.item()
        total_euler += l_eu.item()

    n = len(test_loader)

    return (
        total_loss / n,
        total_dv / n,
        total_disp / n,
        total_euler / n
    )


# ============================================================
# Training
# ============================================================

def train():

    # --------------------------------------------------------
    # CUDA setup
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

        # Faster for fixed-size inputs
        torch.backends.cudnn.benchmark = True

        # Allows TF32 on supported NVIDIA GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    scaler_path = "data/scalers.pkl"

    train_dataset = DeadReckoningDataset(
        "data/train_windows.pkl",
        scaler_path
    )

    test_dataset = DeadReckoningDataset(
        "data/test_windows.pkl",
        scaler_path
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    # Increase this if your GPU has plenty of VRAM.
    batch_size = 16

    # More workers usually improves GPU utilization.
    num_workers = min(
        4,
        os.cpu_count() or 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Batch size:    {batch_size}")
    print(f"Workers:       {num_workers}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DVSE().to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # --------------------------------------------------------
    # Training config
    # --------------------------------------------------------

    epochs = 60

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_dv = 0.0
        total_disp = 0.0
        total_euler = 0.0

        for batch in train_loader:

            (
                acc_t,
                gyro_t,
                mtn_t,
                raw_t,
                grav_t,
                vr_train,
                vr_seed,
                delta_v_seq,
                target_disp,
                heading_target
            ) = move_batch_to_device(
                batch,
                device
            )

            # ------------------------------------------------
            # Data augmentation
            # ------------------------------------------------

            vr_noisy = (
                vr_train
                + torch.randn_like(vr_train) * 0.5
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass with AMP
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):

                delta_v_pred, euler_pred = model(
                    acc_t,
                    gyro_t,
                    vr_noisy,
                    mtn_t,
                    raw_t,
                    grav_t
                )

                (
                    loss,
                    l_dv,
                    l_disp,
                    l_eu
                ) = loss_fn(
                    delta_v_pred,
                    euler_pred,
                    delta_v_seq,
                    vr_seed,
                    target_disp,
                    heading_target
                )

            # ------------------------------------------------
            # Backward pass
            # ------------------------------------------------

            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            total_loss += loss.item()
            total_dv += l_dv.item()
            total_disp += l_disp.item()
            total_euler += l_eu.item()

        # ----------------------------------------------------
        # Average training metrics
        # ----------------------------------------------------

        n_train = len(train_loader)

        train_loss = total_loss / n_train
        train_dv = total_dv / n_train
        train_disp = total_disp / n_train
        train_euler = total_euler / n_train

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_dv,
            val_disp,
            val_euler
        ) = evaluate(
            model,
            test_loader,
            device
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss
                },
                "checkpoints/dvse_dead_reckoning_best.pth"
            )

            best_marker = " *BEST*"

        else:
            best_marker = ""

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"DV: {val_dv:.6f} | "
            f"Disp: {val_disp:.6f} | "
            f"Yaw: {val_euler:.6f}"
            f"{best_marker}"
        )

        # ----------------------------------------------------
        # CUDA memory information
        # ----------------------------------------------------

        if device.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated(device)
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved(device)
                / 1024**3
            )

            print(
                f"           GPU memory: "
                f"{allocated:.2f} GB allocated | "
                f"{reserved:.2f} GB reserved"
            )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        "checkpoints/dvse_dead_reckoning.pth"
    )

    print("\nTraining Complete.")
    print(
        "Final model: "
        "checkpoints/dvse_dead_reckoning.pth"
    )
    print(
        "Best model:  "
        "checkpoints/dvse_dead_reckoning_best.pth"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()