
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DeadReckoningDataset
from models import NoiseNetwork


def sliding_loss(pred_dv, target_dv):
    """Both tensors have shape (B, T, 1)."""

    def sequence_loss(pred, target):
        step_loss = F.smooth_l1_loss(pred, target)

        cum_loss = F.smooth_l1_loss(
            torch.cumsum(pred, dim=1),
            torch.cumsum(target, dim=1),
        )

        return 0.7 * step_loss + 0.3 * cum_loss

    loss_0 = sequence_loss(pred_dv, target_dv)

    if pred_dv.size(1) <= 1:
        return loss_0

    loss_1 = sequence_loss(
        pred_dv[:, 1:, :],
        target_dv[:, :-1, :],
    )

    return torch.minimum(loss_0, loss_1)


def train_stage1():
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset = DeadReckoningDataset(
        "data/train_blackout_windows.pkl",
        "data/scalers.pkl",
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = NoiseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 40
    noise_sigma = 0.5

    print("=== Starting Stage 1: Noise Network (GRU) ===")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            (
                raw_accel_t,
                raw_gyro_t,
                gravity_t,
                vr_train,
                vr_seed,
                delta_v_seq,
                target_disp,
                heading_target,
                start_heading,
            ) = batch

            # Move tensors to the selected device.
            raw_accel_t = raw_accel_t.to(device)
            raw_gyro_t = raw_gyro_t.to(device)
            gravity_t = gravity_t.to(device)

            vr_train = vr_train.to(device)
            vr_seed = vr_seed.to(device)
            delta_v_seq = delta_v_seq.to(device)

            # Shapes:
            # vr_train: (B, T, 1)
            # vr_seed:  (B, 1)
            # delta_v:  (B, T, 1)

            # Construct previous-velocity input:
            # t=0: context velocity seed
            # t>0: ground-truth velocity at t-1
            vr_seed = vr_seed.unsqueeze(1)  # (B, 1, 1)

            vr_input = torch.cat(
                [vr_seed, vr_train[:, :-1, :]],
                dim=1,
            )

            # Gaussian noise augmentation.
            vr_noisy = (
                vr_input
                + torch.randn_like(vr_input) * noise_sigma
            )

            optimizer.zero_grad(set_to_none=True)

            # Preserve the original model interface.
            delta_v_pred, _ = model(
                                raw_accel_t,
                                raw_gyro_t,
                                vr_noisy,
                            )

            if delta_v_pred.shape != delta_v_seq.shape:
                raise ValueError(
                    f"Prediction shape: {delta_v_pred.shape}, "
                    f"target shape: {delta_v_seq.shape}"
                )

            loss = sliding_loss(
                delta_v_pred,
                delta_v_seq,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch + 1}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = raw_accel_t.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {avg_loss:.6f}"
        )

    checkpoint_path = "checkpoints/stage1_noise_net.pth"

    torch.save(model.state_dict(), checkpoint_path)

    print(f"Stage 1 complete. Saved {checkpoint_path}")


if __name__ == "__main__":
    train_stage1()