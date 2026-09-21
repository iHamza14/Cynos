import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from models import DVSEModel
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    """Executes core logic for set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path):
    """Executes core logic for load_pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


class DVSEDataset(Dataset):
    """Enterprise class definition for DVSEDataset."""

    def __init__(self, items, scalers, hz=10):
        """Initializes the instance."""
        self.items = items
        self.acc_scaler = scalers["raw_accel"]
        self.gyro_scaler = scalers["raw_gyro"]
        self.hz = hz

    def __len__(self):
        """Executes core logic for __len__."""
        return len(self.items)

    def __getitem__(self, idx):
        """Executes core logic for __getitem__."""
        item = self.items[idx]

        raw_accel = item["blackout"]["raw_accel"]
        raw_gyro = item["blackout"]["raw_gyro"]

        acc_scaled = self.acc_scaler.transform(raw_accel)
        gyro_scaled = self.gyro_scaler.transform(raw_gyro)

        speeds = item["ground_truth"]["speeds_ms"]  # [600]
        vr_seed = item["context"]["vr_seed_ms"]

        # Calculate 1Hz targets and V_r sequence
        T = len(speeds) // self.hz
        target_speeds = np.zeros(T, dtype=np.float32)
        target_delta_v = np.zeros(T, dtype=np.float32)
        vr_seq = np.zeros((T, 1), dtype=np.float32)

        v_prev = vr_seed
        for i in range(T):
            target_speeds[i] = speeds[(i + 1) * self.hz - 1]  # speed at the end of the second
            target_delta_v[i] = target_speeds[i] - v_prev
            vr_seq[i, 0] = v_prev
            v_prev = target_speeds[i]

        return {
            "acc_window": torch.tensor(acc_scaled, dtype=torch.float32),
            "gyro_window": torch.tensor(gyro_scaled, dtype=torch.float32),
            "vr_seq": torch.tensor(vr_seq, dtype=torch.float32),
            "v_0": torch.tensor([vr_seed], dtype=torch.float32),
            "target_speeds": torch.tensor(target_speeds, dtype=torch.float32),
            "target_delta_v": torch.tensor(target_delta_v, dtype=torch.float32),
        }


def smooth_l1_loss(pred, target):
    """Executes core logic for smooth_l1_loss."""
    return nn.functional.smooth_l1_loss(pred, target)


def dvse_loss(pred_delta_v, pred_v, target_delta_v, target_v, lambda_w=0.7):
    """Executes core logic for dvse_loss."""
    # L_velocity = 0.7 L_Delta v + 0.3 L_v
    l_delta_v = smooth_l1_loss(pred_delta_v, target_delta_v)
    l_v = smooth_l1_loss(pred_v, target_v)
    return lambda_w * l_delta_v + (1.0 - lambda_w) * l_v, l_delta_v, l_v


def evaluate(model, loader, device):
    """Executes core logic for evaluate."""
    model.eval()
    losses, dv_losses, v_losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            acc_window = batch["acc_window"].to(device)
            gyro_window = batch["gyro_window"].to(device)
            vr_seq = batch["vr_seq"].to(device)
            v_0 = batch["v_0"].to(device)
            target_speeds = batch["target_speeds"].to(device)
            target_delta_v = batch["target_delta_v"].to(device)

            delta_v, v, N, angles = model(acc_window, gyro_window, vr_seq, v_0=v_0)

            loss, l_dv, l_v = dvse_loss(delta_v, v, target_delta_v, target_speeds)
            losses.append(loss.item())
            dv_losses.append(l_dv.item())
            v_losses.append(l_v.item())

    return np.mean(losses), np.mean(dv_losses), np.mean(v_losses)


def main():
    """Executes core logic for main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dvse_output"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and just evaluate the best saved model",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    train_path = args.data_dir / "train_blackout_windows.pkl"
    test_path = args.data_dir / "test_blackout_windows.pkl"
    scalers_path = args.data_dir / "scalers.pkl"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing data files. Run build script first.")

    train_items = load_pickle(train_path)
    test_items = load_pickle(test_path)
    scalers = load_pickle(scalers_path)

    train_loader = DataLoader(
        DVSEDataset(train_items, scalers), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        DVSEDataset(test_items, scalers), batch_size=args.batch_size, shuffle=False
    )

    model = DVSEModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_path = args.output_dir / "best_dvse.pt"
    if args.eval_only:
        print(f"Loading best model from {best_path} for evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        val_loss, val_dv, val_v = evaluate(model, test_loader, device)
        print(
            f"\n[Evaluation Results] Test Loss (Total MAE): {val_loss:.4f} m/s | Delta-V MAE: {val_dv:.4f} m/s | Absolute Velocity MAE: {val_v:.4f} m/s"
        )
        return

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            acc_window = batch["acc_window"].to(device)
            gyro_window = batch["gyro_window"].to(device)
            vr_seq = batch["vr_seq"].to(device)
            v_0 = batch["v_0"].to(device)
            target_speeds = batch["target_speeds"].to(device)
            target_delta_v = batch["target_delta_v"].to(device)

            optimizer.zero_grad(set_to_none=True)

            delta_v, v, N, angles = model(acc_window, gyro_window, vr_seq, v_0=v_0)

            loss, l_dv, l_v = dvse_loss(delta_v, v, target_delta_v, target_speeds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        val_loss, val_dv, val_v = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} - Train Loss: {np.mean(epoch_losses):.4f} - "
            f"Val Loss: {val_loss:.4f} (dV: {val_dv:.4f}, V: {val_v:.4f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"Training complete. Best model saved to {best_path}")


if __name__ == "__main__":
    main()
