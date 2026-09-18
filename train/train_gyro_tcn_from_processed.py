#!/usr/bin/env python3
"""
Train a causal TCN gyro denoiser using the already-generated blackout pickle files.

Expected files (created by the user's preprocessing script):
  data/train_blackout_windows.pkl
  data/test_blackout_windows.pkl
  data/scalers.pkl (not required; this script fits its own train-only gyro scaler)

Each item is expected to contain:
  item["blackout"]["raw_gyro"]       shape (600, 3), at 10 Hz
  item["ground_truth"]["headings_deg"] shape (600,)

The preprocessing output must save `ground_truth["yaw_rate_dps"]`, already using
the project convention `-vehicle_yaw_rate_dps`. This script uses that field directly
and does not derive or negate it again.

Run from the project directory:
  pip install numpy pandas scikit-learn matplotlib torch
  python train_gyro_tcn_from_processed.py

Optional:
  python train_gyro_tcn_from_processed.py --data-dir data --epochs 50
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


HZ = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--train-file", default="train_blackout_windows.pkl")
    p.add_argument("--test-file", default="test_blackout_windows.pkl")
    p.add_argument("--output-dir", type=Path, default=Path("gyro_tcn_output"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--heading-loss-weight", type=float, default=0.25)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-dt", type=float, default=0.25)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def wrap_degrees(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0

def get_target_rate(item):
    """
    Use the synchronized vehicle yaw rate saved by preprocessing.

    Preprocessing already applies the project's convention:
        target = -vehicle_yaw_rate_dps

    Do not negate this value again.
    """
    gt = item["ground_truth"]

    if "yaw_rate_dps" not in gt:
        raise KeyError(
            "Missing ground_truth['yaw_rate_dps']. "
            "Regenerate the processed pickle files using the updated "
            "preprocessing script. Refusing to derive a potentially "
            "sign-inconsistent target from heading."
        )

    rate = np.asarray(gt["yaw_rate_dps"], dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(rate)):
        raise ValueError("Non-finite values in ground_truth['yaw_rate_dps'].")

    return rate


def extract_windows(items):
    xs, ys, times = [], [], []
    skipped = 0
    for idx, item in enumerate(items):
        try:
            gyro = np.asarray(item["blackout"]["raw_gyro"], dtype=float)
            target = get_target_rate(item)
            if gyro.ndim != 2 or gyro.shape[1] != 3 or len(gyro) != len(target):
                raise ValueError("raw_gyro and target shapes do not match")
            if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(target)):
                raise ValueError("non-finite gyro or target")
            xs.append(gyro.astype(np.float32))
            ys.append(target.astype(np.float32))
            meta = item.get("metadata", {})
            times.append(float(meta.get("blackout_start_sec", idx)))
        except (KeyError, TypeError, ValueError) as e:
            skipped += 1
            print(f"Skipping window {idx}: {e}")
    if not xs:
        raise RuntimeError("No valid windows found in processed pickle.")
    return np.stack(xs), np.stack(ys), np.asarray(times), skipped


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

from models import GyroTCN


def integrated_heading_loss(pred, target, heading_weight):
    # Rates are deg/s. Integrate at 10 Hz to compare relative heading changes.
    dt = 1.0 / HZ
    pred_delta = torch.cumsum(pred * dt, dim=1)
    target_delta = torch.cumsum(target * dt, dim=1)
    heading_loss = nn.functional.smooth_l1_loss(pred_delta, target_delta)
    rate_loss = nn.functional.smooth_l1_loss(pred, target)
    return rate_loss + heading_weight * heading_loss, rate_loss, heading_loss


def evaluate(model, loader, device, heading_weight):
    model.eval()
    losses, rate_losses, heading_losses = [], [], []
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss, rl, hl = integrated_heading_loss(pred, yb, heading_weight)
            losses.append(loss.item())
            rate_losses.append(rl.item())
            heading_losses.append(hl.item())
            all_pred.append(pred.cpu().numpy())
            all_true.append(yb.cpu().numpy())
    return (
        float(np.mean(losses)),
        float(np.mean(rate_losses)),
        float(np.mean(heading_losses)),
        np.concatenate(all_pred),
        np.concatenate(all_true),
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    train_path = args.data_dir / args.train_file
    test_path = args.data_dir / args.test_file
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected processed files:\n  {train_path}\n  {test_path}\n"
            "Run your preprocessing script first, or pass --data-dir."
        )

    train_items = load_pickle(train_path)
    test_items = load_pickle(test_path)
    x_train_raw, y_train, train_times, skipped_train = extract_windows(train_items)
    x_test_raw, y_test, test_times, skipped_test = extract_windows(test_items)

    # Fit feature scaling on training windows only.
    scaler = StandardScaler()
    scaler.fit(x_train_raw.reshape(-1, 3))
    x_train = scaler.transform(x_train_raw.reshape(-1, 3)).reshape(x_train_raw.shape)
    x_test = scaler.transform(x_test_raw.reshape(-1, 3)).reshape(x_test_raw.shape)

    # Normalize target using training target only; convert back for reporting.
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test_n = ((y_test - y_mean) / y_std).astype(np.float32)

    train_loader = DataLoader(
        WindowDataset(x_train, y_train_n), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers
    )
    test_loader = DataLoader(
        WindowDataset(x_test, y_test_n), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers
    )

    model = GyroTCN(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val = float("inf")
    best_path = args.output_dir / "best_gyro_tcn_processed.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss, _, _ = integrated_heading_loss(
                pred, yb, args.heading_loss_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val, val_rate, val_heading, _, _ = evaluate(
            model, test_loader, device, args.heading_loss_weight
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={np.mean(epoch_losses):.5f} "
            f"heldout={val:.5f} rate={val_rate:.5f} heading={val_heading:.5f}"
        )
        if val < best_val:
            best_val = val
            torch.save({
                "model_state": model.state_dict(),
                "input_mean": scaler.mean_.tolist(),
                "input_scale": scaler.scale_.tolist(),
                "target_mean": y_mean,
                "target_std": y_std,
                "hz": HZ,
                "input_axes": ["gyro_yaw", "gyro_pitch", "gyro_roll"],
                "target_source": "ground_truth.yaw_rate_dps (pre-negated -vehicle yaw rate, deg/s)",
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val, val_rate, val_heading, pred_n, true_n = evaluate(
        model, test_loader, device, args.heading_loss_weight
    )
    pred = pred_n * y_std + y_mean
    true = true_n * y_std + y_mean

    # Per-window integrated heading-change errors at the requested horizons.
    metric_rows = []
    for horizon in (1, 5, 10, 30, 60):
        n_samples = min(int(round(horizon * HZ)), pred.shape[1])
        errors = []
        for w in range(len(pred)):
            pred_change = np.cumsum(pred[w] / HZ)
            true_change = np.cumsum(true[w] / HZ)
            errors.append(abs(pred_change[n_samples - 1] - true_change[n_samples - 1]))
        metric_rows.append({
            "horizon_s": horizon,
            "N_windows": len(errors),
            "MAE_heading_change_deg": float(np.mean(errors)),
            "median_abs_error_deg": float(np.median(errors)),
            "P90_abs_error_deg": float(np.percentile(errors, 90)),
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "heldout_heading_metrics.csv", index=False)

    rows = []
    for wi in range(len(pred)):
        for ti in range(pred.shape[1]):
            rows.append({
                "window_index": wi,
                "blackout_start_sec": test_times[wi],
                "sample_in_blackout": ti,
                "time_into_blackout_s": ti / HZ,
                "reference_yaw_rate_deg_s": true[wi, ti],
                "tcn_yaw_rate_deg_s": pred[wi, ti],
                "reference_heading_change_deg": np.cumsum(true[wi] / HZ)[ti],
                "tcn_heading_change_deg": np.cumsum(pred[wi] / HZ)[ti],
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "heldout_predictions.csv", index=False)

    # Plot the first held-out episode; no vehicle signal is fed to the model.
    wi = 0
    tt = np.arange(pred.shape[1]) / HZ
    pred_heading = np.cumsum(pred[wi] / HZ)
    true_heading = np.cumsum(true[wi] / HZ)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # Raw phone yaw gyro is stored in rad/s. Convert to deg/s and apply
    # the same sign convention as the supervised target (-vehicle yaw rate).
    noisy_yaw = -np.rad2deg(x_test_raw[wi, :, 0])
    axes[0].plot(
        tt, noisy_yaw, label="noisy phone yaw rate",
        alpha=0.28, linewidth=1.0, linestyle="--"
    )
    axes[0].plot(tt, true[wi], label="vehicle reference yaw rate")
    axes[0].plot(tt, pred[wi], label="TCN denoised phone yaw rate")
    axes[0].set_ylabel("Yaw rate (deg/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(tt, true_heading, label="reference heading change")
    axes[1].plot(tt, pred_heading, label="TCN integrated heading")
    axes[1].set_ylabel("Heading change (deg)")
    axes[1].set_xlabel("Time into blackout (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Held-out processed blackout episode: causal TCN")
    fig.tight_layout()
    fig.savefig(args.output_dir / "heldout_tcn_denoising.png", dpi=160)
    plt.close(fig)

    print(f"\nTrain windows: {len(x_train)} (skipped {skipped_train})")
    print(f"Test windows:  {len(x_test)} (skipped {skipped_test})")
    print(f"Best model: {best_path}")
    print(f"Metrics: {args.output_dir / 'heldout_heading_metrics.csv'}")
    print(f"Predictions: {args.output_dir / 'heldout_predictions.csv'}")
    print(f"Plot: {args.output_dir / 'heldout_tcn_denoising.png'}")
    print(metrics_df.to_string(index=False))
    print(
        "\nNOTE: Targets are read directly from ground_truth['yaw_rate_dps'] "
        "using the pre-negated -vehicle yaw-rate convention. The plot overlays the "
        "raw phone yaw gyro (converted from rad/s to deg/s) with transparency."
    )


if __name__ == "__main__":
    main()
