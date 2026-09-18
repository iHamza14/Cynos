#!/usr/bin/env python3
"""Evaluate the velocity GRU+TCN checkpoint against ground-truth blackout speed.

Run from the Cynos repository root:
    python train/evaluate_velocity_gru_tcn.py
Optional:
    python train/evaluate_velocity_gru_tcn.py --checkpoint models/velocity_gru_tcn.pt
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from velocity_dvse_model import VelocityDVSE


FEATURES_PER_AXIS = 6
AXES = 3


def axis_features(x: np.ndarray) -> np.ndarray:
    """x: (600, 3), sampled at 10 Hz.
    Return six statistics per axis per second: (60, 18).
    """
    chunks = x.reshape(60, 10, 3)

    mean = chunks.mean(axis=1)
    std = chunks.std(axis=1)
    maximum = chunks.max(axis=1)
    minimum = chunks.min(axis=1)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))

    centered = chunks - mean[:, None, :]
    skew = np.mean(centered ** 3, axis=1) / (std ** 3 + 1e-8)

    return np.stack(
        [mean, std, maximum, minimum, rms, skew],
        axis=-1,
    ).reshape(60, 18)

def make_features(window: dict) -> np.ndarray:
    accel = axis_features(np.asarray(window["blackout"]["raw_accel"], dtype=np.float32))
    gyro = axis_features(np.asarray(window["blackout"]["raw_gyro"], dtype=np.float32))
    return np.concatenate([accel, gyro], axis=-1).astype(np.float32)


def get_target(window: dict) -> np.ndarray:
    speed = np.asarray(window["ground_truth"]["speeds_ms"], dtype=np.float32)
    if speed.shape[0] == 600:
        speed = speed.reshape(60, 10)[:, -1]
    elif speed.shape[0] != 60:
        raise ValueError(f"Expected 600 or 60 speed samples, got {speed.shape}")
    return speed


def load_windows(path: Path) -> list[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        # Support common wrappers without silently guessing arbitrary schemas.
        for key in ("windows", "data", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of windows in {path}, got {type(data)}")
    return data


class EvalDataset(Dataset):
    def __init__(self, windows: list[dict], mean: np.ndarray, std: np.ndarray):
        self.windows = windows
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        x = (make_features(w) - self.mean) / self.std
        seed = float(w["context"]["vr_seed_ms"])
        y = get_target(w)
        return (
            torch.from_numpy(x),
            torch.tensor([seed], dtype=torch.float32),
            torch.from_numpy(y),
        )


def checkpoint_state(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj
        return obj, {}
    raise TypeError("Checkpoint must be a state_dict or a dict containing one.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/test_blackout_windows.pkl")
    parser.add_argument("--train", default="data/train_blackout_windows.pkl",
                        help="Used only to reconstruct train-only feature normalization.")
    parser.add_argument("--checkpoint", default="models/velocity_gru_tcn.pt")
    parser.add_argument("--out-dir", default="evaluation/velocity_gru_tcn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_windows = load_windows(Path(args.train))
    test_windows = load_windows(Path(args.test))
    # Match training's chronological 80/20 train/validation split: scaler fit
    # on the first 80% of training windows only.
    n_fit = max(1, int(len(train_windows) * 0.8))
    fit_windows = train_windows[:n_fit]
    feature_matrix = np.concatenate([make_features(w) for w in fit_windows], axis=0)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    ds = EvalDataset(test_windows, mean, std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state, metadata = checkpoint_state(ckpt)
    model = VelocityDVSE()
    model.load_state_dict(state)
    model.to(args.device).eval()

    all_pred, all_gt, all_seed = [], [], []
    with torch.no_grad():
        for x, seed, y in loader:
            x = x.to(args.device)
            seed = seed.to(args.device)
            pred, _ = model(x, seed)
            # Expected output is (B, 60) or (B, 60, 1).
            pred = pred.squeeze(-1)
            all_pred.append(pred.cpu().numpy())
            all_gt.append(y.numpy())
            all_seed.append(seed.squeeze(-1).cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    seeds = np.concatenate(all_seed, axis=0)

    err = pred - gt
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(err.mean())
    p80 = float(np.percentile(abs_err, 80))
    seed_pred = np.broadcast_to(seeds[:, None], gt.shape)
    seed_mae = float(np.abs(seed_pred - gt).mean())

    checkpoints = [2, 5, 10, 15, 30, 45, 60]
    print(f"TEST windows: {len(test_windows)}")
    print(f"MAE:  {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"RMSE: {rmse:.4f} m/s ({rmse * 3.6:.2f} km/h)")
    print(f"Bias: {bias:+.4f} m/s ({bias * 3.6:+.2f} km/h)")
    print(f"P80 absolute error: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)")
    print(f"Constant seed MAE: {seed_mae:.4f} m/s ({seed_mae * 3.6:.2f} km/h)")
    print("\nCheckpoint MAE (per-window speed at t):")
    for sec in checkpoints:
        idx = sec - 1
        e = np.abs(pred[:, idx] - gt[:, idx]).mean()
        print(f"  {sec:>2}s: {e:.4f} m/s ({e * 3.6:.2f} km/h)")
    print("\nCumulative MAE (all samples up to t):")
    for sec in checkpoints:
        end = sec
        e = np.abs(pred[:, :end] - gt[:, :end]).mean()
        print(f"  {sec:>2}s: {e:.4f} m/s ({e * 3.6:.2f} km/h)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions_vs_gt.npz",
        pred_ms=pred, gt_ms=gt, seed_ms=seeds, error_ms=err,
    )

    t = np.arange(1, 61)
    mean_pred = pred.mean(axis=0)
    mean_gt = gt.mean(axis=0)
    mean_abs = abs_err.mean(axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, mean_gt * 3.6, label="Ground truth", linewidth=2)
    ax.plot(t, mean_pred * 3.6, label="GRU+TCN prediction", linewidth=2)
    ax.set(title="Mean blackout speed: prediction vs ground truth",
           xlabel="Blackout time (s)", ylabel="Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_speed_gt_vs_pred.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, mean_abs * 3.6, linewidth=2)
    ax.set(title="Mean absolute speed error over blackout",
           xlabel="Blackout time (s)", ylabel="MAE (km/h)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_over_time.png", dpi=160)
    plt.close(fig)

    # Plot a few representative windows, including the largest integrated error.
    per_window_mae = abs_err.mean(axis=1)
    selected = [int(np.argmin(per_window_mae)), int(np.median(np.argsort(per_window_mae))),
                int(np.argmax(per_window_mae))]
    names = ["lowest_error", "median_rank", "highest_error"]
    for idx, name in zip(selected, names):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(t, gt[idx] * 3.6, label="Ground truth", linewidth=2)
        ax.plot(t, pred[idx] * 3.6, label="Prediction", linewidth=2)
        ax.axhline(seeds[idx] * 3.6, linestyle="--", label="Seed speed")
        ax.set(title=f"Test window {idx} ({name}), MAE={per_window_mae[idx] * 3.6:.2f} km/h",
               xlabel="Blackout time (s)", ylabel="Speed (km/h)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"window_{idx}_{name}.png", dpi=160)
        plt.close(fig)

    print(f"\nSaved evaluation outputs to: {out_dir}")
    print("  predictions_vs_gt.npz")
    print("  mean_speed_gt_vs_pred.png")
    print("  mae_over_time.png")
    print("  representative window plots")


if __name__ == "__main__":
    main()
