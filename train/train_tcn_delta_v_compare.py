#!/usr/bin/env python3
"""Evaluate a trained causal TCN for blackout Δv against a zero-Δv baseline.

Expected files:
  data/test_blackout_windows.pkl
  data/scalers.pkl
  data/tcn_delta_v_best.pt

Run from project root:
  python train/evaluate_tcn_delta_v.py
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = [2, 5, 10, 15, 30, 60]
HZ = 10


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


class Episodes(Dataset):
    def __init__(self, windows, scalers):
        self.windows = windows
        self.scalers = scalers

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        x = np.concatenate([
            self.scalers[k].transform(np.asarray(w["blackout"][k]))
            for k in ("raw_accel", "raw_gyro", "gravity")
        ], axis=1).astype(np.float32)
        dv = np.asarray(w["ground_truth"]["delta_v_ms"], dtype=np.float32)
        if x.shape != (600, 9):
            raise ValueError(f"Expected feature shape (600,9), got {x.shape}")
        if dv.shape != (600,):
            raise ValueError(f"Expected delta_v_ms shape (600,), got {dv.shape}")
        seed = np.float32(w["context"]["vr_seed_ms"])
        return torch.from_numpy(x.T), torch.from_numpy(dv), torch.tensor(seed)


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, 3, dilation), nn.GELU(), nn.Dropout(.1),
            CausalConv1d(channels, channels, 3, dilation), nn.GELU(), nn.Dropout(.1),
        )

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Conv1d(9, 64, 1)
        self.blocks = nn.Sequential(*(Block(64, d) for d in (1, 2, 4, 8, 16, 32, 64)))
        self.out = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        return self.out(self.blocks(self.in_proj(x))).squeeze(1)


def evaluate(model, loader, outdir, examples_to_plot=3):
    model.eval()
    all_pred, all_true, all_seed = [], [], []
    dv_loss_sum, vel_loss_sum, count = 0.0, 0.0, 0

    with torch.no_grad():
        for x, dv, seed in loader:
            x, dv, seed = x.to(DEVICE), dv.to(DEVICE), seed.to(DEVICE)
            pred_dv = model(x)
            pred_v = seed[:, None] + pred_dv.cumsum(dim=1)
            true_v = seed[:, None] + dv.cumsum(dim=1)
            dv_loss_sum += nn.functional.smooth_l1_loss(pred_dv, dv, reduction="sum").item()
            vel_loss_sum += nn.functional.smooth_l1_loss(pred_v, true_v, reduction="sum").item()
            count += dv.numel()
            all_pred.append(pred_v.cpu().numpy())
            all_true.append(true_v.cpu().numpy())
            all_seed.append(seed.cpu().numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    seeds = np.concatenate(all_seed)
    err = pred - true
    ae = np.abs(err)
    base = np.broadcast_to(seeds[:, None], true.shape)
    base_ae = np.abs(base - true)
    outdir.mkdir(parents=True, exist_ok=True)

    model_mae, baseline_mae, model_p80, baseline_p80 = {}, {}, {}, {}
    for sec in CHECKPOINTS:
        j = sec * HZ - 1
        model_mae[sec] = float(ae[:, j].mean())
        baseline_mae[sec] = float(base_ae[:, j].mean())
        model_p80[sec] = float(np.quantile(ae[:, j], .8))
        baseline_p80[sec] = float(np.quantile(base_ae[:, j], .8))

    # Exact checkpoint MAE and cumulative MAE curves.
    t = np.arange(1, 601) / HZ
    exact_mae = ae.mean(axis=0)
    cumulative_mae = np.cumsum(ae.mean(axis=0)) / np.arange(1, 601)
    baseline_exact = base_ae.mean(axis=0)
    baseline_cumulative = np.cumsum(base_ae.mean(axis=0)) / np.arange(1, 601)

    plt.figure(figsize=(10, 5))
    plt.plot(t, exact_mae, label="TCN exact-time MAE")
    plt.plot(t, baseline_exact, label="Zero-Δv exact-time MAE")
    plt.plot(t, cumulative_mae, "--", label="TCN cumulative MAE")
    plt.plot(t, baseline_cumulative, "--", label="Baseline cumulative MAE")
    for sec in CHECKPOINTS:
        plt.axvline(sec, alpha=.12, linewidth=.8)
    plt.xlabel("Blackout time (s)")
    plt.ylabel("Absolute velocity error (m/s)")
    plt.title("Velocity MAE over blackout horizon")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "mae_vs_blackout_time.png", dpi=160); plt.close()

    # MAE at exact requested checkpoints.
    plt.figure(figsize=(8, 5))
    plt.plot(CHECKPOINTS, [model_mae[s] for s in CHECKPOINTS], marker="o", label="TCN")
    plt.plot(CHECKPOINTS, [baseline_mae[s] for s in CHECKPOINTS], marker="o", label="Zero-Δv baseline")
    plt.xlabel("Blackout duration (s)")
    plt.ylabel("MAE at checkpoint (m/s)")
    plt.title("Checkpoint MAE: TCN vs zero-Δv")
    plt.xticks(CHECKPOINTS); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "checkpoint_mae.png", dpi=160); plt.close()

    # A few trajectory overlays.
    n = min(max(1, examples_to_plot), len(true))
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8*n), sharex=True, squeeze=False)
    for i in range(n):
        ax = axes[i, 0]
        ax.plot(t, true[i], label="Vehicle GT", linewidth=1.7)
        ax.plot(t, pred[i], label="TCN", linewidth=1.5)
        ax.plot(t, base[i], "--", label="Zero-Δv baseline")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(f"Test episode {i+1}: 60s MAE={ae[i,-1]:.2f} m/s")
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend()
    axes[-1, 0].set_xlabel("Blackout time (s)")
    fig.tight_layout(); fig.savefig(outdir / "test_trajectory_examples.png", dpi=160); plt.close(fig)

    # Per-episode MAE comparison.
    ep_model = ae.mean(axis=1)
    ep_base = base_ae.mean(axis=1)
    plt.figure(figsize=(9, 5))
    plt.hist(ep_model, bins=30, alpha=.65, label="TCN")
    plt.hist(ep_base, bins=30, alpha=.55, label="Zero-Δv baseline")
    plt.xlabel("Per-episode 60s MAE (m/s)")
    plt.ylabel("Episode count")
    plt.title("Per-episode MAE distribution")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "per_episode_mae.png", dpi=160); plt.close()

    # Signed error distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(err.ravel(), bins=60)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Prediction error (m/s)")
    plt.ylabel("Count")
    plt.title("TCN signed velocity error")
    plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(outdir / "signed_error_distribution.png", dpi=160); plt.close()

    # CSV report.
    with (outdir / "checkpoint_metrics.csv").open("w") as f:
        f.write("seconds,model_mae_mps,zero_dv_mae_mps,model_p80_mps,baseline_p80_mps\\n")
        for sec in CHECKPOINTS:
            f.write(f"{sec},{model_mae[sec]:.6f},{baseline_mae[sec]:.6f},"
                    f"{model_p80[sec]:.6f},{baseline_p80[sec]:.6f}\\n")

    print(f"Test episodes: {len(true)}")
    print(f"Δv SmoothL1: {dv_loss_sum / max(count, 1):.6f}")
    print(f"Velocity SmoothL1: {vel_loss_sum / max(count, 1):.6f}")
    print(f"Overall model MAE: {ae.mean():.4f} m/s ({ae.mean()*3.6:.2f} km/h)")
    print(f"Overall zero-Δv MAE: {base_ae.mean():.4f} m/s ({base_ae.mean()*3.6:.2f} km/h)")
    print(f"Overall model RMSE: {np.sqrt(np.mean(err**2)):.4f} m/s")
    print(f"Overall model bias: {err.mean():+.4f} m/s")
    print("\\nCheckpoint MAE (m/s):")
    print("sec | TCN exact | zero-Δv | TCN P80 | baseline P80")
    for sec in CHECKPOINTS:
        print(f"{sec:>3} | {model_mae[sec]:>9.4f} | {baseline_mae[sec]:>7.4f} | "
              f"{model_p80[sec]:>8.4f} | {baseline_p80[sec]:>12.4f}")
    print(f"Saved plots and CSV to: {outdir.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_blackout_windows.pkl")
    ap.add_argument("--scalers", default="data/scalers.pkl")
    ap.add_argument("--checkpoint", default="data/tcn_delta_v_best.pt")
    ap.add_argument("--outdir", default="outputs/tcn_delta_v_eval")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    windows = load_pickle(args.test)
    scalers = load_pickle(args.scalers)
    if not windows:
        raise RuntimeError(f"No test episodes found in {args.test}")
    loader = DataLoader(Episodes(windows, scalers), batch_size=args.batch_size, shuffle=False)

    model = TCN().to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    # Accept either a raw state_dict or a training checkpoint dict.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint} on {DEVICE}")
    evaluate(model, loader, Path(args.outdir), args.examples)


if __name__ == "__main__":
    main()
