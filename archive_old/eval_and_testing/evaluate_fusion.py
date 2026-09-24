#!/usr/bin/env python3
"""
Stage C: Velocity + Heading Fusion and Dead-Reckoning Evaluation.
Evaluates the combined Cynos DR stack across 60-second blackout episodes
at the configured benchmark checkpoints (2, 5, 10, 15, 30, 60 seconds).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from model.velocity.velocity_estimator import DVSEModel, GyroTCN

HZ = 10.0
CHECKPOINTS_SEC = [2, 5, 10, 15, 30, 60]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Stage C: DR Navigation Fusion")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--test-file", default="test_blackout_windows.pkl")
    p.add_argument("--dvse-weights", type=Path, default=Path("dvse_output/best_dvse.pt"))
    p.add_argument("--gyro-weights", type=Path, default=Path("gyro_tcn_output/best_gyro_tcn_processed.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("fusion_output"))
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    test_path = args.data_dir / args.test_file
    scalers_path = args.data_dir / "scalers.pkl"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_items = load_pickle(test_path)
    scalers = load_pickle(scalers_path) if scalers_path.exists() else None

    # Load Stage A DVSE Velocity Model
    dvse_model = DVSEModel().to(device)
    if args.dvse_weights.exists():
        dvse_model.load_state_dict(torch.load(args.dvse_weights, map_location=device, weights_only=False))
        print(f"Loaded DVSE velocity weights from {args.dvse_weights}")
    else:
        print(f"Warning: {args.dvse_weights} not found. Running with uninitialized velocity weights.")
    dvse_model.eval()

    # Load Stage B Gyro TCN Heading Model
    gyro_model = GyroTCN().to(device)
    gyro_ckpt = None
    if args.gyro_weights.exists():
        gyro_ckpt = torch.load(args.gyro_weights, map_location=device, weights_only=False)
        gyro_model.load_state_dict(gyro_ckpt["model_state"])
        print(f"Loaded Gyro TCN weights from {args.gyro_weights}")
    else:
        print(f"Warning: {args.gyro_weights} not found. Running with uninitialized gyro weights.")
    gyro_model.eval()

    acc_scaler = scalers["raw_accel"] if scalers else None
    gyro_scaler = scalers["raw_gyro"] if scalers else None

    # Gyro target scaling params if saved in checkpoint
    y_mean = gyro_ckpt.get("target_mean", 0.0) if gyro_ckpt else 0.0
    y_std = gyro_ckpt.get("target_std", 1.0) if gyro_ckpt else 1.0

    checkpoint_errors = {cp: [] for cp in CHECKPOINTS_SEC}
    episode_records = []

    with torch.no_grad():
        for ep_idx, item in enumerate(test_items):
            raw_accel = np.asarray(item["blackout"]["raw_accel"], dtype=np.float32)
            raw_gyro = np.asarray(item["blackout"]["raw_gyro"], dtype=np.float32)
            gt_speeds = np.asarray(item["ground_truth"]["speeds_ms"], dtype=np.float32)
            gt_headings_deg = np.asarray(item["ground_truth"]["headings_deg"], dtype=np.float32)
            gt_disp = np.asarray(item["ground_truth"]["cumulative_displacement_ne_m"], dtype=np.float32)
            vr_seed = float(item["context"]["vr_seed_ms"])
            init_heading_deg = float(item["ground_truth"]["headings_deg"][0])

            # 1. Scale inputs
            acc_scaled = acc_scaler.transform(raw_accel) if acc_scaler else raw_accel
            gyro_scaled = gyro_scaler.transform(raw_gyro) if gyro_scaler else raw_gyro

            # 2. Heading Inference (10 Hz)
            gyro_tensor = torch.from_numpy(gyro_scaled).unsqueeze(0).to(device)
            pred_yaw_rate_norm = gyro_model(gyro_tensor).squeeze(0).cpu().numpy()
            pred_yaw_rate_dps = pred_yaw_rate_norm * y_std + y_mean
            
            # Integrate heading change at 10 Hz
            dt_10hz = 1.0 / HZ
            heading_change_deg = np.cumsum(pred_yaw_rate_dps * dt_10hz)
            pred_heading_deg = (init_heading_deg + heading_change_deg) % 360.0
            pred_heading_rad = np.radians(pred_heading_deg)

            # 3. Velocity Inference (1 Hz sequence)
            T = len(raw_accel) // int(HZ)
            vr_seq = np.zeros((1, T, 1), dtype=np.float32)
            vr_seq[0, 0, 0] = vr_seed
            # Auto-regressive reference speed for subsequent seconds during DR
            # (In true inference, feed recursively estimated speed)
            acc_t = torch.from_numpy(acc_scaled).unsqueeze(0).to(device)
            gyro_t = torch.from_numpy(gyro_scaled).unsqueeze(0).to(device)
            vr_t = torch.from_numpy(vr_seq).to(device)
            v0_t = torch.tensor([[vr_seed]], dtype=torch.float32).to(device)

            delta_v, pred_v_seq, _, _ = dvse_model(acc_t, gyro_t, vr_t, v_0=v0_t, hz=int(HZ))
            pred_v = pred_v_seq.squeeze(0).cpu().numpy() # [T] (1 Hz)

            # 4. Dead-Reckoning Navigation Fusion (1 Hz)
            # Sample heading at the midpoint or end of each 1-second interval
            psi_1hz = np.array([pred_heading_rad[min((i + 1) * int(HZ) - 1, len(pred_heading_rad) - 1)] for i in range(T)])
            
            dt_1s = 1.0
            delta_s = pred_v * dt_1s
            delta_E = delta_s * np.sin(psi_1hz)
            delta_N = delta_s * np.cos(psi_1hz)

            dr_E = np.cumsum(delta_E)
            dr_N = np.cumsum(delta_N)

            # Ground truth positions at 1 Hz
            gt_1hz_indices = [min((i + 1) * int(HZ) - 1, len(gt_disp) - 1) for i in range(T)]
            gt_N = gt_disp[gt_1hz_indices, 0]
            gt_E = gt_disp[gt_1hz_indices, 1]

            pos_errors = np.sqrt((dr_E - gt_E) ** 2 + (dr_N - gt_N) ** 2)

            for cp in CHECKPOINTS_SEC:
                if cp <= T:
                    checkpoint_errors[cp].append(pos_errors[cp - 1])

            episode_records.append({
                "episode": ep_idx,
                "dr_E": dr_E,
                "dr_N": dr_N,
                "gt_E": gt_E,
                "gt_N": gt_N,
                "final_error_m": pos_errors[-1] if len(pos_errors) else np.nan,
            })

    # Summary table
    metric_rows = []
    for cp in CHECKPOINTS_SEC:
        errs = checkpoint_errors[cp]
        metric_rows.append({
            "Blackout Horizon (s)": cp,
            "N Episodes": len(errs),
            "Mean Error (m)": float(np.mean(errs)) if errs else 0.0,
            "Median Error (m)": float(np.median(errs)) if errs else 0.0,
            "P90 Error (m)": float(np.percentile(errs, 90)) if errs else 0.0,
            "Max Error (m)": float(np.max(errs)) if errs else 0.0,
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "fusion_checkpoint_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("STAGE C: FUSED DEAD-RECKONING EVALUATION RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nMetrics saved to {args.output_dir / 'fusion_checkpoint_metrics.csv'}")

    # Plot sample trajectory
    if episode_records:
        sample = episode_records[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(sample["gt_E"], sample["gt_N"], "k-", label="Ground Truth Path", linewidth=2)
        ax.plot(sample["dr_E"], sample["dr_N"], "r--", label="Cynos Dead Reckoning", linewidth=2)
        ax.scatter([0], [0], color="green", s=100, zorder=5, label="Blackout Start")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("60s Blackout Dead-Reckoning Trajectory (Stage C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        plot_path = args.output_dir / "fused_dr_sample_trajectory.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"Sample trajectory plot saved to {plot_path}")


if __name__ == "__main__":
    main()
