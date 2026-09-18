"""
Plot deterministic EKF trajectory vs ground truth.
15s windows only — p20, median, p80, worst.
Run from project root: python -m scripts.plot_traj
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch

from ekf.deterministic_ekf import DeterministicEKF, wrap_angle_rad
from models.rotation_utils import euler_to_rotation_matrix


def compass_to_math(yaw_c):
    return wrap_angle_rad(np.pi / 2 - yaw_c)


def estimate_roll_pitch(accel):
    a = np.asarray(accel)
    a_mean = a.mean(axis=0) if a.ndim == 2 else a
    roll = float(np.arctan2(a_mean[1], a_mean[2]))
    pitch = float(np.arctan2(-a_mean[0], np.sqrt(a_mean[1]**2 + a_mean[2]**2)))
    return roll, pitch


def init_accel_bias(ctx_accel, ctx_yaw_rate_rad_s, roll, pitch, yaw_math):
    steady = np.abs(ctx_yaw_rate_rad_s) < np.deg2rad(2.0)
    if steady.sum() < 10:
        return 0.0, 0.0

    bias_body = np.asarray(ctx_accel)[steady].mean(axis=0)

    R = euler_to_rotation_matrix(
        torch.tensor(float(roll)),
        torch.tensor(float(pitch)),
        torch.tensor(float(yaw_math)),
    ).numpy()

    bias_world = R @ bias_body - np.array([0.0, 0.0, 9.81])
    bias_residual_body = R.T @ bias_world

    return float(bias_residual_body[0]), float(bias_residual_body[1])


def run_window(win):
    roll, pitch = estimate_roll_pitch(win['blackout']['raw_accel'][:5])

    vr = float(win['context']['vr_seed'])
    yaw0 = compass_to_math(float(win['context']['heading_seq'][-1]))

    ctx_headings = np.asarray(win['context']['heading_seq'])
    ctx_yaw_rate = np.gradient(np.unwrap(ctx_headings)) / 0.1
    b_ax, b_ay = init_accel_bias(
        win['context']['raw_accel'], ctx_yaw_rate, roll, pitch, yaw0
    )

    ekf = DeterministicEKF(roll=roll, pitch=pitch, dt=0.1)

    ekf.x[4] = yaw0
    ekf.x[2] = vr * np.cos(yaw0)
    ekf.x[3] = vr * np.sin(yaw0)
    ekf.x[6] = b_ax
    ekf.x[7] = b_ay

    N = win['blackout']['raw_accel'].shape[0]
    traj = np.zeros((N, 2))
    yaw_hist = np.zeros(N)
    speed_hist = np.zeros(N)

    for t in range(N):
        ekf.predict(win['blackout']['raw_accel'][t],
                    win['blackout']['raw_gyro'][t])
        ekf.update_nhc()

        traj[t] = ekf.x[0:2].detach().numpy()
        yaw_hist[t] = ekf.x[4].detach().numpy()
        speed_hist[t] = np.linalg.norm(ekf.x[2:4].detach().numpy())

    return traj, yaw_hist, speed_hist


def drift_pct(pred, gt):
    end_err = np.linalg.norm(pred[-1] - gt[-1])
    path_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    return 100.0 * end_err / max(path_len, 1.0)


def select_15s_windows(windows):
    """Returns (picks, labels) — p20, median, p80, worst for 15s windows."""
    subset = [w for w in windows if w['blackout_dur_sec'] == 15]

    drifts = np.array([
        drift_pct(run_window(w)[0],
                  np.asarray(w['ground_truth']['cum_disp_m'])[:, :2])
        for w in subset
    ])
    order = np.argsort(drifts)
    n = len(order)

    idxs = {
        'p20':    order[int(0.20 * n)],
        'median': order[int(0.50 * n)],
        'p80':    order[int(0.80 * n)],
        'worst':  order[-1],
    }

    picks, labels = [], []
    for name in ['p20', 'median', 'p80', 'worst']:
        i = idxs[name]
        picks.append(subset[i])
        labels.append(f"{name} 15s | drift={drifts[i]:.1f}%")

    print(f"15s windows: N={n}  "
          f"p20={drifts[idxs['p20']]:.1f}%  "
          f"median={drifts[idxs['median']]:.1f}%  "
          f"p80={drifts[idxs['p80']]:.1f}%  "
          f"worst={drifts[idxs['worst']]:.1f}%")

    return picks, labels


if __name__ == '__main__':
    with open('data/test_blackout_windows.pkl', 'rb') as f:
        windows = pickle.load(f)

    picks, labels = select_15s_windows(windows)

    # ── Figure 1: trajectories ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, w, label in zip(axes, picks, labels):
        pred, _, _ = run_window(w)
        gt = np.asarray(w['ground_truth']['cum_disp_m'])[:, :2]

        pred_len = np.sum(np.linalg.norm(np.diff(pred, axis=0), axis=1))
        gt_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))

        ax.plot(gt[:, 0], gt[:, 1], 'k-', lw=2.0, label='Ground truth')
        ax.plot(pred[:, 0], pred[:, 1], 'r-', lw=1.8, label='Deterministic EKF')
        ax.scatter(gt[0, 0], gt[0, 1], c='k', s=80, marker='o', zorder=5,
                   label='start')
        ax.scatter(gt[-1, 0], gt[-1, 1], c='k', s=80, marker='x', zorder=5,
                   label='end (GT)')
        ax.scatter(pred[-1, 0], pred[-1, 1], c='r', s=80, marker='x', zorder=5,
                   label='end (pred)')

        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title(f"{label} | len ratio={pred_len/gt_len:.2f}")
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig('diag_traj_15s.png', dpi=120)
    print("Saved diag_traj_15s.png")

    # ── Figure 2: speed and heading for the median 15s window ────────────
    w = picks[1]   # median
    _, yaw_h, speed_h = run_window(w)

    gt_speed = np.asarray(w['ground_truth']['speeds_ms'])
    gt_headings = np.asarray(w['ground_truth']['headings_rad'])

    gt_yaw_compass = np.unwrap(gt_headings)
    ekf_yaw_compass = np.unwrap(np.pi / 2 - yaw_h)

    offset = round((gt_yaw_compass[0] - ekf_yaw_compass[0])
                   / (2 * np.pi)) * 2 * np.pi
    gt_yaw_compass = gt_yaw_compass - offset

    t = np.arange(len(gt_speed)) * 0.1

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, gt_speed, 'k-', lw=2, label='GT speed')
    axes[0].plot(t, speed_h, 'r-', lw=1.5, label='EKF speed')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Median 15s window | {labels[1]}")

    axes[1].plot(t, np.rad2deg(gt_yaw_compass), 'k-', lw=2,
                 label='GT heading')
    axes[1].plot(t, np.rad2deg(ekf_yaw_compass), 'r-', lw=1.5,
                 label='EKF heading')
    axes[1].set_ylabel('Heading (deg, unwrapped)')
    axes[1].set_xlabel('time (s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diag_speed_yaw_15s.png', dpi=120)
    print("Saved diag_speed_yaw_15s.png")