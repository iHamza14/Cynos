import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pickle
import torch

from ekf.deterministic_ekf import DeterministicEKF, wrap_angle_rad


def compass_to_math(yaw_c):
    return wrap_angle_rad(np.pi / 2 - yaw_c)


def estimate_roll_pitch(accel_ctx):
    a = np.asarray(accel_ctx).mean(axis=0)
    roll = float(np.arctan2(a[1], a[2]))
    pitch = float(np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2)))
    return roll, pitch


def run_variant(win, gyro_axis=1, gyro_sign=+1, use_accel=True, use_nhc=True):
    first_blackout = win['blackout']['raw_accel'][:5]
    roll, pitch = estimate_roll_pitch(first_blackout)
    ekf = DeterministicEKF(roll=roll, pitch=pitch, dt=0.1)

    vr = float(win['context']['vr_seed'])
    yaw0 = compass_to_math(float(win['context']['heading_seq'][-1]))
    ekf.x[4] = yaw0
    ekf.x[2] = vr * np.cos(yaw0)
    ekf.x[3] = vr * np.sin(yaw0)

    N = win['blackout']['raw_accel'].shape[0]
    traj = np.zeros((N, 2))
    speeds = np.zeros(N)
    yaws = np.zeros(N)

    for t in range(N):
        a = win['blackout']['raw_accel'][t]
        g = win['blackout']['raw_gyro'][t]

        # Monkey-patch the gyro axis/sign into a fake gyro vector
        g_patched = np.zeros(3)
        g_patched[1] = gyro_sign * g[gyro_axis]

        if use_accel:
            ekf.predict(a, g_patched)
        else:
            # Skip accel integration: constant velocity, but still propagate yaw
            dt = ekf.dt
            px, py, vx, vy, yaw, b_gz = ekf.x
            yaw_new = wrap_angle_rad(yaw + (g_patched[1] - b_gz) * dt)
            px_new = px + vx * dt
            py_new = py + vy * dt
            ekf.x = torch.stack([px_new, py_new, vx, vy, yaw_new, b_gz])

        if use_nhc:
            ekf.update_nhc()

        traj[t] = ekf.x[0:2].detach().numpy()
        speeds[t] = np.linalg.norm(ekf.x[2:4].detach().numpy())
        yaws[t] = ekf.x[4].detach().numpy()

    return traj, speeds, yaws


def drift_pct(pred, gt):
    end_err = np.linalg.norm(pred[-1] - gt[-1])
    path_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    return 100.0 * end_err / max(path_len, 1.0)


if __name__ == '__main__':
    with open('data/test_blackout_windows.pkl', 'rb') as f:
        windows = pickle.load(f)

    # Pick one 60s window
    w = next(x for x in windows if x['blackout_dur_sec'] == 60)
    gt = np.asarray(w['ground_truth']['cum_disp_m'])[:, :2]
    gt_speed = np.asarray(w['ground_truth']['speeds_ms'])
    gt_yaw = np.unwrap(np.asarray(w['ground_truth']['headings_rad']))
    gt_yaw_math = np.pi / 2 - gt_yaw     # compass → math

    print(f"Window: {w['blackout_dur_sec']}s, N={len(gt)}")
    print(f"vr_seed = {float(w['context']['vr_seed']):.2f}")
    print(f"gt avg speed = {gt_speed.mean():.2f} m/s "
          f"(= {gt_speed.mean()*3.6:.1f} km/h)")
    print(f"gt path length = {np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)):.1f} m")
    print()

    variants = [
        ('axis1 sign+ accel nhc',  1, +1, True,  True),
        ('axis1 sign- accel nhc',  1, -1, True,  True),
        ('axis2 sign+ accel nhc',  2, +1, True,  True),
        ('axis2 sign- accel nhc',  2, -1, True,  True),
        ('axis0 sign+ accel nhc',  0, +1, True,  True),
        ('axis1 sign+ noaccel nhc',1, +1, False, True),
        ('axis1 sign+ accel nonhc',1, +1, True,  False),
        ('axis1 sign+ noaccel nonhc',1,+1, False, False),
    ]

    print(f"{'variant':<30} {'drift%':>8} {'v_start':>8} {'v_end':>8} "
          f"{'yaw_drift':>10}")
    print('-' * 72)

    for name, ax, sign, accel, nhc in variants:
        try:
            traj, speeds, yaws = run_variant(w, ax, sign, accel, nhc)
            d = drift_pct(traj, gt)
            yaw_drift_deg = np.rad2deg(wrap_angle_rad(yaws[-1] - gt_yaw_math[-1]))
            print(f"{name:<30} {d:>7.1f}% {speeds[0]:>8.2f} {speeds[-1]:>8.2f} "
                  f"{yaw_drift_deg:>9.1f}°")
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")

    # Also print diagnostic curves for the best variant
    print()
    print("Best variant curves (first 10 samples of speed):")
    traj, speeds, yaws = run_variant(w, 1, +1, True, True)
    print(f"  EKF speed: {speeds[:10].round(2)}")
    print(f"  GT  speed: {gt_speed[:10].round(2)}")
    print(f"  EKF yaw math (deg): {np.rad2deg(yaws[:10]).round(1)}")
    print(f"  GT  yaw math (deg): {np.rad2deg(gt_yaw_math[:10]).round(1)}")