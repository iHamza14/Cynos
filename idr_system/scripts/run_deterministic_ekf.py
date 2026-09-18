import numpy as np
import pickle
import torch

from ekf.deterministic_ekf import DeterministicEKF, wrap_angle_rad


def estimate_roll_pitch_from_gravity(accel_ctx):
    """Roll/pitch from gravity direction, averaged over context (numpy in, floats out)."""
    a = np.asarray(accel_ctx).mean(axis=0)     # (3,)
    roll = np.arctan2(a[1], a[2])
    pitch = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
    return float(roll), float(pitch)

def compass_to_math(yaw_compass_rad):
    return wrap_angle_rad(np.pi / 2 - yaw_compass_rad)

def run_window(win):
    ctx_accel = win['context']['raw_accel']
    roll, pitch = estimate_roll_pitch_from_gravity(ctx_accel)

    ekf = DeterministicEKF(roll=roll, pitch=pitch, dt=0.1)

    vr = float(win['context']['vr_seed'])

    # Seed yaw from GT heading (last context bin), converted compass → math
    yaw0_compass = float(win['context']['heading_seq'][-1])
    yaw0_math = compass_to_math(yaw0_compass)

    ekf.x[4] = yaw0_math
    ekf.x[2] = vr * np.cos(yaw0_math)
    ekf.x[3] = vr * np.sin(yaw0_math)

    N = win['blackout']['raw_accel'].shape[0]
    traj = np.zeros((N, 2))
    yaw_hist = np.zeros(N)

    for t in range(N):
        a = win['blackout']['raw_accel'][t]
        g = win['blackout']['raw_gyro'][t]

        ekf.predict(a, g)
        ekf.update_nhc()
        # NO yaw soft measurement — ORIENTATION Yaw is unusable

        traj[t] = ekf.x[0:2].detach().numpy()
        yaw_hist[t] = ekf.x[4].detach().numpy()

    return traj, yaw_hist

def drift_percent(pred_traj, gt_traj):
    end_err = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
    path_len = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    return 100.0 * end_err / max(path_len, 1.0)



if __name__ == '__main__':
    with open('data/test_blackout_windows.pkl', 'rb') as f:
        windows = pickle.load(f)

    # Check vr_seed units
    sample_speeds = [float(w['context']['vr_seed']) for w in windows[:20]]
    sample_gt = [float(np.mean(w['ground_truth']['speeds_ms'])) for w in windows[:20]]
    print(f"vr_seed samples: {sample_speeds[:5]}")
    print(f"GT speeds (m/s): {sample_gt[:5]}")
    print(f"ratio (vr_seed / GT m/s): "
        f"{np.mean(sample_speeds) / np.mean(sample_gt):.2f}")
    results = {}
    for w in windows:
        pred, yaw = run_window(w)
        gt = np.asarray(w['ground_truth']['cum_disp_m'])[:, :2]
        d = drift_percent(pred, gt)
        dur = w['blackout_dur_sec']
        results.setdefault(dur, []).append(d)

    print()
    print(f"{'Duration':>10} {'Median drift':>14} {'p80 drift':>12} {'N':>5}")
    print("-" * 45)
    for dur in sorted(results):
        vals = np.array(results[dur])
        print(f"{dur:>10} {np.median(vals):>13.1f}% "
              f"{np.percentile(vals, 80):>11.1f}% {len(vals):>5}")