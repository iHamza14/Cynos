"""
Run inference on test windows and produce a Folium map.

Loads a trained checkpoint, runs the same EKF forward as training, converts
predicted ENU trajectories back to lat/lon, and overlays them on OpenStreetMap.

Usage:
    python -m scripts.inference_map
    python -m scripts.inference_map --checkpoint data/best_model.pth --n 5

Outputs:
    data/inference_map.html         — Folium map with selected windows
    data/inference_map_summary.txt  — Per-window drift numbers
"""

import os
import sys
import argparse
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import folium

from models.idr_model import IDRModel
from ekf.differentiable_ekf import DifferentiableEKF
from data.dataset import IDRDataset
from torch.utils.data import DataLoader


# ── WGS84 constants ─────────────────────────────────────────────────────
R_EARTH_DEG = 111320.0   # meters per degree of latitude (approx)


def enu_to_latlon(east, north, start_lat, start_lon):
    """Convert local ENU meters (relative to start) into lat/lon degrees."""
    lat = start_lat + north / R_EARTH_DEG
    lon = start_lon + east / (R_EARTH_DEG * np.cos(np.radians(start_lat)))
    return lat, lon


def _run_ekf_forward(model, ekf, batch, config, device):
    """Same EKF rollout as training, but no autocast / no AMP."""
    accel = batch["raw_accel_10hz"].to(device)
    gyro = batch["raw_gyro_10hz"].to(device)

    with torch.inference_mode():
        net = model(
            batch["mtn_input"].to(device),
            batch["accel_feat"].to(device),
            batch["gyro_feat"].to(device),
            batch["fft_feat"].to(device),
            batch["broadcast_feat"].to(device),
        )

    B = accel.shape[0]
    T = accel.shape[1]

    def up(t):
        return torch.repeat_interleave(t, 10, dim=1)[:, :T]

    roll_10 = up(net["roll"])
    pitch_10 = up(net["pitch"])
    yaw_res_10 = up(net["yaw_residual"])
    dv_10 = up(net["dv"])
    dpsi_10 = up(net["dpsi"])
    psi_rate_10 = up(net["psi_rate_corr"])

    state = batch["initial_state"].to(device).clone()
    vr_seed = state[:, 17].clone()
    init_yaw = state[:, 8].clone()

    P = torch.eye(18, device=device).unsqueeze(0).expand(B, -1, -1).clone() * 0.01

    H_yaw = torch.zeros(B, 1, 18, device=device)
    H_yaw[:, 0, 8] = 1.0
    R_yaw = torch.eye(1, device=device).unsqueeze(0).expand(B, -1, -1) \
            * config["ekf"]["r_yaw_straight"]

    H_vel = torch.zeros(B, 2, 18, device=device)
    H_vel[:, 0, 3] = 1.0
    H_vel[:, 1, 4] = 1.0
    R_vel = torch.eye(2, device=device).unsqueeze(0).expand(B, -1, -1) * 0.1

    traj = torch.empty(B, T, 2, device=device)
    speed = torch.empty(B, T, device=device)

    for t in range(T):
        state = state.clone()
        state[:, 6] = roll_10[:, t]
        state[:, 7] = pitch_10[:, t]
        state[:, 15] = psi_rate_10[:, t]

        z_yaw = (init_yaw + yaw_res_10[:, t] + dpsi_10[:, t]).unsqueeze(1)
        v_mag = vr_seed + dv_10[:, t]
        cy = state[:, 8]
        z_vel = torch.stack((v_mag * torch.cos(cy), v_mag * torch.sin(cy)), dim=1)

        state, P = ekf.predict(state, P, accel[:, t], gyro[:, t])
        P = P.detach()
        state, P = ekf.update(state, P, z_yaw, H_yaw, R_yaw)
        P = P.detach()
        state, P = ekf.update(state, P, z_vel, H_vel, R_vel)
        P = P.detach()

        traj[:, t] = state[:, 0:2]
        speed[:, t] = (state[:, 3:6].square().sum(dim=1) + 1e-8).sqrt()

    return traj.cpu().numpy(), speed.cpu().numpy()


def pick_windows(test_windows, n):
    """Return n windows spread across the drift distribution."""
    if n <= 0:
        return []
    if n >= len(test_windows):
        return list(range(len(test_windows)))
    # Even spacing
    idxs = np.linspace(0, len(test_windows) - 1, n).astype(int)
    return list(idxs)


def build_map(windows_data, out_path):
    """windows_data: list of dicts with 'gt_ll', 'pred_ll', 'start', 'end_gt', 'end_pred', 'drift', 'tag'"""
    # Center on first window's start
    center = windows_data[0]['start']
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    colors = ['red', 'purple', 'darkred', 'cadetblue', 'darkgreen',
              'darkblue', 'orange', 'darkpurple']

    for i, w in enumerate(windows_data):
        col = colors[i % len(colors)]

        # GT in blue
        folium.PolyLine(
            locations=w['gt_ll'],
            color='blue', weight=4, opacity=0.8,
            tooltip=f"GT {w['tag']} (drift {w['drift']:.1f}%)"
        ).add_to(m)

        # Predicted in colored
        folium.PolyLine(
            locations=w['pred_ll'],
            color=col, weight=4, opacity=0.9, dash_array='8',
            tooltip=f"Pred {w['tag']} (drift {w['drift']:.1f}%)"
        ).add_to(m)

        # Markers
        folium.CircleMarker(
            location=w['start'], radius=6, color='black', fill=True,
            fill_opacity=1.0, tooltip=f"Start {w['tag']}"
        ).add_to(m)
        folium.Marker(
            location=w['end_gt'], icon=folium.Icon(color='blue', icon='flag'),
            tooltip=f"GT end {w['tag']}"
        ).add_to(m)
        folium.Marker(
            location=w['end_pred'], icon=folium.Icon(color='red', icon='flag'),
            tooltip=f"Pred end {w['tag']}"
        ).add_to(m)

    m.save(out_path)
    print(f"Saved map: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='data/best_model.pth')
    ap.add_argument('--config', default='config/default.yaml')
    ap.add_argument('--n', type=int, default=5,
                    help='number of test windows to visualize')
    ap.add_argument('--duration', type=int, default=15)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────
    model = IDRModel(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    ekf = DifferentiableEKF(config).to(device)
    ekf.eval()
    for p in ekf.parameters():
        p.requires_grad = False

    # ── Load test set (single duration) ─────────────────────────────
    test_ds = IDRDataset(
        'data/test_blackout_windows.pkl',
        scalers_path='data/scalers.pkl',
        augment=False,
        durations=[args.duration],
    )
    print(f"Test windows (duration={args.duration}s): {len(test_ds)}")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Run inference and collect drift for ranking ─────────────────
    per_window = []

    for idx, batch in enumerate(loader):
        traj, speed = _run_ekf_forward(model, ekf, batch, config, device)
        pred_traj = traj[0]                           # (T, 2) — East, North
        gt_traj = batch['gt_cum_disp'][0].numpy()     # (T, 2)
        total_dist = float(batch['total_dist'][0, 0])

        end_err = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
        drift = 100.0 * end_err / max(total_dist, 1.0)

        per_window.append({
            'idx': idx,
            'pred_traj': pred_traj,
            'gt_traj': gt_traj,
            'start_lat': float(batch['start_lat'][0]) if 'start_lat' in batch
                         else float(test_ds.windows[idx]['ground_truth']['start_lat']),
            'start_lon': float(batch['start_lon'][0]) if 'start_lon' in batch
                         else float(test_ds.windows[idx]['ground_truth']['start_lon']),
            'drift': drift,
            'total_dist': total_dist,
        })

    drifts = np.array([w['drift'] for w in per_window])
    print(f"\nDrift statistics (N={len(drifts)}):")
    print(f"  mean   = {drifts.mean():.2f}%")
    print(f"  median = {np.median(drifts):.2f}%")
    print(f"  p20    = {np.percentile(drifts, 20):.2f}%")
    print(f"  p80    = {np.percentile(drifts, 80):.2f}%")
    print(f"  min    = {drifts.min():.2f}%")
    print(f"  max    = {drifts.max():.2f}%")

    # ── Pick windows spread across distribution ────────────────────
    order = np.argsort(drifts)
    n = args.n
    if n >= len(order):
        picks = list(order)
    else:
        # Spread across p20..p80
        idxs = np.linspace(int(0.2 * len(order)), int(0.8 * len(order)), n).astype(int)
        picks = [order[i] for i in idxs]

    # ── Convert ENU to lat/lon for each pick ───────────────────────
    windows_data = []
    for rank, i in enumerate(picks):
        w = per_window[i]
        start_lat = w['start_lat']
        start_lon = w['start_lon']

        gt_ll = []
        for e, n_ in w['gt_traj']:
            lat, lon = enu_to_latlon(e, n_, start_lat, start_lon)
            gt_ll.append((lat, lon))

        pred_ll = []
        for e, n_ in w['pred_traj']:
            lat, lon = enu_to_latlon(e, n_, start_lat, start_lon)
            pred_ll.append((lat, lon))

        windows_data.append({
            'gt_ll': gt_ll,
            'pred_ll': pred_ll,
            'start': (start_lat, start_lon),
            'end_gt': gt_ll[-1],
            'end_pred': pred_ll[-1],
            'drift': w['drift'],
            'total_dist': w['total_dist'],
            'tag': f"#{i} (d={w['drift']:.1f}%)",
        })

    # ── Build map ──────────────────────────────────────────────────
    os.makedirs('data', exist_ok=True)
    out_html = f'data/inference_map_{args.duration}s.html'
    build_map(windows_data, out_html)

    # ── Save summary ───────────────────────────────────────────────
    summary_path = f'data/inference_map_{args.duration}s_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Duration: {args.duration}s, N={len(drifts)}\n")
        f.write(f"Mean drift   : {drifts.mean():.2f}%\n")
        f.write(f"Median drift : {np.median(drifts):.2f}%\n")
        f.write(f"p20 / p80    : {np.percentile(drifts, 20):.2f}% / "
                f"{np.percentile(drifts, 80):.2f}%\n")
        f.write(f"Min / max    : {drifts.min():.2f}% / {drifts.max():.2f}%\n")
        f.write("\nWindows plotted:\n")
        for w in windows_data:
            f.write(f"  {w['tag']:>20s}  total_dist={w['total_dist']:.1f} m\n")
    print(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()