"""
Training entry point for the IDR model — optimized for small CUDA GPUs.

Gradient stability measures:
  - LR warmup (10% → 100%) over `warmup_epochs`
  - Lower base LR (3e-4)
  - Trajectory loss normalized by total distance
  - State detached every `detach_every` steps (truncated BPTT)
  - NaN guard skips bad batches
  - Tighter gradient clip (0.5)
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.idr_model import IDRModel
from ekf.differentiable_ekf import DifferentiableEKF
from data.dataset import IDRDataset


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def _env_bool(name, default):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    v = os.getenv(name)
    return default if v is None else int(v)


def _autocast(enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _choose_amp_dtype():
    r = os.getenv("AMP_DTYPE", "auto").strip().lower()
    if r == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if r == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def move_batch_to_device(batch, device):
    nb = device.type == "cuda"
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.is_floating_point():
            batch[k] = v.to(device=device, dtype=torch.float32, non_blocking=nb)
        else:
            batch[k] = v.to(device=device, non_blocking=nb)


def upsample_1hz_to_10hz(tensor, target_len):
    t10 = torch.repeat_interleave(tensor, repeats=10, dim=1)
    if t10.shape[1] >= target_len:
        return t10[:, :target_len]
    pad = t10[:, -1:].expand(-1, target_len - t10.shape[1], *t10.shape[2:])
    return torch.cat([t10, pad], dim=1)


# ──────────────────────────────────────────────────────────────────────
# EKF step
# ──────────────────────────────────────────────────────────────────────

def _ekf_step(ekf, state, P, accel, gyro, z_yaw, H_yaw, R_yaw, z_vel, H_vel, R_vel):
    state, P = ekf.predict(state, P, accel, gyro)
    P = P.detach()
    state, P = ekf.update(state, P, z_yaw, H_yaw, R_yaw)
    P = P.detach()
    state, P = ekf.update(state, P, z_vel, H_vel, R_vel)
    P = P.detach()
    return state, P


# ──────────────────────────────────────────────────────────────────────
# EKF rollout
# ──────────────────────────────────────────────────────────────────────

def _run_ekf_forward(model, ekf, batch, config, amp_enabled, amp_dtype):
    device = batch["raw_accel_10hz"].device
    B = batch["raw_accel_10hz"].shape[0]
    T = batch["raw_accel_10hz"].shape[1]

    with _autocast(amp_enabled, amp_dtype):
        net = model(
            batch["mtn_input"], batch["accel_feat"],
            batch["gyro_feat"], batch["fft_feat"], batch["broadcast_feat"],
        )

    psi_rate = net["psi_rate_corr"].float()
    dv       = net["dv"].float()
    dpsi     = net["dpsi"].float()
    roll_m   = net["roll"].float()
    pitch_m  = net["pitch"].float()
    yaw_res  = net["yaw_residual"].float()

    psi_rate_10 = upsample_1hz_to_10hz(psi_rate, T)
    dv_10       = upsample_1hz_to_10hz(dv, T)
    dpsi_10     = upsample_1hz_to_10hz(dpsi, T)
    roll_10     = upsample_1hz_to_10hz(roll_m, T)
    pitch_10    = upsample_1hz_to_10hz(pitch_m, T)
    yaw_res_10  = upsample_1hz_to_10hz(yaw_res, T)

    init_state = batch["initial_state"]
    state = init_state.clone()
    vr_seed  = state[:, 17].clone()
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

    raw_a = batch["raw_accel_10hz"]
    raw_g = batch["raw_gyro_10hz"]

    preds_speed = torch.empty(B, T, device=device)
    preds_traj  = torch.empty(B, T, 2, device=device)

    # Truncated BPTT — detach the state periodically to cap the
    # backward graph depth. Without this, gradients compound over
    # 150+ unrolled EKF steps and blow up.
    detach_every = config["train"].get("detach_every", 50)

    for t in range(T):
        if detach_every > 0 and t > 0 and t % detach_every == 0:
            state = state.detach()

        state = state.clone()
        state[:, 6]  = roll_10[:, t]
        state[:, 7]  = pitch_10[:, t]
        state[:, 15] = psi_rate_10[:, t]

        z_yaw = (init_yaw + yaw_res_10[:, t] + dpsi_10[:, t]).unsqueeze(1)

        v_mag = vr_seed + dv_10[:, t]
        cur_yaw = state[:, 8]
        z_vel = torch.stack(
            (v_mag * torch.cos(cur_yaw), v_mag * torch.sin(cur_yaw)), dim=1
        )

        state, P = _ekf_step(
            ekf, state, P,
            raw_a[:, t], raw_g[:, t],
            z_yaw, H_yaw, R_yaw,
            z_vel, H_vel, R_vel,
        )

        v = state[:, 3:6]
        preds_speed[:, t] = (v.square().sum(dim=1) + 1e-8).sqrt()
        preds_traj[:, t] = state[:, 0:2]

    return preds_speed, preds_traj


# ──────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────

def make_loader(dataset, batch_size, shuffle, num_workers, device):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Loss helper
# ──────────────────────────────────────────────────────────────────────

def compute_loss(criterion, preds_speed, preds_traj, batch, train_cfg):
    """
    Speed loss in m/s + trajectory loss normalized by path length.
    Both terms are dimensionless / near-unit-scale by construction.
    """
    loss_speed = criterion(preds_speed, batch["gt_speeds"])

    dist = batch["total_dist"].reshape(-1) + 1.0    # (B,)
    loss_traj = criterion(
        preds_traj / dist[:, None, None],
        batch["gt_cum_disp"] / dist[:, None, None],
    )

    return (train_cfg.get("loss_speed_scale", 1.0) * loss_speed
            + train_cfg.get("loss_traj_scale", 0.05) * loss_traj)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train():
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    BATCH     = _env_int("BATCH_SIZE", train_cfg["batch_size"])
    VAL_BATCH = _env_int("VAL_BATCH_SIZE", 32)
    VAL_EVERY = _env_int("VAL_EVERY", 3)

    default_workers = min(4, os.cpu_count() or 1)
    num_workers = _env_int("NUM_WORKERS", default_workers)

    train_ds = IDRDataset(
        "data/train_blackout_windows.pkl",
        scalers_path="data/scalers.pkl",
        augment=True, durations=[15],
    )
    val_ds = IDRDataset(
        "data/test_blackout_windows.pkl",
        scalers_path="data/scalers.pkl",
        augment=False, durations=[15],
    )

    train_loader = make_loader(train_ds, BATCH, True, num_workers, device)
    val_loader   = make_loader(val_ds, VAL_BATCH, False, num_workers, device)

    model = IDRModel(config).to(device)
    ekf = DifferentiableEKF(config).to(device)
    ekf.eval()
    for p in ekf.parameters():
        p.requires_grad = False

    base_lr = train_cfg["lr"]
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
    )

    # Linear warmup from 10% → 100%, then cosine decay
    sched_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg["epochs"] - warmup_epochs),
        eta_min=base_lr * 0.01,
    )

    criterion = nn.SmoothL1Loss(beta=train_cfg["smooth_l1_beta"])

    amp_enabled = device.type == "cuda" and _env_bool("USE_AMP", True)
    amp_dtype = _choose_amp_dtype() if amp_enabled else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"Training on {device} | batch={BATCH} | val_batch={VAL_BATCH} | "
        f"AMP={amp_enabled} ({amp_dtype}) | workers={num_workers} | "
        f"val_every={VAL_EVERY} | base_lr={base_lr:.1e} | warmup={warmup_epochs}"
    )

    best_drift = float("inf")
    patience = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        skipped = 0

        for batch in train_loader:
            move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            preds_speed, preds_traj = _run_ekf_forward(
                model, ekf, batch, config, amp_enabled, amp_dtype
            )
            loss = compute_loss(criterion, preds_speed, preds_traj,
                                batch, train_cfg)

            # Guard: skip batches that produced non-finite loss
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                optimizer.step()

            train_loss += loss.detach().item()

        train_loss /= max(1, len(train_loader) - skipped)

        # Scheduler step
        if epoch < warmup_epochs:
            sched_warmup.step()
        else:
            sched_cosine.step()

        lr_now = optimizer.param_groups[0]["lr"]
        suffix = f" | LR: {lr_now:.2e}"
        if skipped:
            suffix += f" | skipped: {skipped}"

        # ── Validation ────────────────────────────────────────────
        if (epoch + 1) % VAL_EVERY != 0 and epoch != train_cfg["epochs"] - 1:
            print(f"Epoch {epoch + 1:03d}/{train_cfg['epochs']} | "
                  f"Train: {train_loss:.4f}{suffix}")
            continue

        model.eval()
        val_loss, total_drift, n_windows = 0.0, 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                move_batch_to_device(batch, device)
                preds_speed, preds_traj = _run_ekf_forward(
                    model, ekf, batch, config, amp_enabled, amp_dtype
                )
                val_loss += compute_loss(
                    criterion, preds_speed, preds_traj, batch, train_cfg
                ).item()

                err = torch.linalg.vector_norm(
                    preds_traj[:, -1, :] - batch["gt_cum_disp"][:, -1, :], dim=1
                )
                drift = err / (batch["total_dist"].reshape(-1) + 1e-8) * 100.0
                total_drift += drift.sum().item()
                n_windows += drift.numel()

        val_loss /= max(1, len(val_loader))
        avg_drift = total_drift / max(1, n_windows)
        print(f"Epoch {epoch + 1:03d}/{train_cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Drift: {avg_drift:.2f}%{suffix}")

        if avg_drift < best_drift:
            best_drift = avg_drift
            patience = 0
            torch.save(model.state_dict(), "data/best_model.pth")
            print("  -> saved")
        else:
            patience += 1
            if patience >= train_cfg["early_stop_patience"]:
                print(f"Early stop at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    train()