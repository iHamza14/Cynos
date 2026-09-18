"""Train/evaluate a causal TCN for blackout Δv; compare against zero-Δv baseline."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA = Path("data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE, EPOCHS, LR = 32, 30, 1e-3
CHECKPOINTS = [2, 5, 10, 15, 30, 60]
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_pickle(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


class Episodes(Dataset):
    def __init__(self, windows, scalers):
        self.windows, self.scalers = windows, scalers

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        x = np.concatenate([
            self.scalers[k].transform(w["blackout"][k])
            for k in ("raw_accel", "raw_gyro", "gravity")
        ], axis=1).astype(np.float32)
        dv = np.asarray(w["ground_truth"]["delta_v_ms"], dtype=np.float32)
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


def metrics(model, loader, make_plots=False):
    model.eval()
    pred_mae = {s: [] for s in CHECKPOINTS}
    base_mae = {s: [] for s in CHECKPOINTS}
    dv_losses, vel_losses, examples = [], [], []
    with torch.no_grad():
        for x, dv, seed in loader:
            x, dv, seed = x.to(DEVICE), dv.to(DEVICE), seed.to(DEVICE)
            pred_dv = model(x)
            pred_v = seed[:, None] + pred_dv.cumsum(1)
            true_v = seed[:, None] + dv.cumsum(1)
            base_v = seed[:, None].expand_as(true_v)
            dv_losses.append(nn.functional.smooth_l1_loss(pred_dv, dv).item())
            vel_losses.append(nn.functional.smooth_l1_loss(pred_v, true_v).item())
            for sec in CHECKPOINTS:
                j = sec * 10 - 1
                pred_mae[sec].extend((pred_v[:, j] - true_v[:, j]).abs().cpu().tolist())
                base_mae[sec].extend((base_v[:, j] - true_v[:, j]).abs().cpu().tolist())
            if make_plots and len(examples) < 3:
                for k in range(min(3 - len(examples), len(x))):
                    examples.append((
                        true_v[k].cpu().numpy(),
                        pred_v[k].cpu().numpy(),
                        base_v[k].cpu().numpy(),
                    ))
    result = {
        "dv_smoothl1": float(np.mean(dv_losses)) if dv_losses else float("nan"),
        "velocity_smoothl1": float(np.mean(vel_losses)) if vel_losses else float("nan"),
        "model_mae": {s: float(np.mean(v)) for s, v in pred_mae.items()},
        "baseline_mae": {s: float(np.mean(v)) for s, v in base_mae.items()},
    }
    if make_plots:
        DATA.mkdir(exist_ok=True)
        t = np.arange(1, 601) / 10
        for i, (truth, pred, base) in enumerate(examples):
            plt.figure(figsize=(10, 4))
            plt.plot(t, truth, label="Vehicle GT")
            plt.plot(t, pred, label="TCN")
            plt.plot(t, base, label="Zero-Δv baseline", alpha=.8)
            plt.xlabel("Blackout time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.title(f"Test episode {i + 1}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(DATA / f"tcn_test_episode_{i+1}.png", dpi=150)
            plt.close()
    return result


def print_metrics(label, m):
    print(f"{label}: Δv SmoothL1={m['dv_smoothl1']:.5f}, "
          f"velocity SmoothL1={m['velocity_smoothl1']:.5f}")
    print("  Model checkpoint MAE (m/s):",
          {s: round(m["model_mae"][s], 4) for s in CHECKPOINTS})
    print("  Zero-Δv baseline MAE (m/s):",
          {s: round(m["baseline_mae"][s], 4) for s in CHECKPOINTS})


def main():
    train_all = load_pickle("train_blackout_windows.pkl")
    test_windows = load_pickle("test_blackout_windows.pkl")
    scalers = load_pickle("scalers.pkl")

    # Chronological holdout from the tail of training windows for model selection.
    # Final test set remains untouched until the best checkpoint is selected.
    train_all = sorted(train_all, key=lambda w: w["metadata"]["window_start_sec"])
    cut = max(1, int(.9 * len(train_all)))
    fit_windows, val_windows = train_all[:cut], train_all[cut:]
    if not val_windows:
        raise RuntimeError("Need at least 2 training windows for a validation holdout.")

    train_loader = DataLoader(Episodes(fit_windows, scalers), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Episodes(val_windows, scalers), batch_size=BATCH_SIZE)
    test_loader = DataLoader(Episodes(test_windows, scalers), batch_size=BATCH_SIZE)

    model = TCN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    best_val, best_epoch = float("inf"), 0
    DATA.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for x, dv, seed in train_loader:
            x, dv, seed = x.to(DEVICE), dv.to(DEVICE), seed.to(DEVICE)
            opt.zero_grad()
            pdv = model(x)
            pv = seed[:, None] + pdv.cumsum(1)
            tv = seed[:, None] + dv.cumsum(1)
            loss = loss_fn(pdv, dv) + loss_fn(pv, tv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(x)
            n += len(x)

        val = metrics(model, val_loader)
        score = val["model_mae"][60]
        print(f"Epoch {epoch:02d}/{EPOCHS} train_loss={running/max(n,1):.5f} "
              f"val_60s_MAE={score:.4f} m/s")
        if score < best_val:
            best_val, best_epoch = score, epoch
            torch.save(model.state_dict(), DATA / "tcn_delta_v_best.pt")

    print(f"Best checkpoint: epoch {best_epoch}, validation 60s MAE={best_val:.4f} m/s")
    model.load_state_dict(torch.load(DATA / "tcn_delta_v_best.pt", map_location=DEVICE))
    test = metrics(model, test_loader, make_plots=True)
    print_metrics("FINAL TEST (best validation checkpoint)", test)
    print(f"Saved model and plots under {DATA}/")


if __name__ == "__main__":
    main()
