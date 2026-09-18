"""Minimal TCN baseline for blackout Δv prediction."""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA = Path("data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 60
LR = 5e-3
IN_CHANNELS = 9
CHECKPOINTS = [2, 5, 10, 15, 30, 60]


def load_pickle(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


class Episodes(Dataset):
    def __init__(self, windows, scalers):
        self.windows = windows
        self.scalers = scalers

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        fields = []
        for key in ("raw_accel", "raw_gyro", "gravity"):
            x = w["blackout"][key]
            fields.append(self.scalers[key].transform(x))
        x = np.concatenate(fields, axis=1).astype(np.float32)  # (600, 9)
        dv = w["ground_truth"]["delta_v_ms"].astype(np.float32)
        seed = np.float32(w["context"]["vr_seed_ms"])
        return torch.from_numpy(x.T), torch.from_numpy(dv), torch.tensor(seed)


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, 3, dilation),
            nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, channels, 3, dilation),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.blocks = nn.Sequential(*[
            TCNBlock(64, d) for d in (1, 2, 4, 8, 16, 32, 64)
        ])
        self.head = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        return self.head(self.blocks(self.input(x))).squeeze(1)


def evaluate(model, loader):
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    total_dv, total_v, n = 0.0, 0.0, 0
    checkpoint_errs = {s: [] for s in CHECKPOINTS}
    with torch.no_grad():
        for x, dv, seed in loader:
            x, dv, seed = x.to(DEVICE), dv.to(DEVICE), seed.to(DEVICE)
            pred_dv = model(x)
            pred_v = seed[:, None] + torch.cumsum(pred_dv, dim=1)
            true_v = seed[:, None] + torch.cumsum(dv, dim=1)
            total_dv += loss_fn(pred_dv, dv).item() * len(x)
            total_v += loss_fn(pred_v, true_v).item() * len(x)
            n += len(x)
            for sec in CHECKPOINTS:
                i = sec * 10 - 1
                checkpoint_errs[sec].extend(
                    (pred_v[:, i] - true_v[:, i]).abs().cpu().tolist()
                )
    print(f"Δv SmoothL1={total_dv/max(n,1):.5f} | velocity SmoothL1={total_v/max(n,1):.5f}")
    print("Checkpoint MAE (m/s):", {
        sec: round(float(np.mean(vals)), 4) if vals else None
        for sec, vals in checkpoint_errs.items()
    })


def main():
    train_windows = load_pickle("train_blackout_windows.pkl")
    test_windows = load_pickle("test_blackout_windows.pkl")
    scalers = load_pickle("scalers.pkl")

    train_loader = DataLoader(Episodes(train_windows, scalers), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Episodes(test_windows, scalers), batch_size=BATCH_SIZE)

    model = TCN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, count = 0.0, 0
        for x, dv, seed in train_loader:
            x, dv, seed = x.to(DEVICE), dv.to(DEVICE), seed.to(DEVICE)
            optimizer.zero_grad()
            pred_dv = model(x)
            pred_v = seed[:, None] + torch.cumsum(pred_dv, dim=1)
            true_v = seed[:, None] + torch.cumsum(dv, dim=1)
            loss = loss_fn(pred_dv, dv) + loss_fn(pred_v, true_v)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(x)
            count += len(x)
        print(f"Epoch {epoch:02d}/{EPOCHS} train_loss={running/max(count,1):.5f}")
        evaluate(model, test_loader)

    torch.save(model.state_dict(), DATA / "tcn_delta_v.pt")
    print(f"Saved: {DATA / 'tcn_delta_v.pt'}")


if __name__ == "__main__":
    main()
