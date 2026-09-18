"""Diagnose why a trained blackout TCN loses to the zero-Δv baseline.

Run from the project directory containing data/{train,test}_blackout_windows.pkl,
scalers.pkl and tcn_delta_v_best.pt (or pass --model).
"""
import argparse
import pickle
from datetime import datetime
from pathlib import Path
import scipy.signal
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CHECKPOINTS = [2, 5, 10, 15, 30, 60]

class Episodes(Dataset):
    def __init__(self, windows, scalers): self.windows, self.scalers = windows, scalers
    def __len__(self): return len(self.windows)
    def __getitem__(self, i):
        w = self.windows[i]
        x = np.concatenate([self.scalers[k].transform(np.concatenate([w['context'][k], w['blackout'][k]], axis=0))
                            for k in ('raw_accel', 'raw_gyro', 'gravity')], axis=1).astype(np.float32)

        dv = np.asarray(w['ground_truth']['delta_v_ms'], dtype=np.float32)
        seed = np.float32(w['context']['vr_seed_ms'])
        return torch.from_numpy(x.T), torch.from_numpy(dv), torch.tensor(seed), i

class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__(); self.pad=(kernel-1)*dilation; self.conv=nn.Conv1d(cin,cout,kernel,dilation=dilation)
    def forward(self,x): return self.conv(nn.functional.pad(x,(self.pad,0)))
class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__(); self.net=nn.Sequential(CausalConv1d(channels,channels,3,dilation),nn.GELU(),nn.Dropout(.1),CausalConv1d(channels,channels,3,dilation),nn.GELU(),nn.Dropout(.1))
    def forward(self,x): return x+self.net(x)
class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(9, 64, 1)
        self.blocks = nn.Sequential(*(Block(64, d) for d in (1, 2, 4, 8, 16)))
        self.head = nn.Conv1d(64, 1, 1)
    def forward(self, x):
        return self.head(self.blocks(self.input(x))).squeeze(1)

def load(path):
    with open(path,'rb') as f: return pickle.load(f)
def mae(a,b): return float(np.mean(np.abs(a-b))) if len(a) else float('nan')
def corr(a,b):
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data'); ap.add_argument('--model',default=None)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--plots',action='store_true')
    args=ap.parse_args(); root=Path(args.data_dir); model_path=Path(args.model) if args.model else root/'tcn_delta_v_best.pt'
    windows=load(root/'test_blackout_windows.pkl'); scalers=load(root/'scalers.pkl')
    ds=Episodes(windows,scalers); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=TCN().to(device)
    model.load_state_dict(torch.load(model_path,map_location=device)); model.eval()
    all_pd=[]; all_td=[]; all_seed=[]; all_idx=[]; all_x=[]
    with torch.no_grad():
        for x,dv,seed,idx in loader:
            pred=model(x.to(device))[:, -600:].cpu().numpy()
            all_pd.extend(pred); all_td.extend(dv.numpy()); all_seed.extend(seed.numpy()); all_idx.extend(idx.numpy()); all_x.extend(x.numpy())
    # Preserve original dataset order.
    order=np.argsort(all_idx); pdv=np.asarray(all_pd)[order]; tdv=np.asarray(all_td)[order]; seeds=np.asarray(all_seed)[order]
    valid=np.isfinite(seeds)&np.isfinite(tdv).all(1)&np.isfinite(pdv).all(1)
    pdv,tdv,seeds=pdv[valid],tdv[valid],seeds[valid]
    if not len(seeds): raise SystemExit('No test windows with finite seed and targets.')
    pred_v=seeds[:,None]+np.cumsum(pdv,axis=1); true_v=seeds[:,None]+np.cumsum(tdv,axis=1); base_v=np.broadcast_to(seeds[:,None],true_v.shape)
    print(f'Test windows: {len(windows)} | valid: {len(seeds)} | dropped non-finite: {len(windows)-len(seeds)} | device: {device}')
    # Target algebra check, independent of model.
    max_err=[]
    for j,w in enumerate(windows):
        try:
            s=float(w['context']['vr_seed_ms']); d=np.asarray(w['ground_truth']['delta_v_ms']); v=np.asarray(w['ground_truth']['speeds_ms'])
            if np.isfinite(s) and np.isfinite(d).all() and np.isfinite(v).all() and len(d)==len(v): max_err.append(np.max(np.abs(s+np.cumsum(d)-v)))
        except (KeyError,TypeError,ValueError): pass
    print('Target reconstruction max abs error (m/s):', f'{max(max_err):.8g}' if max_err else 'unavailable')
    flatp,flatt=pdv.ravel(),tdv.ravel()
    print('\nPer-step Δv:')
    print(f'  TCN MAE={mae(flatp,flatt):.5f} m/s | zero-Δv MAE={mae(np.zeros_like(flatt),flatt):.5f} m/s | corr={corr(flatp,flatt):.5f}')
    print(f'  true mean/std={np.mean(flatt):.5f}/{np.std(flatt):.5f} | pred mean/std={np.mean(flatp):.5f}/{np.std(flatp):.5f}')
    print('\nVelocity errors by horizon:')
    print('  seconds | TCN MAE | zero-Δv MAE | TCN minus baseline')
    for sec in CHECKPOINTS:
        j=sec*10-1; a=mae(pred_v[:,j],true_v[:,j]); b=mae(base_v[:,j],true_v[:,j])
        print(f'  {sec:>7} | {a:>8.4f} | {b:>12.4f} | {a-b:>+18.4f}')
    # Horizon-local cumulative increments, avoids conflating all earlier accumulated drift.
    print('\nCumulative Δv error from blackout start:')
    for sec in CHECKPOINTS:
        j=sec*10-1; e=np.cumsum(pdv,axis=1)[:,j]-np.cumsum(tdv,axis=1)[:,j]
        print(f'  {sec:>2}s: signed bias={np.mean(e):+.4f} m/s, MAE={mae(e,np.zeros_like(e)):.4f} m/s')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = root / 'tcn_diagnostics' / timestamp
    if args.plots:
        out.mkdir(parents=True, exist_ok=True)
        # Randomly chosen reproducible episodes, not cherry-picked by error.
        rng=np.random.default_rng(42); picks=np.sort(rng.choice(len(seeds),size=min(5,len(seeds)),replace=False))
        t=np.arange(1,pdv.shape[1]+1)/10
        for n,i in enumerate(picks,1):
            fig,ax=plt.subplots(figsize=(10,4)); ax.plot(t,true_v[i],label='Vehicle GT'); ax.plot(t,pred_v[i],label='TCN'); ax.plot(t,base_v[i],label='Zero-Δv',alpha=.8)
            ax.set(xlabel='Blackout time (s)',ylabel='Velocity (m/s)',title=f'Test episode {i}'); ax.legend(); fig.tight_layout(); fig.savefig(out/f'episode_{n}.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(flatt,flatp,s=3,alpha=.25); lim=max(np.max(np.abs(flatt)),np.max(np.abs(flatp)),.1); ax.plot([-lim,lim],[-lim,lim],linestyle='--'); ax.set(xlabel='True Δv (m/s)',ylabel='Predicted Δv (m/s)',title='Per-step Δv: prediction vs target'); fig.tight_layout(); fig.savefig(out/'delta_v_scatter.png',dpi=160); plt.close(fig)
        print(f'Plots saved to {out}/')
if __name__=='__main__': main()
