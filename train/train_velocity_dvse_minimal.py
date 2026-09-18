#!/usr/bin/env python3
"""Train GRU + causal TCN velocity head on existing blackout windows."""
from __future__ import annotations
import argparse, pickle, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from velocity_dvse_model import VelocityDVSE

HZ=10
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pickle(path):
    with Path(path).open("rb") as f: return pickle.load(f)

def sec_features(x):
    """(600,3) -> (60,18), six stats per axis."""
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=3 or x.shape[0]%HZ:
        raise ValueError(f"Expected (N,3), N divisible by {HZ}; got {x.shape}")
    chunks=x.reshape(-1,HZ,3); feats=[]
    for axis in range(3):
        a=chunks[:,:,axis]; mean=a.mean(1); std=a.std(1)
        c=a-mean[:,None]; safe=np.maximum(std,1e-6)
        feats.extend([mean,std,a.max(1),a.min(1),np.sqrt(np.mean(a*a,1)),
                      np.mean(c**3,1)/safe**3])
    return np.stack(feats,axis=-1).astype(np.float32)

def item_from_window(w):
    b,g=w["blackout"],w["ground_truth"]
    x=np.concatenate([sec_features(b["raw_accel"]),sec_features(b["raw_gyro"])],axis=-1)
    v=np.asarray(g["speeds_ms"],dtype=np.float32)
    if v.size!=600: raise ValueError(f"Expected 600 GT speed samples, got {v.shape}")
    return x,float(w["context"]["vr_seed_ms"]),v.reshape(60,HZ)[:,-1]

class Windows(Dataset):
    def __init__(self, windows): self.items=[item_from_window(w) for w in windows]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x,s,y=self.items[i]
        return torch.from_numpy(x),torch.tensor([s],dtype=torch.float32),torch.from_numpy(y)

def normalize(train_items, datasets):
    xx=np.concatenate([x for x,_,_ in train_items],axis=0)
    mu=xx.mean(0).astype(np.float32); sd=xx.std(0).astype(np.float32); sd[sd<1e-6]=1.
    for ds in datasets: ds.items=[((x-mu)/sd,s,y) for x,s,y in ds.items]
    return mu,sd

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); ps=[]; ys=[]
    for x,s,y in loader:
        p,_=model(x.to(DEVICE),s.to(DEVICE))
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    p=np.concatenate(ps); y=np.concatenate(ys); e=np.abs(p-y)
    return float(e.mean()),float(np.quantile(e,.8))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",default="data/train_blackout_windows.pkl")
    ap.add_argument("--test",default="data/test_blackout_windows.pkl")
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=5e-3)
    ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--out",default="models/velocity_gru_tcn.pt")
    args=ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train_all=load_pickle(args.train); test=Windows(load_pickle(args.test))
    # Chronological validation tail. Uses metadata order when available.
    train_all=sorted(train_all,key=lambda w:w.get("metadata",{}).get("window_start_sec",0))
    full=Windows(train_all)
    cut=max(1,int(.8*len(full)))
    if cut>=len(full): raise RuntimeError("Need >=2 training windows for train/validation split")
    fit=Windows.__new__(Windows); fit.items=full.items[:cut]
    val=Windows.__new__(Windows); val.items=full.items[cut:]
    mu,sd=normalize(fit.items,[fit,val,test])
    tl=DataLoader(fit,batch_size=args.batch_size,shuffle=True)
    vl=DataLoader(val,batch_size=args.batch_size)
    testl=DataLoader(test,batch_size=args.batch_size)
    model=VelocityDVSE().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    best=float("inf"); stale=0; best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; seen=0
        for x,s,y in tl:
            x,s,y=x.to(DEVICE),s.to(DEVICE),y.to(DEVICE)
            pred,dv=model(x,s)
            dv_target=torch.cat([y[:,:1]-s,y[:,1:]-y[:,:-1]],dim=1)
            loss=.3*nn.functional.smooth_l1_loss(pred,y)+.7*nn.functional.smooth_l1_loss(dv,dv_target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()*len(x); seen+=len(x)
        vm,vp=evaluate(model,vl)
        print(f"epoch {epoch:03d} train_loss={total/max(seen,1):.5f} val_MAE={vm:.3f} m/s val_P80={vp:.3f} m/s")
        if vm<best:
            best=vm; best_epoch=epoch; stale=0
            torch.save({"model":model.state_dict(),"feature_mean":mu,"feature_std":sd,
                        "best_val_mae":best,"epoch":epoch},out)
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping at epoch {epoch}"); break
    ck=torch.load(out,map_location=DEVICE,weights_only=False)
    model.load_state_dict(ck["model"])
    tm,tp=evaluate(model,testl)
    print(f"Best validation epoch={best_epoch}, val_MAE={best:.3f} m/s")
    print(f"FINAL TEST MAE={tm:.3f} m/s ({tm*3.6:.2f} km/h), P80={tp:.3f} m/s")
    print(f"Saved {out}")

if __name__=="__main__": main()
