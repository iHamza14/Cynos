#!/usr/bin/env python3
"""Plot baselines and model performance on held-out blackout windows."""
import argparse, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from velocity_dvse_model import VelocityDVSE
from train_velocity_dvse_minimal import sec_features

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_pickle(p):
    with Path(p).open("rb") as f: return pickle.load(f)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test",default="data/test_blackout_windows.pkl")
    ap.add_argument("--checkpoint",default="models/velocity_dvse.pt")
    ap.add_argument("--outdir",default="outputs/velocity_eval")
    ap.add_argument("--samples",type=int,default=6)
    args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location=DEVICE,weights_only=False)
    mu=np.asarray(ck["feature_mean"]); sd=np.asarray(ck["feature_std"])
    model=VelocityDVSE().to(DEVICE); model.load_state_dict(ck["model"]); model.eval()
    windows=load_pickle(args.test); xs=[]; seeds=[]; truths=[]
    for w in windows:
        b,g=w["blackout"],w["ground_truth"]
        x=np.concatenate([sec_features(b["raw_accel"]),sec_features(b["raw_gyro"])],axis=-1)
        xs.append(((x-mu)/sd).astype(np.float32))
        seeds.append(float(w["context"]["vr_seed_ms"]))
        truths.append(np.asarray(g["speeds_ms"],dtype=np.float32).reshape(60,10)[:,-1])
    if not windows: raise ValueError("Test pickle is empty")
    X=torch.tensor(np.stack(xs),device=DEVICE)
    S=torch.tensor(np.asarray(seeds,dtype=np.float32)[:,None],device=DEVICE)
    with torch.no_grad(): pred,_=model(X,S)
    pred=pred.cpu().numpy(); truth=np.stack(truths); seeds=np.asarray(seeds)
    err=pred-truth; ae=np.abs(err); t=np.arange(1,61)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    # MAE at requested elapsed blackout times (cumulative through each second).
    checkpoints=[2,5,10,15,30,45,60]
    checkpoint_mae={sec: float(ae[:, sec-1].mean()) for sec in checkpoints}
    checkpoint_cum_mae={sec: float(ae[:, :sec].mean()) for sec in checkpoints}

    # Save a compact CSV with point-in-time and cumulative horizon MAE.
    with (out/"mae_at_blackout_timesteps.csv").open("w") as f:
        f.write("blackout_seconds,point_mae_mps,cumulative_mae_mps\\n")
        for sec in checkpoints:
            f.write(f"{sec},{checkpoint_mae[sec]:.6f},{checkpoint_cum_mae[sec]:.6f}\\n")

    # Plot both instantaneous (at t) and cumulative MAE through t.
    point_mae=ae.mean(axis=0)
    cumulative_mae=np.array([ae[:, :sec].mean() for sec in t])
    plt.figure(figsize=(9,5))
    plt.plot(t,point_mae,label="MAE at exact second")
    plt.plot(t,cumulative_mae,label="Cumulative MAE through second")
    for sec in checkpoints:
        plt.scatter([sec],[checkpoint_mae[sec]],s=22)
    plt.xlabel("Seconds into blackout")
    plt.ylabel("Absolute speed error (m/s)")
    plt.title("MAE at blackout timesteps")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(out/"mae_at_blackout_timesteps.png",dpi=160); plt.close()
    # Baselines: constant seed and training-independent test-window mean are diagnostic only.
    seed_pred=np.repeat(seeds[:,None],60,axis=1)
    baseline_mae=np.mean(np.abs(seed_pred-truth))
    model_mae=np.mean(ae)
    # 1: horizon curves, model vs constant-seed baseline
    plt.figure(figsize=(9,5))
    plt.plot(t,ae.mean(0),label="Model MAE")
    plt.plot(t,np.abs(seed_pred-truth).mean(0),label="Constant seed MAE")
    plt.plot(t,np.quantile(ae,.8,axis=0),label="Model P80 |error|")
    plt.xlabel("Seconds into blackout"); plt.ylabel("Absolute speed error (m/s)")
    plt.title("Error growth over blackout horizon"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(out/"error_vs_time.png",dpi=160); plt.close()
    # 2: trajectory examples
    n=min(max(1,args.samples),len(truth)); fig,axs=plt.subplots(n,1,figsize=(11,2.6*n),sharex=True,squeeze=False)
    for i in range(n):
        ax=axs[i,0]; ax.plot(t,truth[i],label="Vehicle GT",lw=1.8)
        ax.plot(t,pred[i],label="Model",lw=1.5); ax.plot(t,seed_pred[i],"--",label="Constant seed")
        ax.set_ylabel("m/s"); ax.set_title(f"Window {i}: model MAE {ae[i].mean():.2f} m/s")
        ax.grid(alpha=.3)
        if i==0: ax.legend()
    axs[-1,0].set_xlabel("Seconds into blackout"); fig.tight_layout()
    fig.savefig(out/"trajectory_examples.png",dpi=160); plt.close(fig)
    # 3: scatter
    plt.figure(figsize=(6.5,6)); plt.scatter(truth.ravel(),pred.ravel(),s=7,alpha=.25)
    lo=min(float(truth.min()),float(pred.min())); hi=max(float(truth.max()),float(pred.max()))
    plt.plot([lo,hi],[lo,hi],"--",label="Ideal"); plt.xlabel("Vehicle GT (m/s)")
    plt.ylabel("Predicted (m/s)"); plt.title("Predicted vs actual"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(out/"predicted_vs_actual.png",dpi=160); plt.close()
    # 4: signed error distribution
    plt.figure(figsize=(8,5)); plt.hist(err.ravel(),bins=60)
    plt.xlabel("Prediction error (m/s)"); plt.ylabel("Count"); plt.title("Signed error distribution")
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"error_distribution.png",dpi=160); plt.close()
    # 5: per-window MAE, model and baseline
    wm=np.mean(ae,axis=1); wb=np.mean(np.abs(seed_pred-truth),axis=1)
    plt.figure(figsize=(9,5)); plt.hist(wm,bins=30,alpha=.7,label="Model")
    plt.hist(wb,bins=30,alpha=.5,label="Constant seed"); plt.xlabel("Per-window MAE (m/s)")
    plt.ylabel("Windows"); plt.title("Per-window error comparison"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out/"per_window_mae.png",dpi=160); plt.close()
    # Metrics
    print(f"Test windows: {len(truth)}")
    print(f"Model MAE: {model_mae:.4f} m/s ({model_mae*3.6:.2f} km/h)")
    print(f"Constant-seed MAE: {baseline_mae:.4f} m/s ({baseline_mae*3.6:.2f} km/h)")
    print(f"Model RMSE: {np.sqrt(np.mean(err**2)):.4f} m/s")
    print(f"Model bias: {err.mean():+.4f} m/s")
    print(f"Model P80 absolute error: {np.quantile(ae,.8):.4f} m/s")
    print(f"60s MAE: {ae[:,-1].mean():.4f} m/s")
    print("\\nMAE by blackout timestep:")
    print(" sec | exact-second MAE | cumulative MAE")
    for sec in checkpoints:
        print(f"{sec:>4} | {checkpoint_mae[sec]:>16.4f} | {checkpoint_cum_mae[sec]:>14.4f}")
    print(f"Plots and CSV: {out.resolve()}")
if __name__=="__main__": main()
