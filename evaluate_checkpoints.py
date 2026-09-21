import torch
import numpy as np
import pickle
import sys

sys.path.append("train")
from models import DVSEModel

def main():
    device = torch.device("cpu")
    
    with open("data/test_blackout_windows.pkl", "rb") as f:
        test_items = pickle.load(f)
    with open("data/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    model = DVSEModel().to(device)
    model.load_state_dict(torch.load("train/dvse_output/best_dvse.pt", map_location=device, weights_only=False))
    model.eval()
    
    checkpoints = [5, 15, 30, 60]
    model_errors = {cp: [] for cp in checkpoints}
    baseline_errors = {cp: [] for cp in checkpoints}
    
    with torch.no_grad():
        for item in test_items:
            # Full 60s inference leaping 10s at a time
            raw_acc = item["blackout"]["raw_accel"]
            raw_gyro = item["blackout"]["raw_gyro"]
            speeds = item["ground_truth"]["speeds_ms"]
            vr_seed = item["context"]["vr_seed_ms"]
            
            baseline_seed = vr_seed
            
            scaled_acc = scalers["raw_accel"].transform(raw_acc)
            scaled_gyro = scalers["raw_gyro"].transform(raw_gyro)
            
            predicted_60s = []
            
            current_v_seed = vr_seed
            
            for k in range(6): # 6 leaps of 10 seconds
                acc_chunk = scaled_acc[k*100 : (k+1)*100]
                gyro_chunk = scaled_gyro[k*100 : (k+1)*100]
                
                acc_t = torch.tensor(acc_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                gyro_t = torch.tensor(gyro_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                
                # vr_seq and v_0 are exactly the scalar current_v_seed
                v0_t = torch.tensor([[current_v_seed]], dtype=torch.float32).to(device)
                vr_t = torch.full((1, 10, 1), current_v_seed, dtype=torch.float32).to(device)
                
                _, v_pred, _, _ = model(acc_t, gyro_t, vr_t, v_0=v0_t)
                
                v_pred_np = v_pred.numpy()[0]
                predicted_60s.extend(v_pred_np)
                
                # Shift reference velocity for next leap
                current_v_seed = v_pred_np[-1]
                
            predicted_60s = np.array(predicted_60s)
            gt_speeds_1hz = np.array([speeds[(i+1)*10 - 1] for i in range(60)])
            
            for cp in checkpoints:
                idx = cp - 1
                if idx < len(predicted_60s):
                    err = np.abs(predicted_60s[idx] - gt_speeds_1hz[idx])
                    model_errors[cp].append(err)
                    
                    base_err = np.abs(baseline_seed - gt_speeds_1hz[idx])
                    baseline_errors[cp].append(base_err)
                    
    print("\n--- CHECKPOINT EVALUATION (10s LEAPING INFERENCE) ---")
    print(f"{'Time (s)':<10} | {'Model MAE':<15} | {'Baseline MAE':<15} | {'Improvement':<15}")
    print("-" * 65)
    for cp in checkpoints:
        m_mae = np.mean(model_errors[cp])
        b_mae = np.mean(baseline_errors[cp])
        imp = b_mae - m_mae
        print(f"{cp:<10} | {m_mae:<15.2f} | {b_mae:<15.2f} | {imp:+.2f} m/s")

if __name__ == "__main__":
    main()
