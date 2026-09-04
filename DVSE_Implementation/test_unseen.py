import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import DVSE
from dataset import DVSEDataset
from data_sync import get_cols, extract_features

def prepare_unseen_data(s_path, v_path, out_file):
    print(f"Loading unseen datasets: {s_path}")
    df_s = pd.read_csv(s_path, encoding='latin-1')
    
    df_s.columns = df_s.columns.str.strip()
    
    s_date_col = get_cols(df_s, 'DATE')[0]
    def parse_s_time(date_str):
        try:
            time_part = str(date_str).strip().split(' ')[1]
            h, m, s, ms = map(float, time_part.split(':'))
            return h * 3600 + m * 60 + s + ms/1000.0
        except:
            return np.nan
            
    df_s['UTC_TIME'] = df_s[s_date_col].apply(parse_s_time)
    df_s.dropna(subset=['UTC_TIME'], inplace=True)
    
    s_speed_col = get_cols(df_s, 'GPS SPEED')[0]
    df_s['GPS_SPEED_MS'] = df_s[s_speed_col]
    
    df_s['sec_bin'] = np.floor(df_s['UTC_TIME']).astype(int)
    
    acc_cols = get_cols(df_s, 'ACCELEROMETER')[:3]
    gyro_cols = get_cols(df_s, 'GYROSCOPE')[:3]
    grav_cols = get_cols(df_s, 'GRAVITY')[:3]
    
    windows = []
    
    print("Generating 1-second windows for unseen test dataset...")
    for sec, group in df_s.groupby('sec_bin'):
        if len(group) < 5:
            continue
            
        acc_data = group[acc_cols].values
        gyro_data = group[gyro_cols].values
        grav_data = group[grav_cols].values
        
        acc_feat = extract_features(acc_data)
        gyro_feat = extract_features(gyro_data)
        
        dt = 1.0 / len(group)
        int_acc = np.sum(acc_data, axis=0) * dt
        avg_grav = np.mean(grav_data, axis=0)
        avg_acc = np.mean(acc_data, axis=0)
        
        mtn_in = int_acc.tolist() + avg_grav.tolist()
        gps_speed = float(group['GPS_SPEED_MS'].mean())
        
        windows.append({
            'sec': sec,
            'acc_feat': acc_feat,
            'gyro_feat': gyro_feat,
            'mtn_in': mtn_in,
            'raw_acc': avg_acc.tolist(),
            'grav': avg_grav.tolist(),
            'gps_speed': gps_speed
        })
        
    win_df = pd.DataFrame(windows).sort_values('sec').reset_index(drop=True)
    win_df['delta_v_target'] = win_df['gps_speed'].diff().fillna(0)
    
    win_df.to_csv(out_file, index=False)
    print(f"Saved {len(win_df)} windows to {out_file} (100% used for testing)")

def evaluate_unseen(data_path, plot_name="dvse_unseen_plot.png"):
    dataset = DVSEDataset(data_path, seq_len=10)
    # Batch size 1 for sequential prediction
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    model = DVSE()
    if os.path.exists('dvse_model.pth'):
        model.load_state_dict(torch.load('dvse_model.pth', weights_only=True))
        print("Loaded trained model: dvse_model.pth")
    else:
        print("Model file 'dvse_model.pth' not found!")
        return

    model.eval()
    all_pred_v = []
    all_target_v = []
    
    print(f"Evaluating {len(dataset)} unseen sequences...")
    with torch.no_grad():
        for batch in test_loader:
            acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav, target_dv, target_v = batch
            pred_dv, euler = model(acc_feat, gyro_feat, vr, mtn_in, raw_acc, grav)
            
            pred_v = torch.zeros_like(pred_dv)
            pred_v[:, 0] = target_v[:, 0]
            for t in range(1, pred_dv.size(1)):
                pred_v[:, t] = pred_v[:, t-1] + pred_dv[:, t]
                
            all_pred_v.append(pred_v[0, -1, 0].item())
            all_target_v.append(target_v[0, -1, 0].item())

    all_pred_v = np.array(all_pred_v)
    all_target_v = np.array(all_target_v)
    
    errors = np.abs(all_pred_v - all_target_v)
    mae = np.mean(errors)
    p80 = np.percentile(errors, 80)
    
    print("\n=== DVSE Cross-Driver Evaluation Results (Unseen S4) ===")
    print(f"Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"Velocity P80 Error: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)")
    
    def avg_distance_error(pred, target, frame_size):
        if len(pred) < frame_size: return None
        errors = []
        for i in range(0, len(pred) - frame_size + 1, frame_size):
            dist_pred = np.sum(pred[i:i+frame_size])
            dist_tgt = np.sum(target[i:i+frame_size])
            errors.append(np.abs(dist_pred - dist_tgt))
        return np.mean(errors)

    err_30s = avg_distance_error(all_pred_v, all_target_v, 30)
    err_60s = avg_distance_error(all_pred_v, all_target_v, 60)
    
    if err_30s is not None:
        print(f"Average Distance Error (30-sec timeframe): {err_30s:.2f} meters")
    if err_60s is not None:
        print(f"Average Distance Error (60-sec timeframe): {err_60s:.2f} meters")
        
    # --- Logging ---
    import datetime
    log_file = "evaluation_log.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Tested on: {data_path} (Unseen Driver A S4)\n")
        f.write(f"  - Velocity MAE: {mae:.4f} m/s ({mae * 3.6:.2f} km/h)\n")
        f.write(f"  - Velocity P80: {p80:.4f} m/s ({p80 * 3.6:.2f} km/h)\n")
        f.write(f"  - Avg Distance Error (30s): {err_30s:.2f} m\n" if err_30s else "  - Avg Distance Error (30s): N/A\n")
        f.write(f"  - Avg Distance Error (60s): {err_60s:.2f} m\n" if err_60s else "  - Avg Distance Error (60s): N/A\n")
        f.write("-" * 50 + "\n")
    print(f"Results appended to {log_file}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(all_target_v * 3.6, label='GPS Ground Truth', color='blue', linewidth=2)
    plt.plot(all_pred_v * 3.6, label='DVSE Predicted Speed', color='red', linestyle='dashed', linewidth=2)
    plt.title("Zero-Shot Cross-Driver Generalization (Driver E Model -> Driver A S4)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Vehicle Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name)
    print(f"Saved generalization plot to '{plot_name}'")

if __name__ == '__main__':
    s_path = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/S-S4.csv'
    v_path = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/V-S4.csv'
    out_file = 'data/unseen_s4.csv'
    
    if not os.path.exists(out_file):
        os.makedirs('data', exist_ok=True)
        prepare_unseen_data(s_path, v_path, out_file)
    
    evaluate_unseen(out_file)
