import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import the models and dataset from your training script
from train_tcn_ekf import TCN, DifferentiableEKF, BlackoutDataset
from ekf_initialization import process_and_initialize_ekf

def run_inference_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data Paths
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    scalers_path = os.path.join(data_dir, 'scalers.pkl')
    test_data_path = os.path.join(data_dir, 'test_blackout_windows.pkl')
    
    # Fallback to train data if test data hasn't been generated yet
    if not os.path.exists(test_data_path):
        print(f"Test dataset not found at {test_data_path}. Falling back to train_blackout_windows.pkl...")
        test_data_path = os.path.join(data_dir, 'train_blackout_windows.pkl')
        
    dataset = BlackoutDataset(test_data_path, scalers_path)
    print(f"Loaded dataset with {len(dataset)} blackout windows.")
    
    # Grab the first window for visualization
    sample = dataset[0]
    
    # 2. Initialize EKF Static States
    static_file = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw01/S-Vw1.csv'
    try:
        static_x, static_P, _ = process_and_initialize_ekf(static_file)
    except Exception as e:
        print(f"Warning: Could not load static config, falling back to zero-init. {e}")
        static_x = np.zeros(15)
        static_P = np.eye(15) * 1e-3

    # 3. Instantiate Models
    tcn = TCN().to(device)
    ekf = DifferentiableEKF(static_x, static_P).to(device)
    
    weights_path = 'tcn_weights.pth'
    if os.path.exists(weights_path):
        tcn.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded pre-trained TCN weights.")
    else:
        print("No pre-trained weights found ('tcn_weights.pth'). Running with random initialization (results will be messy).")
        
    tcn.eval()
    ekf.eval()
    
    # 4. Prepare Batch Tensors (unsqueeze adds the Batch dimension = 1)
    vr_seed = sample['vr_seed'].unsqueeze(0).to(device)
    ctx_accel = sample['context_raw_accel'].unsqueeze(0).to(device)
    ctx_gyro = sample['context_raw_gyro'].unsqueeze(0).to(device)
    blk_accel = sample['blackout_raw_accel'].unsqueeze(0).to(device)
    blk_gyro = sample['blackout_raw_gyro'].unsqueeze(0).to(device)
    
    # TCN needs (Batch, Channels, SeqLen)
    blk_norm = sample['blackout_norm_imu'].unsqueeze(0).transpose(1, 2).to(device)
    
    # Extract Ground Truth Map Coordinates
    gt_disp = sample['cum_disp_m'].numpy()
    
    # 5. Run Rollout Inference
    with torch.no_grad():
        pseudo_vel, r_scale = tcn(blk_norm)
        pred_disp, pred_speeds = ekf(
            vr_seed, ctx_accel, ctx_gyro, blk_accel, blk_gyro, pseudo_vel, r_scale
        )
        
    # Remove batch dim and move to CPU for Matplotlib
    pred_disp = pred_disp.squeeze(0).cpu().numpy()
    
    # 6. Plot the Map
    plt.figure(figsize=(10, 10))
    
    gt_x, gt_y = gt_disp[:, 0], gt_disp[:, 1]
    pred_x, pred_y = pred_disp[:, 0], pred_disp[:, 1]
    
    # Trajectories
    plt.plot(gt_x, gt_y, label='Ground Truth (GNSS)', color='blue', linewidth=2.5, alpha=0.8)
    plt.plot(pred_x, pred_y, label='Predicted (TCN + EKF)', color='red', linestyle='--', linewidth=2.5)
    
    # Start and End Markers
    plt.scatter([gt_x[0]], [gt_y[0]], color='green', marker='o', s=150, zorder=5, label='Start Point')
    plt.scatter([gt_x[-1]], [gt_y[-1]], color='blue', marker='X', s=150, zorder=5, label='GT End')
    plt.scatter([pred_x[-1]], [pred_y[-1]], color='red', marker='X', s=150, zorder=5, label='Pred End')
    
    plt.title('Vehicle Map Inference: 60s GNSS Blackout', fontsize=16)
    plt.xlabel('X Displacement (meters)', fontsize=12)
    plt.ylabel('Y Displacement (meters)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Ensure map scaling is 1:1 so corners aren't distorted
    plt.axis('equal') 
    
    # Save & Show
    save_path = 'map_inference_trajectory.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    run_inference_and_plot()
