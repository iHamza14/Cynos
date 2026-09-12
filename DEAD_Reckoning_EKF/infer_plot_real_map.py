import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Try importing contextily for real map tiles, fallback gracefully if not installed
try:
    import contextily as cx
    HAS_CX = True
except ImportError:
    HAS_CX = False
    print("Notice: 'contextily' not installed. Map background won't load. (Run: pip install contextily)")

from train_tcn_ekf import TCN, DifferentiableEKF, BlackoutDataset
from ekf_initialization import process_and_initialize_ekf

def run_inference_on_real_map():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data Paths
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    scalers_path = os.path.join(data_dir, 'scalers.pkl')
    test_data_path = os.path.join(data_dir, 'test_blackout_windows.pkl')
    
    if not os.path.exists(test_data_path):
        test_data_path = os.path.join(data_dir, 'train_blackout_windows.pkl')
        
    dataset = BlackoutDataset(test_data_path, scalers_path)
    sample = dataset[0]
    
    # We need the original ground_truth dictionary for absolute lat/lon
    # Since dataset[0] only returns tensors, we can grab it from dataset.data directly
    raw_gt = dataset.data[0]['ground_truth']
    start_lat = raw_gt['start_lat']
    start_lon = raw_gt['start_lon']
    
    # 2. Initialize EKF Static States
    static_file = '/Users/hamza/SIH/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw01/S-Vw1.csv'
    try:
        static_x, static_P, _ = process_and_initialize_ekf(static_file)
    except Exception:
        static_x = np.zeros(15)
        static_P = np.eye(15) * 1e-3

    # 3. Instantiate Models
    tcn = TCN().to(device)
    ekf = DifferentiableEKF(static_x, static_P).to(device)
    
    weights_path = 'tcn_weights.pth'
    if os.path.exists(weights_path):
        tcn.load_state_dict(torch.load(weights_path, map_location=device))
        
    tcn.eval()
    ekf.eval()
    
    # 4. Prepare Batch Tensors
    vr_seed = sample['vr_seed'].unsqueeze(0).to(device)
    initial_yaw = sample['initial_yaw'].unsqueeze(0).to(device)
    ctx_accel = sample['context_raw_accel'].unsqueeze(0).to(device)
    ctx_gyro = sample['context_raw_gyro'].unsqueeze(0).to(device)
    blk_accel = sample['blackout_raw_accel'].unsqueeze(0).to(device)
    blk_gyro = sample['blackout_raw_gyro'].unsqueeze(0).to(device)
    blk_norm = sample['blackout_norm_imu'].unsqueeze(0).transpose(1, 2).to(device)
    
    # Extract Ground Truth Displacements
    gt_disp = sample['cum_disp_m'].numpy()
    
    # 5. Run Inference
    with torch.no_grad():
        pseudo_vel, r_scale = tcn(blk_norm, vr_seed)
        pred_disp, _ = ekf(vr_seed, initial_yaw, ctx_accel, ctx_gyro, blk_accel, blk_gyro, pseudo_vel, r_scale)
        
    pred_disp = pred_disp.squeeze(0).cpu().numpy()
    
    # 6. Convert Local Displacements (North, East) -> Global Coordinates (Longitude, Latitude)
    R = 6371000.0 # Earth radius in meters
    
    # In helper.py: index 0 is North, index 1 is East
    gt_north, gt_east = gt_disp[:, 0], gt_disp[:, 1]
    pred_north, pred_east = pred_disp[:, 0], pred_disp[:, 1]
    
    # Translate to Lat/Lon degrees
    gt_lat = start_lat + np.degrees(gt_north / R)
    gt_lon = start_lon + np.degrees(gt_east / (R * np.cos(np.radians(start_lat))))
    
    pred_lat = start_lat + np.degrees(pred_north / R)
    pred_lon = start_lon + np.degrees(pred_east / (R * np.cos(np.radians(start_lat))))

    # 7. Plot on Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Note: Longitude is X-axis, Latitude is Y-axis
    ax.plot(gt_lon, gt_lat, label='Ground Truth (GNSS)', color='blue', linewidth=3, alpha=0.8)
    ax.plot(pred_lon, pred_lat, label='Predicted (TCN + EKF)', color='red', linestyle='--', linewidth=3)
    
    # Start and End Markers
    ax.scatter([gt_lon[0]], [gt_lat[0]], color='green', marker='o', s=150, zorder=5, label='Start Point')
    ax.scatter([gt_lon[-1]], [gt_lat[-1]], color='blue', marker='X', s=150, zorder=5, label='GT End')
    ax.scatter([pred_lon[-1]], [pred_lat[-1]], color='red', marker='X', s=150, zorder=5, label='Pred End')
    
    # Contextily mapping
    if HAS_CX:
        # Convert GPS CRS (EPSG:4326) to Web Mercator (EPSG:3857) for tile mapping
        import geopandas as gpd
        from shapely.geometry import Point
        
        # We temporarily convert to GeoSeries to project to Web Mercator
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(gt_lon, gt_lat)] + 
                               [Point(lon, lat) for lon, lat in zip(pred_lon, pred_lat)],
                               crs="EPSG:4326").to_crs(epsg=3857)
        
        # Re-extract coordinates in Web Mercator space for plotting
        mercator_gt_x = [p.x for p in points[:len(gt_lon)]]
        mercator_gt_y = [p.y for p in points[:len(gt_lat)]]
        mercator_pred_x = [p.x for p in points[len(gt_lon):]]
        mercator_pred_y = [p.y for p in points[len(gt_lat):]]
        
        ax.clear()
        
        ax.plot(mercator_gt_x, mercator_gt_y, label='Ground Truth (GNSS)', color='blue', linewidth=3, alpha=0.8)
        ax.plot(mercator_pred_x, mercator_pred_y, label='Predicted (TCN + EKF)', color='red', linestyle='--', linewidth=3)
        ax.scatter([mercator_gt_x[0]], [mercator_gt_y[0]], color='green', marker='o', s=150, zorder=5, label='Start Point')
        ax.scatter([mercator_gt_x[-1]], [mercator_gt_y[-1]], color='blue', marker='X', s=150, zorder=5, label='GT End')
        ax.scatter([mercator_pred_x[-1]], [mercator_pred_y[-1]], color='red', marker='X', s=150, zorder=5, label='Pred End')
        
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Longitude (Web Mercator)")
        ax.set_ylabel("Latitude (Web Mercator)")
    else:
        # Standard Lat/Lon plotting fallback
        # Ensure aspect ratio preserves roughly realistic map appearance based on latitude
        ax.set_aspect(1.0 / np.cos(np.radians(start_lat)))
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.ticklabel_format(useOffset=False, style='plain') # Avoid scientific notation on coords
        
    ax.set_title('Vehicle Trajectory Map: 60s GNSS Blackout', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    save_path = 'map_trajectory_real_map.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Map plot saved successfully as: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    run_inference_on_real_map()
