import os
import torch
import numpy as np
import folium

from train_tcn_ekf import TCN, DifferentiableEKF, BlackoutDataset
from ekf_initialization import process_and_initialize_ekf

def generate_html_map():
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
    
    # Grab origin lat/lon directly from the dataset dictionary
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
    
    # 6. Convert Local Displacements (North, East) -> Global Coordinates (Latitude, Longitude)
    R = 6371000.0 # Earth radius in meters
    
    gt_north, gt_east = gt_disp[:, 0], gt_disp[:, 1]
    pred_north, pred_east = pred_disp[:, 0], pred_disp[:, 1]
    
    # Arrays of Latitudes and Longitudes
    gt_lat = start_lat + np.degrees(gt_north / R)
    gt_lon = start_lon + np.degrees(gt_east / (R * np.cos(np.radians(start_lat))))
    
    pred_lat = start_lat + np.degrees(pred_north / R)
    pred_lon = start_lon + np.degrees(pred_east / (R * np.cos(np.radians(start_lat))))
    
    # Zip together for Folium, which expects lists of [lat, lon]
    gt_coords = list(zip(gt_lat, gt_lon))
    pred_coords = list(zip(pred_lat, pred_lon))

    # 7. Generate Interactive Folium HTML Map
    # Center map on the start point
    m = folium.Map(location=[start_lat, start_lon], zoom_start=18, tiles='OpenStreetMap')
    
    # Add Ground Truth trajectory (Solid Blue Line)
    folium.PolyLine(
        gt_coords, 
        color='blue', 
        weight=4, 
        opacity=0.8, 
        tooltip='Ground Truth (GNSS)'
    ).add_to(m)
    
    # Add Predicted trajectory (Dashed/Dotted Red Line)
    folium.PolyLine(
        pred_coords, 
        color='red', 
        weight=4, 
        opacity=0.8, 
        dash_array='10, 10', 
        tooltip='Predicted (TCN + EKF)'
    ).add_to(m)
    
    # Add Markers
    # Start Point
    folium.CircleMarker(
        location=[start_lat, start_lon],
        radius=8,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=1.0,
        popup='Start Point'
    ).add_to(m)
    
    # Ground Truth End Point
    folium.Marker(
        location=[gt_lat[-1], gt_lon[-1]],
        icon=folium.Icon(color='blue', icon='info-sign'),
        popup='GT End Point'
    ).add_to(m)
    
    # Prediction End Point
    folium.Marker(
        location=[pred_lat[-1], pred_lon[-1]],
        icon=folium.Icon(color='red', icon='info-sign'),
        popup='Prediction End Point'
    ).add_to(m)
    
    # 8. Save HTML File
    html_file = 'trajectory_map.html'
    m.save(html_file)
    print(f"Interactive HTML map generated successfully! Open '{html_file}' in your web browser.")

if __name__ == '__main__':
    try:
        generate_html_map()
    except ImportError as e:
        print("Required library missing. Please install folium by running: pip install folium")
