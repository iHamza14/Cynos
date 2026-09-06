import torch
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
from inference import inference_step
import folium
import numpy as np
import pickle

def create_trajectory_map():
    print("Loading Test Dataset and Scalers...")
    test_dataset = DeadReckoningDataset('data/test_windows.pkl', 'data/scalers.pkl')
    # Use batch size 1 so we can easily plot individual windows
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Loading DVSE Dead Reckoning Model...")
    model = DVSE()
    model.load_state_dict(torch.load('checkpoints/dvse_dead_reckoning.pth', weights_only=True))
    model.eval()
    
    # We will pick a few contiguous windows from the test set to plot
    # step_sec is 15, window_sec is 60. So jumping by 4 indices gives contiguous 60s blocks.
    num_windows_to_plot = 5
    indices_to_plot = [i * 4 for i in range(num_windows_to_plot) if i * 4 < len(test_dataset)]
    
    # Initialize Folium Map (will center on the first window)
    m = None
    R_earth = 6371000.0 # meters
    
    print(f"Generating Folium Map for {len(indices_to_plot)} consecutive 60s test windows...")
    
    with torch.no_grad():
        for plot_idx, dataset_idx in enumerate(indices_to_plot):
            batch = test_dataset[dataset_idx]
            
            # Add batch dimension
            acc_t = batch[0].unsqueeze(0)
            gyro_t = batch[1].unsqueeze(0)
            mtn_t = batch[2].unsqueeze(0)
            raw_t = batch[3].unsqueeze(0)
            grav_t = batch[4].unsqueeze(0)
            vr_train = batch[5].unsqueeze(0)
            vr_seed = batch[6].unsqueeze(0)
            delta_v_seq = batch[7].unsqueeze(0)
            target_disp = batch[8].unsqueeze(0)
            heading_target = batch[9].unsqueeze(0)
            
            # Predict
            delta_v_pred, euler_pred = inference_step(model, acc_t, gyro_t, mtn_t, raw_t, grav_t, vr_seed)
            
            # Extract predicted sequence
            v_cum_pred = (vr_seed + torch.cumsum(delta_v_pred.squeeze(-1), dim=1))[0].numpy()
            yaw_pred_rad = euler_pred[0, :, 2].numpy()
            
            # Extract ground truth sequence (using odometry and vehicle heading)
            v_cum_gt = vr_train[0, :, 0].numpy()
            yaw_gt_rad = heading_target[0, :].numpy()
            
            # We need the start coordinates from the original pickle to anchor the plot
            with open('data/test_windows.pkl', 'rb') as f:
                windows = pickle.load(f)
            win_data = windows[dataset_idx]
            
            start_lat = win_data['start_lat']
            start_lon = win_data['start_lon']
            
            if m is None:
                m = folium.Map(location=[start_lat, start_lon], zoom_start=16, tiles="OpenStreetMap")
            
            # Integrate Paths
            gt_coords = [(start_lat, start_lon)]
            pred_coords = [(start_lat, start_lon)]
            
            cur_lat_gt = start_lat
            cur_lon_gt = start_lon
            
            cur_lat_pred = start_lat
            cur_lon_pred = start_lon
            
            for t in range(len(v_cum_pred)):
                # GT Step
                dn_gt = v_cum_gt[t] * np.cos(yaw_gt_rad[t])
                de_gt = v_cum_gt[t] * np.sin(yaw_gt_rad[t])
                cur_lat_gt += np.degrees(dn_gt / R_earth)
                cur_lon_gt += np.degrees(de_gt / (R_earth * np.cos(np.radians(cur_lat_gt))))
                gt_coords.append((cur_lat_gt, cur_lon_gt))
                
                # Pred Step
                dn_pr = v_cum_pred[t] * np.cos(yaw_pred_rad[t])
                de_pr = v_cum_pred[t] * np.sin(yaw_pred_rad[t])
                cur_lat_pred += np.degrees(dn_pr / R_earth)
                cur_lon_pred += np.degrees(de_pr / (R_earth * np.cos(np.radians(cur_lat_pred))))
                pred_coords.append((cur_lat_pred, cur_lon_pred))
                
            # Plot Ground Truth (Blue)
            folium.PolyLine(
                locations=gt_coords,
                color="blue",
                weight=5,
                opacity=0.8,
                tooltip=f"GT Window {plot_idx+1}"
            ).add_to(m)
            
            # Plot Predicted (Red)
            folium.PolyLine(
                locations=pred_coords,
                color="red",
                weight=5,
                opacity=0.8,
                dash_array='10',
                tooltip=f"DVSE Predicted Window {plot_idx+1}"
            ).add_to(m)
            
            # Add markers for start and end
            folium.CircleMarker(location=[start_lat, start_lon], radius=4, color='green', fill=True, tooltip="Start").add_to(m)
            folium.CircleMarker(location=gt_coords[-1], radius=4, color='blue', fill=True, tooltip="GT End").add_to(m)
            folium.CircleMarker(location=pred_coords[-1], radius=4, color='red', fill=True, tooltip="Pred End").add_to(m)

    if m is not None:
        m.save('trajectory_map.html')
        print("Map successfully saved to trajectory_map.html!")
    else:
        print("Not enough test windows to plot.")

if __name__ == '__main__':
    create_trajectory_map()
