import torch
from torch.utils.data import DataLoader
from dataset import DeadReckoningDataset
from models import DVSE
from inference import inference_step

import folium
import numpy as np
import pickle


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

R_EARTH = 6371000.0  # meters
DT = 1.0              # seconds


# ============================================================
# Helpers
# ============================================================

def to_device(x):
    """Move tensor to the selected device."""
    if torch.is_tensor(x):
        return x.to(DEVICE, non_blocking=True)
    return x


def integrate_trajectory(
    start_lat,
    start_lon,
    velocity,
    yaw_rad,
    dt=DT
):
    """
    Integrate velocity + heading into latitude/longitude.

    velocity : (T,)
    yaw_rad  : (T,)
    """

    cur_lat = float(start_lat)
    cur_lon = float(start_lon)

    coords = [(cur_lat, cur_lon)]

    for t in range(len(velocity)):

        # Velocity components in local North/East frame
        dn = velocity[t] * np.cos(yaw_rad[t]) * dt
        de = velocity[t] * np.sin(yaw_rad[t]) * dt

        # Update latitude first
        cur_lat += np.degrees(
            dn / R_EARTH
        )

        # Prevent numerical problems near poles
        cos_lat = np.cos(
            np.radians(cur_lat)
        )

        cos_lat = max(
            abs(cos_lat),
            1e-8
        )

        # Update longitude
        cur_lon += np.degrees(
            de / (R_EARTH * cos_lat)
        )

        coords.append(
            (cur_lat, cur_lon)
        )

    return coords


# ============================================================
# Main
# ============================================================

def create_trajectory_map():
    print("=" * 60)
    print("DVSE Dead Reckoning Trajectory Visualization")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    test_pkl = 'data/test_blackout_windows.pkl'
    scaler_pkl = 'data/scalers.pkl'

    print(f"Loading Test Dataset from {test_pkl}...")
    test_dataset = DeadReckoningDataset(test_pkl, scaler_pkl)
    print(f"Test windows: {len(test_dataset)}")

    print("Loading test window metadata...")
    import pickle
    with open(test_pkl, 'rb') as f:
        windows = pickle.load(f)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading DVSE Dead Reckoning Model...")
    model = DVSE()
    checkpoint_path = 'checkpoints/dvse_dead_reckoning.pth'
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    # --------------------------------------------------------
    # Select windows
    # --------------------------------------------------------

    num_windows_to_plot = 50
    # Assuming test step is 30s for a 60s window (step_sec=30). 
    # To get non-overlapping windows of 60s, we jump by 60/30 = 2 indices.
    indices_to_plot = [i * 2 for i in range(num_windows_to_plot) if i * 2 < len(test_dataset)]

    m = None
    print(f"\nGenerating map for {len(indices_to_plot)} test windows...\n")

    with torch.no_grad():
        for plot_idx, dataset_idx in enumerate(indices_to_plot):
            print(f"Processing window {plot_idx + 1}/{len(indices_to_plot)} (dataset index {dataset_idx})")
            batch = test_dataset[dataset_idx]

            (
                acc_t, gyro_t, mtn_t, raw_t, grav_t,
                vr_train, vr_seed, delta_v_seq, target_disp,
                heading_target, raw_gyro_t, start_heading_t
            ) = batch

            # Add batch dimension
            acc_t = acc_t.unsqueeze(0)
            gyro_t = gyro_t.unsqueeze(0)
            mtn_t = mtn_t.unsqueeze(0)
            raw_t = raw_t.unsqueeze(0)
            grav_t = grav_t.unsqueeze(0)

            vr_train = vr_train.unsqueeze(0)
            vr_seed = vr_seed.unsqueeze(0)
            delta_v_seq = delta_v_seq.unsqueeze(0)
            target_disp = target_disp.unsqueeze(0)
            heading_target = heading_target.unsqueeze(0)
            raw_gyro_t = raw_gyro_t.unsqueeze(0)

            # ------------------------------------------------
            # Move everything to CUDA
            # ------------------------------------------------

            acc_t = to_device(acc_t)
            gyro_t = to_device(gyro_t)
            mtn_t = to_device(mtn_t)
            raw_t = to_device(raw_t)
            grav_t = to_device(grav_t)

            vr_train = to_device(vr_train)
            vr_seed = to_device(vr_seed)
            delta_v_seq = to_device(delta_v_seq)
            target_disp = to_device(target_disp)
            heading_target = to_device(heading_target)
            raw_gyro_t = to_device(raw_gyro_t)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            delta_v_pred, euler_pred = inference_step(
                model,
                acc_t,
                gyro_t,
                mtn_t,
                raw_t,
                grav_t,
                vr_seed
            )

            # ------------------------------------------------
            # Convert predictions to NumPy
            # ------------------------------------------------

            v_cum_pred = (
                vr_seed
                + torch.cumsum(
                    delta_v_pred.squeeze(-1),
                    dim=1
                )
            )

            v_cum_pred = (
                v_cum_pred[0]
                .detach()
                .cpu()
                .numpy()
            )

            win_data = windows[dataset_idx]
            # Relative yaw integration
            start_heading = start_heading_t[0].item()

            gyro_yaw_rate = (
                raw_gyro_t[0, :, 2]
                .detach()
                .cpu()
                .numpy()
            )

            yaw_pred_rad = np.zeros(len(v_cum_pred))
            yaw_pred_rad[0] = start_heading
            for t in range(1, len(v_cum_pred)):
                yaw_pred_rad[t] = yaw_pred_rad[t - 1] + gyro_yaw_rate[t - 1]

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------
            v_cum_gt = (
                vr_train.squeeze(0)[:, 0].numpy()
            )
            yaw_gt_rad = (
                heading_target[0]
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Start coordinates
            # ------------------------------------------------

            win_data = windows[dataset_idx]

            start_lat = float(
                win_data["ground_truth"]["start_lat"]
            )

            start_lon = float(
                win_data["ground_truth"]["start_lon"]
            )

            print(
                f"Start: "
                f"{start_lat:.6f}, "
                f"{start_lon:.6f}"
            )

            # ------------------------------------------------
            # Initialize map
            # ------------------------------------------------

            if m is None:

                m = folium.Map(
                    location=[
                        start_lat,
                        start_lon
                    ],
                    zoom_start=16,
                    tiles="OpenStreetMap"
                )

            # ------------------------------------------------
            # Integrate trajectories
            # ------------------------------------------------

            gt_coords = integrate_trajectory(
                start_lat,
                start_lon,
                v_cum_gt,
                yaw_gt_rad
            )

            pred_coords = integrate_trajectory(
                start_lat,
                start_lon,
                v_cum_pred,
                yaw_pred_rad
            )

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            folium.PolyLine(
                locations=gt_coords,
                color="blue",
                weight=5,
                opacity=0.8,
                tooltip=(
                    f"GT Window {plot_idx + 1}"
                )
            ).add_to(m)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            folium.PolyLine(
                locations=pred_coords,
                color="red",
                weight=5,
                opacity=0.8,
                dash_array="10",
                tooltip=(
                    f"DVSE Predicted "
                    f"Window {plot_idx + 1}"
                )
            ).add_to(m)

            # ------------------------------------------------
            # Start marker
            # ------------------------------------------------

            folium.CircleMarker(
                location=[
                    start_lat,
                    start_lon
                ],
                radius=4,
                color="green",
                fill=True,
                tooltip="Start"
            ).add_to(m)

            # ------------------------------------------------
            # Ground truth endpoint
            # ------------------------------------------------

            folium.CircleMarker(
                location=gt_coords[-1],
                radius=4,
                color="blue",
                fill=True,
                tooltip="GT End"
            ).add_to(m)

            # ------------------------------------------------
            # Prediction endpoint
            # ------------------------------------------------

            folium.CircleMarker(
                location=pred_coords[-1],
                radius=4,
                color="red",
                fill=True,
                tooltip="Pred End"
            ).add_to(m)

            # ------------------------------------------------
            # Print endpoint error
            # ------------------------------------------------

            gt_end = np.array(
                gt_coords[-1]
            )

            pred_end = np.array(
                pred_coords[-1]
            )

            lat_error = (
                pred_end[0] - gt_end[0]
            )

            lon_error = (
                pred_end[1] - gt_end[1]
            )

            print(
                f"GT End:   "
                f"{gt_end[0]:.6f}, "
                f"{gt_end[1]:.6f}"
            )

            print(
                f"Pred End: "
                f"{pred_end[0]:.6f}, "
                f"{pred_end[1]:.6f}"
            )

            print(
                f"Lat error: "
                f"{lat_error:.6f} deg"
            )

            print(
                f"Lon error: "
                f"{lon_error:.6f} deg"
            )

    # --------------------------------------------------------
    # Save map
    # --------------------------------------------------------

    if m is not None:

        output_path = (
            "trajectory_map.html"
        )

        m.save(output_path)

        print("\n" + "=" * 60)
        print(
            f"Map successfully saved to "
            f"{output_path}"
        )
        print("=" * 60)

    else:

        print(
            "No trajectories were generated."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    create_trajectory_map()