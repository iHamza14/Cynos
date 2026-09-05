# DVSE Dead Reckoning Implementation Plan

We will create a completely new, rigorous data pipeline and training loop to support the new dead reckoning requirements. The codebase will be placed in a new directory: `DVSE_DeadReckoning/`.

## User Review Required

> [!IMPORTANT]
> The specification requires calculating vehicle odometry speed from `Wheel Speed Front Left (rad/sec)` and `Right`. This requires knowing the **wheel radius** (`r_wheel`). Since this isn't provided, I will use a standard approximation for a typical car (`0.3 meters`), or do you want me to infer it dynamically by comparing against the GPS `Velocity (km/hr)`? Let me know if you have a specific value.

## Proposed Changes

We will create the following files in `DVSE_DeadReckoning/`:

### 1. `data_pipeline.py`
- Implements `bin_dataset(df_merged)` which:
  - Parses times and nearest-neighbor syncs Mobile to Vehicle (max_gap=0.2s).
  - Groups by `floor(abs_time)` and calculates the 18-dim `acc_feat` and `gyro_feat`.
  - Calculates `v_odo_speed = (Wheel_Left + Wheel_Right) / 2 * r_wheel`.
  - Calculates `v_delta_speed` as the diff of `v_odo_speed`.
- Implements `build_windows(bins, window_sec=60, step_sec=15)` which:
  - Slides across bins to create tensors.
  - Implements the strict GPS quality gate filtering (mobile GPS accuracy, mobile/vehicle sat counts).
  - Computes `target_disp` (haversine formula) in meters for North/East.
  - Saves the resulting train/val/test splits as pickled lists of dictionaries for fast PyTorch loading.

### 2. `dataset.py`
- Implements `DeadReckoningDataset(torch.utils.data.Dataset)`.
- Handles fitting and applying `StandardScaler` on the training set, transforming the `_feat`, `mtn_input`, `raw_accel`, and `gravity` fields (leaving `vr` untouched as requested).

### 3. `models.py`
- Updates the `DVSE` neural architecture signature to precisely match the requested `forward(self, acc_feat, gyro_feat, vr, mtn_input, raw_accel, gravity)`.

### 4. `train.py`
- Implements the custom 3-part loss function:
  1. `loss_dv` (MSE against `delta_v_target`).
  2. `loss_disp` (MSE between Haversine target and integrated path using predicted heading and predicted cumulative velocity).
  3. `loss_euler` (MSE on Yaw against Vehicle Heading).
- Runs the `train_step` using `vr_train` (the clean wheel odometry speed) as Teacher Forcing to stabilize learning.

### 5. `inference.py`
- Implements `inference_step`, initializing `vr` with `vr_seed` (starting mobile GPS speed), and looping autoregressively to compute `vr[t] = vr_seed + cumsum(delta_v)`.

## Verification Plan

### Automated Tests
- Running `python data_pipeline.py` to verify nearest-neighbor joins and window filtering criteria.
- Running `python train.py` for 1 epoch to verify PyTorch tensor dimensions match across the autoregressive loss function without throwing size mismatch errors.

Please approve this implementation plan and let me know how to handle the wheel radius!
