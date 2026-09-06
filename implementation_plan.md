# DVSE Two-Stage Training Plan

I have reviewed the Two-Stage Training Strategy. While the conceptual separation of the Noise Network (GRU) and the Motion Transformation Network (TCN) makes sense to isolate pose from noise, there are significant mathematical and pipeline contradictions in the proposed specification that we must resolve before implementation.

## User Review Required

> [!WARNING]
> **Contradiction 1: The Noise Network's Target**
> You specified that in Stage 1, the Noise Network is trained with the target `delta_v_seq` (the full per-second velocity change). 
> However, in Stage 2's inference step, you specify the integration as: `v[t] = v[t-1] + (rotated_accel_integral * forward_unit + N) * dt`. 
> If `N` was trained to perfectly predict `delta_v` in Stage 1, then adding the physics component (`rotated_accel_integral`) to it will double-count the velocity! `N` must either be trained to predict the *residual* error (which is impossible in Stage 1 without the MTN), OR we use the Noise Network to directly predict the final `delta_v` and only use the MTN to calculate heading.

> [!CAUTION]
> **Contradiction 2: Rotated Acceleration Features**
> In Stage 2, you state: *"Noise network takes rotated acceleration features"*.
> `acc_feat` contains statistical features (Standard Deviation, Min, Max, Skew, Kurtosis) calculated from the 100Hz raw IMU data *during* the dataset binning phase. It is mathematically impossible to rotate the pre-computed `acc_feat` matrix. To get "rotated acceleration features", we would need to pass the massive 100Hz raw IMU sequences into the PyTorch DataLoader, rotate them on the GPU per batch, and then compute Skew/Kurtosis on the GPU, which is incredibly inefficient and unsupported by native PyTorch.

## Proposed Resolution

Here is a mathematically sound, pipeline-compatible way to implement the Two-Stage approach using your existing dataset:

### Stage 1: Noise/Residual Proxy Training
We train the GRU (Noise Network) to predict the **direct** `delta_v_seq` using the unrotated `acc_feat` and `gyro_feat`. It becomes an expert at pure pattern-matching for speed changes.
* Loss: `0.7 * SmoothL1(pred_dv, target_dv) + 0.3 * SmoothL1(cumsum(pred_dv), cumsum(target_dv))`
* *Offset Loss*: We will compute the loss at `offset=0` and `offset=1` (1 second), taking the minimum to absorb GPS lag.

### Stage 2: MTN Training with Physics-Residual Formulation
Since `acc_feat` cannot be rotated on the fly, the Noise Network will continue to use the **unrotated** `acc_feat` to predict its term `N`. 
We redefine the relationship in Stage 2 to be a learned interpolation or residual. For example:
`delta_v_final = alpha * v_physics + (1 - alpha) * N`
Where `v_physics` is derived from rotating `raw_accel` (the 1Hz mean vector) using the MTN's predicted Euler angles. 
Since you have ground truth vehicle heading, we will apply a **strong direct supervision loss** on the MTN's Yaw (`euler[..., 2]`), forcing it to learn correct pose independent of the velocity loss.

### Data Augmentation
To simulate arbitrary phone placements, we will apply 3D rotation matrices to `mtn_input` and `raw_accel` during Stage 2 training, and apply the exact inverse rotation to the target `euler` angles.

Please confirm if this proposed resolution aligns with your goals, or if you'd prefer a different mathematical formulation for combining the MTN and Noise Network!
